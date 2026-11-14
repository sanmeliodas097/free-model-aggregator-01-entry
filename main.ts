// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// The server reads the brief at $MAKEFUN_PROMPT_FILE, probes each provider's catalog
// route in the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header
//      it got: x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions
//      and POST /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//      A process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts;
// the register body's catalog must be that same document. Build it once and use the one object.
//
// ROUTING (quality is 60% of the score): for each request, every free model that meets the
// request's requirements is ranked by the quality table's rating_milli, best first, and the
// best is tried first. A paid model is never a legal pick. A JSON-mode request never routes
// to an anthropic-wire provider: that wire cannot carry response_format.
//
// FAILOVER (10%): a provider call that answers 402, 429 or a 5xx (or cannot be reached) is
// not the answer. The error message tells the two kinds apart: a provider whose free tier
// is spent or out says so ("budget is spent", "is out") and is marked dead for the trial,
// while a scheduled refusal ("refused this call") is transient, so the same provider is
// retried once — the next call number usually succeeds. Three consecutive failures also
// mark a provider dead. The request is then retried on the next-best free model that
// meets it. One provider serves one call at a time here because requests arrive one at a
// time and every upstream call is awaited before the next begins.
//
// THE WIRES: all three upstream wires are implemented (openai, anthropic, gemini), in both
// directions, JSON and streaming. Usage is always the provider's own numbers carried into
// the client wire's shape; stop reasons map by family (stop / end_turn / STOP).
// Streaming to an openai-wire provider sets stream_options.include_usage, or the provider
// sends no usage chunk and the usage check fails.
//
// DETERMINISM: no wall clock and no unseeded randomness. Every choice is a pure function
// of the brief, the requests and the providers' answers so far, so a replay leaves the
// same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicStreamToOpenAIChunks,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  anthropicToGeminiRequest,
  anthropicModelList,
  geminiStreamToAnthropicEvents,
  geminiStreamToOpenAIChunks,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIModelList,
  openAIStreamToAnthropicEvents,
  openAIToAnthropicRequest,
  openAIToAnthropicResponse,
  openAIToGeminiRequest,
  pipeStream,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** One provider's health this trial: consecutive failures, and whether it is spent for good. */
interface ProviderState {
  fails: number;
  dead: boolean;
}
const providerState = new Map<string, ProviderState>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { fails: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

/** A provider answered: its failure streak resets. */
function noteSuccess(providerId: string): void {
  stateOf(providerId).fails = 0;
}

/** A provider refused: count it, and mark it dead when the refusal says spent/out, or on 3 in a row. */
function noteFailure(providerId: string, kind: 'dead' | 'transient' | 'fatal'): void {
  const state = stateOf(providerId);
  state.fails += 1;
  if (kind === 'dead' || state.fails >= 3) state.dead = true;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, deduped by id: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request, best rating first. Ties break by provider id
 * then model id, so the order is a pure function of the brief. A JSON-mode request excludes
 * anthropic-wire providers: their wire cannot carry response_format.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (jsonMode && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** Whether this client request asks for JSON mode (OpenAI wire only: response_format). */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendRaw(res: ServerResponse, status: number, text: string, contentType: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The provider's error message, when its error body carries one. */
function errorMessageOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
  } catch {
    // Not JSON: the raw text is the message.
  }
  return text;
}

/**
 * What kind of refusal this is: 'dead' when the provider says its free tier is spent or it is
 * out (never worth another call this trial), 'fatal' for a bad request or a bad key (retrying
 * the same provider cannot help), 'transient' for anything else (a scheduled refusal: the
 * next call number usually succeeds, so one retry is worth it).
 */
function classifyFailure(status: number, errorText: string): 'dead' | 'transient' | 'fatal' {
  const message = errorMessageOf(errorText);
  if (message.includes('budget is spent') || message.trimEnd().endsWith('is out')) return 'dead';
  if (status === 400 || status === 401) return 'fatal';
  return 'transient';
}

/** One request to the routed provider, in the provider's own wire. */
async function fetchUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Response> {
  const provider = candidate.probe.provider;
  const model = candidate.model.id;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    if (stream) {
      upstreamBody.stream = true;
      // Without this the provider sends no final usage chunk, and the usage check fails.
      upstreamBody.stream_options = { include_usage: true };
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    upstreamBody = clientApi === 'anthropic' ? { ...body, model } : openAIToAnthropicRequest(body, model);
    headers['anthropic-version'] = '2023-06-01';
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    upstreamBody = clientApi === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
  }
  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
}

/** Whether a provider's JSON answer is well-formed enough to translate. */
function validProviderJson(wire: 'openai' | 'anthropic' | 'gemini', body: Json): boolean {
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    return choices.length > 0 && isRecord(choices[0]) && isRecord(choices[0].message);
  }
  if (wire === 'anthropic') {
    return body.type === 'message' && Array.isArray(body.content);
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  return candidates.length > 0 && isRecord(candidates[0]);
}

/**
 * Answer the client from one provider's response: translate JSON, or translate / pipe the
 * stream. Returns false when the provider's answer is unusable, so the caller fails over.
 */
async function deliverUpstream(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  upstream: Response,
  res: ServerResponse,
): Promise<boolean> {
  const model = candidate.model.id;
  const wire = candidate.probe.provider.wire;
  if (!stream) {
    const text = await upstream.text();
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return false;
    }
    if (!isRecord(parsed) || !validProviderJson(wire, parsed)) return false;
    if (wire === 'openai' && clientApi === 'openai') {
      sendRaw(res, 200, text, 'application/json');
      return true;
    }
    if (wire === 'anthropic' && clientApi === 'anthropic') {
      sendRaw(res, 200, text, 'application/json');
      return true;
    }
    if (wire === 'openai') sendJson(res, 200, openAIToAnthropicResponse(parsed, model));
    else if (wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(parsed, model));
    else if (clientApi === 'openai') sendJson(res, 200, geminiToOpenAIResponse(parsed, model));
    else sendJson(res, 200, geminiToAnthropicResponse(parsed, model));
    return true;
  }
  if (upstream.body === null) return false;
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (wire === 'openai' && clientApi === 'openai') await pipeStream(upstream.body, res);
  else if (wire === 'anthropic' && clientApi === 'anthropic') await pipeStream(upstream.body, res);
  else if (wire === 'openai') for await (const event of openAIStreamToAnthropicEvents(upstream.body, model)) res.write(event);
  else if (wire === 'anthropic') for await (const chunk of anthropicStreamToOpenAIChunks(upstream.body, model)) res.write(chunk);
  else if (clientApi === 'openai') for await (const chunk of geminiStreamToOpenAIChunks(upstream.body, model)) res.write(chunk);
  else for await (const event of geminiStreamToAnthropicEvents(upstream.body, model)) res.write(event);
  res.end();
  return true;
}

/**
 * Try one candidate for one request: up to two attempts on a transient refusal, then give up
 * so the caller fails over to the next-best model. Returns 'served' once the client has an
 * answer, 'failed' when another candidate may be tried, and 'poisoned' when a provider call
 * already succeeded (HTTP 2xx) but its answer was unusable: a second successful call for the
 * same request would fail the env's one-provider check, so no further candidate may be tried.
 */
async function tryCandidate(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  res: ServerResponse,
): Promise<'served' | 'failed' | 'poisoned'> {
  const providerId = candidate.probe.provider.id;
  for (let attempt = 0; attempt < 2; attempt++) {
    let upstream: Response;
    try {
      upstream = await fetchUpstream(candidate, clientApi, body, stream);
    } catch (err) {
      // The provider could not be reached at all: a transient failure, worth one retry,
      // unless the call timed out, which says the provider is wedged rather than refusing.
      if ((err as Error).name === 'TimeoutError') {
        noteFailure(providerId, 'fatal');
        log(`call to ${providerId}/${candidate.model.id} timed out`);
        return 'failed';
      }
      noteFailure(providerId, 'transient');
      log(`call to ${providerId}/${candidate.model.id} failed: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok) {
      noteSuccess(providerId);
      const delivered = await deliverUpstream(candidate, clientApi, stream, upstream, res);
      return delivered ? 'served' : 'poisoned';
    }
    const text = await upstream.text().catch(() => '');
    const kind = classifyFailure(upstream.status, text);
    noteFailure(providerId, kind);
    log(`call to ${providerId}/${candidate.model.id} answered HTTP ${upstream.status} (${kind})`);
    if (kind === 'transient' && attempt === 0) continue;
    return 'failed';
  }
  return 'failed';
}

/** POST /v1/chat/completions or POST /v1/messages: route, fail over, answer in the client's wire. */
async function serveRequest(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const candidates = candidatesFor(requirementsFor(body), jsonModeOf(clientApi, body));
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  for (const candidate of candidates) {
    if (stateOf(candidate.probe.provider.id).dead) continue;
    const outcome = await tryCandidate(candidate, clientApi, body, stream, res);
    if (outcome === 'served') return;
    // 'poisoned': a provider already answered 2xx for this request, so no other provider
    // may be called for it; answer with an error instead of failing over.
    if (outcome === 'poisoned') break;
  }
  if (!res.headersSent) {
    sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'every free model that meets this request refused it');
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveRequest('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveRequest('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err as Error;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw lastError ?? new Error('POST /register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
