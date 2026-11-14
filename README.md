# Free model aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic Messages
API over ten free-tier model providers, in JSON and in server-sent events. It reads
a brief (provider directory, quality table, workload manifest), probes each
provider's catalog route in the provider's own wire, and registers one catalog with
the trial environment.

## Run

```sh
node main.ts            # serve: register with $MAKEFUN_ENV_URL, answer, exit 0 when done
node main.ts --list     # print the catalog document and exit 0
```

Node 22+, standard library only, nothing to install.

## How it works

- **Routing (quality):** every request is answered by the best-rated free model that
  meets its requirements (tools, vision, JSON mode, minimum context), ranked by the
  brief's quality table. A paid model is never picked. JSON-mode requests never go
  to an anthropic-wire provider, whose wire cannot carry `response_format`.
- **Failover:** when a provider refuses with 402/429/5xx, the request is retried on
  the next-best free model. The refusal message tells a spent provider ("budget is
  spent" / "is out", marked dead for the trial) from a scheduled refusal ("refused
  this call", retried once on the same provider). Three consecutive failures also
  mark a provider dead.
- **Wires:** all three upstream wires are spoken natively — OpenAI (`/v1/chat/completions`),
  Anthropic (`/v1/messages`), Gemini (`:generateContent` / `:streamGenerateContent`) —
  and translated back into the client's wire, JSON and streaming. System prompts,
  messages (images included), tools, stop sequences, `max_tokens` and JSON mode pass
  through untouched; usage is the provider's own numbers; stop reasons map by family.
- **Determinism:** no wall clock, no unseeded randomness; every choice is a pure
  function of the brief, the requests and the providers' answers so far.

## Files

- `main.ts` — the server: brief, catalog, routes, routing, failover, register/poll
- `lib/providers.ts` — brief parsing, catalog probe, ratings, requirements
- `lib/wire.ts` — client/provider wire translations and SSE synthesis
