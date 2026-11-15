# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten free-tier model providers, routing each request to
the best free model that can serve it and failing over when a provider
refuses.

Entry for the make.fun competition `free-model-aggregator-01`.

## What it does

- Serves `POST /v1/chat/completions` (OpenAI wire) and
  `POST /v1/messages` (Anthropic wire), in JSON and in server-sent
  events, plus `GET /v1/models` in the wire the auth header names
  (`Authorization: Bearer` gets the OpenAI list, `x-api-key` the
  Anthropic one).
- Reads the trial brief (`MAKEFUN_PROMPT_FILE`): the ten providers, the
  sealed quality table and the workload manifest.
- Probes every provider's own model list and registers the catalog it
  found with the env (`POST $MAKEFUN_ENV_URL/register`).
- Routes each request to the highest-rated free model meeting its
  requirements (tools, vision, JSON mode, minimum context). A request
  that needs JSON mode is never sent to an Anthropic-wire provider,
  which cannot carry it.
- Speaks each provider's own wire (OpenAI, Anthropic, Gemini) and auth
  scheme (`Bearer`, `x-api-key`, `x-goog-api-key`, none), translating
  the system prompt, user text and images, tools, `max_tokens`, stop
  sequences and JSON mode faithfully, and returns the provider's answer
  whole, with its usage counts and stop reason mapped into the wire the
  client used.
- On a refusal (402/429) it counts the failure and answers the request
  with the next-best model. A provider that refuses twice running is
  treated as spent for the rest of the trial; a second pass over the
  candidates gives a provider refused on its own failure schedule
  another chance, since that refusal is transient.
- Polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when the workload is
  done.
- `node main.ts --list` prints the catalog document it registers,
  byte for byte. Run outside a trial it prints `{"providers":[]}`.

## Running it

Node 22, standard library only; nothing to install.

```sh
node main.ts           # serve a trial (env vars set by the harness)
node main.ts --list    # print the catalog and exit 0
```

## Practice scores

Graded locally with the kit (`grade-local --driver process`), all three
practice instances, replay matched byte for byte on each:

| trial | score | quality | served | wire | failover | catalog | calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |

Quality 99 is the practical ceiling: the bound ignores the providers'
per-trial budgets, and the single best free model (1475) cannot serve
all 17 requests that want it, so the last few fall back to the 1470
models. Every request was served validly on all three trials (233/233
response checks passed on trial 0).

## License

MIT. See `LICENSE`.
