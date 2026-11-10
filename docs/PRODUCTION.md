# Phase 4: production hardening, provider adapters and operational telemetry

Phase 4 makes the gateway deployable. It adds real-provider streaming adapters behind
the Phase 3 ports, a provider resilience state machine, Prometheus metrics with a status
endpoint, signed-token admission with tenant policy, abuse guards and mutation testing.
The gateway still has **zero runtime dependencies**, and the whole suite is
network-free.

## 1. Topology

```
                 ┌──────────────── GatewayServer.handle(connection, headers) ────────────────┐
 client ─WS──►   │ admit: drain? → bearer JWT (HS256) → tenant → capacity → sid → codecs      │
                 │        └ refuse: 1008 policy / 1013 try-again-later                        │
                 │ WebSocketVoiceTransport (rate-limited control, 1008 size guards)          │
                 │ VoiceAgentPipeline (Phase 3)                                              │
                 │   ├─ DeepgramStreamingASR ─┐                                              │
                 │   ├─ OpenAIRealtimeLLM ────┼─ ProviderLink (retry/backoff state machine)  │
                 │   └─ CartesiaStreamingTTS ─┘      └─ injected Connector (websockets lib)  │
                 │ watchdog: no media within media_timeout_s → 1008; max duration → 1008      │
                 │ closers: provider connections released on every outcome                    │
                 └──────────────────────────────┬────────────────────────────────────────────┘
                                 telemetry events ▼
                 GatewayMetrics (TelemetrySink) ── MetricsRegistry ── StatusApp (ASGI)
                 SessionRegistry (live gauges)                         /healthz /readyz /metrics
```

## 2. Provider adapters

Each adapter implements a Phase 3 port. It owns exactly one pure parser function per
provider wire format, and receives its network connection from an injected
`Connector`: `async () -> WebSocketConnection`. In production the connector wraps the
`websockets` library, builds the URL and auth header, and maps HTTP 401/403 to
`UpstreamError(AUTH)` and 402 to `UpstreamError(QUOTA)`. In tests it pairs with a
simulated provider over in-memory sockets.

| Adapter | Port | Wire highlights | Recovery |
|---|---|---|---|
| `DeepgramStreamingASR` | `StreamingASR` | binary PCM up, `Results` (interim/final, word timings) down, `CloseStream` to flush, `Metadata` marks the end | bounded **replay** of audio not yet covered by a final; on reconnect, unfinalized audio is resent and timings shifted, so no word is lost or duplicated |
| `OpenAIRealtimeLLM` | `StreamingLLM` | out-of-band `response.create` (`conversation: "none"`, explicit `input` rebuilt from `ConversationState`), `response.output_text.delta` (beta `response.text.delta` accepted), `response.done`, `error` | retry before the first token; after tokens, end the reply gracefully; `response.cancel` on barge-in |
| `CartesiaStreamingTTS` | `StreamingTTS` | one `context_id` per unit, raw `pcm_s16le`, base64 `chunk`, `word_timestamps`, `done`, `error.status_code`, cancel by context | retry the unit before audio; after audio, end the unit gracefully; zero-copy 20 ms framing; conservative word alignment |

The out-of-band design for OpenAI keeps the gateway's history authoritative: after a
barge-in truncated the previous agent turn, the provider never holds the full, unheard
reply.

**Verification caveat.** The message shapes follow the providers' public documentation
as of this phase. They are isolated in `parse_*` functions that are strict on the
fields used and tolerant of unknown message types, but they cannot be verified against
the live APIs from a network-free suite. Run the adapters against each provider's
sandbox before production and pin the API version in the connector.

## 3. Provider resilience state machine

```
IDLE ──open()──► CONNECTING ──ok──► STREAMING ──drop / protocol error──► reopen()
                    │   ▲                                                   │
          retryable │   │ backoff elapsed                                   │ fatal, or
                    ▼   │                                                   │ budget spent
                  BACKOFF ───── max_attempts exhausted ──────────────► FAILED
                                                           close() ─► CLOSED
```

| Kind | Examples | Policy |
|---|---|---|
| `auth` | HTTP 401/403, `invalid_api_key` | **fatal** immediately → stage failure → `SessionTrigger.FAULT` |
| `quota` | HTTP 402, `insufficient_quota` | **fatal** immediately |
| `rate_limit` | 429, close 1013, `rate_limit_exceeded` | retry, honouring `retry_after` |
| `transient` | network loss (1006), 5xx, timeouts | retry with backoff |
| `protocol` | malformed frame, bad base64 | retry (often a single bad frame); fatal once retries are exhausted |

Backoff is `min(max_backoff, initial · multiplier^(attempt−1))`, multiplied by a
uniform jitter factor in `[1 − j, 1 + j)`. `max_attempts` bounds each connect and
`max_reconnects` bounds mid-stream recoveries. Every failure emits a `ProviderError`
event, counted in `upstream_provider_errors_total{provider,kind,fatal}`. Every retry
goes through the link. That fixed a bug caught in review: a retryable server error that
kept its connection would otherwise have retried in a tight loop with no backoff and no
budget.

## 4. Metric exposition model

Text format 0.0.4, zero dependencies (`telemetry/metrics.py`). Label values are bounded
enumerations, and every family additionally caps its series count. Live gauges are
computed at scrape time.

| Metric | Type | Labels | Source |
|---|---|---|---|
| `voice_gateway_active_sessions_count` | gauge | | `SessionRegistry` |
| `voice_gateway_active_audio_rings_bytes` | gauge | | ingress + utterance + egress rings |
| `voice_gateway_glass_to_glass_seconds` | histogram | | `ResponseTimeline` (VAD offset → first egress) |
| `voice_gateway_first_token_latency_seconds` | summary (p50/p95/p99) | | `FirstTokenLatency` |
| `voice_gateway_tts_barge_in_halt_microseconds` | histogram | | `TtsHalted` (onset → TTS stream closed) |
| `voice_gateway_barge_in_total` | counter | | `BargeInHandled` |
| `voice_gateway_packet_drops_ingress_total` | counter | `reason` = overflow / network_loss | `BufferOverflow`, `IngressGap` |
| `voice_gateway_packet_drops_egress_total` | counter | | `EgressDropped` |
| `voice_gateway_upstream_provider_errors_total` | counter | `provider`, `kind`, `fatal` | `ProviderError` |
| `voice_gateway_transport_closed_total` | counter | `code` | `TransportClosed` |
| `voice_gateway_sessions_total` | counter | `outcome` = admitted / rejected / completed / faulted | server |

`StatusApp` is an ASGI 3 app serving `/healthz` (liveness), `/readyz` (503 while
draining, so load balancers stop routing before shutdown) and `/metrics`. It supports
`HEAD`, returns 405 with `Allow` and 404 for unknown paths, sets `no-store`, and
handles the lifespan protocol.

## 5. Admission and abuse guards

**Tokens:** compact JWS, `HS256`, `kid` key rotation, keys of at least 32 bytes. `alg`
must be exactly `HS256`. The signature is compared in constant time **before** any
claim is read. Then come time bounds with leeway and claim validation: `aud`, `exp`,
`nbf`/`iat`, `tenant`, `codecs`, `max_duration_s`, and an optional `sid` that binds the
token to one `X-Session-ID`. Refusal reasons are short, public and never echo token
contents.

**Admission order:** drain (1013) → token (1008) → tenant policy (1008) → per-tenant
concurrency (1013) → session-id binding (1008) → codecs = token ∩ tenant, enforced by
the transport (1003 on violation) → duration = min(token, tenant).

| Threat | Guard | Close |
|---|---|---|
| Ping flood / control spam | per-connection token bucket (`control_rate_per_s`, `control_burst`) | 1008 |
| Oversized text or binary frame | `max_control_chars`, `max_payload_bytes`, checked before parsing | 1008 |
| Control backlog (consumer lag) | `max_pending_control` | 1008 |
| Slow-loris / slot squatting | watchdog: no audio within `media_timeout_s` | 1008 |
| Endless session | watchdog: admitted maximum duration | 1008 |
| Dead peer | heartbeat timeout | 1001 |
| Cross-session audio injection | packet `X-Session-ID` must match the connection | 1008 |
| Client-driven memory growth | bounded rings, queues and replay buffers everywhere | n/a |

HTTP-level slow-loris (a client trickling the upgrade request) belongs to the ASGI
server's handshake timeout. Configure it there.

## 6. Mutation testing

`scripts/mutation-test.py` is an AST mutation harness:
- **operators:** comparison flips and boundary shifts, `and` ↔ `or`, dropping `not`,
  arithmetic and augmented-assignment swaps, integer `±1`, and `True` ↔ `False`;
- **never mutated:** annotations, `TYPE_CHECKING` blocks, docstrings, `__all__` and
  `__slots__`;
- **isolation:** each worker runs its own copy of `src/`, imported ahead of the
  editable install, with a private, derandomized Hypothesis database. The harness
  checks that the copy is really what's imported, and that the unmutated baseline
  passes;
- **timeouts:** a hanging mutant counts as killed.

Targets are the audio framing and dispatch core: the barge-in critical path
(`domain/duplex.py`), the sentence chunker, the PCM frame contract (`core/frames.py`),
the wire packet codec (`transport/framing.py`), the audio ring (buffer overflow logic),
the task supervisor (non-blocking cancellation) and the finiteness predicates every
contract validates through (`_validation.py`).

**Results** (40 Hypothesis examples per property, derandomized):

| Target | Mutants | Killed | Score |
|---|---|---|---|
| barge-in critical path (`domain/duplex.py`) | 43 | 43 | **100.00 %** |
| sentence chunker (`inference/chunker.py`) | 93 | 90 | **96.77 %** |
| PCM frame contract (`core/frames.py`) | 32 | 32 | **100.00 %** |
| wire packet codec (`transport/framing.py`) | 30 | 30 | **100.00 %** |
| audio ring / overflow (`core/ring.py`) | 72 | 72 | **100.00 %** |
| task supervisor (`core/supervisor.py`) | 6 | 6 | **100.00 %** |
| finiteness predicates (`_validation.py`) | 9 | 9 | **100.00 %** |
| **overall** | **285** | **282** | **98.95 %** |

The frame and codec targets were added after v0.5.0. Their first run left 9 survivors, all
test gaps, all now killed:
- the smallest valid sample rate and duration (1);
- a 2-D byte buffer, which `AudioFrame.wrap` must flatten;
- the 32 KiB default payload limit, the wire codec IDs (PCM16 = 1, Opus = 2) and the
  full u64 range of the sender timestamp, now pinned;
- a payload exactly at the decoder's limit, which must be accepted.

v0.4.0 measured 207 / 210 (98.57 %). In v0.5.0 the abbreviation lookback became
bounded by the unit start (a split-invariance bug found by the fuzzer), which replaced
a mutable `0` literal with a name. That leaves 93 chunker mutants, the same 3
equivalent survivors, and an overall score of 98.56 %.

The first run scored 86.19 %. The survivors were triaged one by one:

* **Test gaps, now killed:**
  - the exact 15 ms budget boundary (`<` vs `<=`), plus the specified constants and
    defaults;
  - validation boundaries (a 1 ns budget and a 1-byte ring must be accepted);
  - initial `faulted` and high-water-mark values, and the epoch's strict monotonicity;
  - a deferred turn end that a client cancel could otherwise have replayed twice;
  - `Dr.` at the very start of the text, uppercase non-initials (`OK.`) and lowercase
    single letters;
  - sentences of exactly `max_chars`, and clauses exactly at their minimum;
  - an overflow measured from the wrong origin;
  - the first-unit clause rule after chunker reuse.
* **Code simplified:** the chunker's `final` flag was redundant, because a deferred
  boundary can only sit at the end of the buffer, so flushing the remainder is exactly
  equivalent. The overflow cut was rewritten on `rstrip` / `rsplit` with identical
  behavior, as the split-invariance and conservation fuzzers confirm.
* **Equivalent (3), documented rather than contorted:**
  - the integer value of the `_WAIT` sentinel (any negative number behaves identically);
  - the scanner's initial index, set in the constructor and again on flush. Position 0
    can never be a boundary, because a boundary needs speech before it.

Run it with `sh scripts/mutate.sh [--target NAME ...] [--jobs N] [--min-score 95]`. The
script wraps the harness, fails below the minimum score, and verifies that tracked sources
are unchanged afterwards. The report is written to `.cache/mutation/report.json`.
