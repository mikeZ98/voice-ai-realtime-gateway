# Architecture — Phase 1 core engine

> Phase 2 (WebSocket transport, framing, control protocol, streaming VAD and the session
> runner) is documented in [`TRANSPORT.md`](TRANSPORT.md).

Phase 1 delivers the transport-agnostic heart of the gateway: the session lifecycle, the
egress playback path with low-latency barge-in, the bounded audio data plane, and the
telemetry spine. WebRTC/SIP transports, VAD, ASR, LLM and TTS adapters plug in later
through the ports defined here (`AudioEgress`, `TelemetrySink`, VAD callbacks).

## 1. Layering

```
voice_ai_realtime_gateway
├── domain/      session semantics            (depends on core, telemetry)
│   ├── states.py     SessionState, SessionTrigger, DrainReason, transition matrix
│   ├── clock.py      LamportClock, MonotonicClock
│   ├── machine.py    SessionStateMachine, Transition
│   ├── duplex.py     DuplexSession (barge-in aware controller), AudioEgress port
│   └── errors.py
├── core/        data plane                   (depends on telemetry)
│   ├── frames.py     AudioFrame (zero-copy), AudioCodec
│   ├── ring.py       AudioRingBuffer (bounded, epoch-fenced)
│   └── supervisor.py TaskSupervisor (leak-free task ownership)
└── telemetry/   leaf layer                   (depends on nothing)
    ├── ids.py        SessionId, AudioSpanId, TraceContext, header codec
    ├── events.py     typed frozen/slotted events
    ├── sink.py       TelemetrySink port, NullSink, FlightRecorder
    └── latency.py    LatencyWindow, GlassToGlassTracker, FirstTokenTracker
```

Imports only point downwards; telemetry serialises domain enums as their `str` values so
it never imports upwards. The package has **zero runtime dependencies**.

### Concurrency model

Each session is confined to one asyncio event loop. Every state-mutating operation
(`SessionStateMachine.fire`, ring `put_nowait/get_nowait/clear/abort/reset`,
`TaskSupervisor.cancel`) is **synchronous**, hence atomic between `await` points. Races
are therefore limited to *which* atomic step runs first. The race suites enumerate those
orderings (fuzzed yield schedules) instead of relying on locks. Nothing here is
thread-safe, by design. Cross-thread producers must hop onto the loop with
`loop.call_soon_threadsafe`.

## 2. Session state machine

### States

| State | Meaning |
|---|---|
| `IDLE` | Created, no transport yet |
| `CONNECTING` | Transport / ICE / DTLS negotiation in progress |
| `LISTENING` | Media up; user audio flows in, agent is silent |
| `PROCESSING_THOUGHT` | User turn ended; LLM is generating, no audio out yet |
| `SPEAKING` | Agent audio is being played out |
| `BARGED_IN` | User speech interrupted the agent (transient, never observable across an `await`) |
| `DRAINING` | Output cancelled; supervised tasks are being reaped (`drain_reason` = `BARGE_IN` or `SHUTDOWN`) |
| `TERMINATED` | Absorbing final state |

### Formal transition matrix

Rows are source states, columns are triggers. `·` marks a **rejected** pair: the pair
raises `InvalidTransitionError`, leaves state, Lamport clock and history untouched, and
emits `TransitionRejected`.

| source \ trigger | CONNECT | CONNECTED | USER_TURN_ENDED | RESPONSE_AUDIO_STARTED | RESPONSE_SKIPPED | RESPONSE_AUDIO_FINISHED | BARGE_IN | DRAIN | DRAIN_COMPLETE | CLOSE | FAULT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **IDLE** | CONNECTING | · | · | · | · | · | · | · | · | TERMINATED | TERMINATED |
| **CONNECTING** | · | LISTENING | · | · | · | · | · | · | · | TERMINATED | TERMINATED |
| **LISTENING** | · | · | PROCESSING_THOUGHT | · | · | · | · | · | · | DRAINING⁽ˢ⁾ | TERMINATED |
| **PROCESSING_THOUGHT** | · | · | · | SPEAKING | LISTENING | · | BARGED_IN | · | · | DRAINING⁽ˢ⁾ | TERMINATED |
| **SPEAKING** | · | · | · | · | · | LISTENING | BARGED_IN | · | · | DRAINING⁽ˢ⁾ | TERMINATED |
| **BARGED_IN** | · | · | · | · | · | · | · | DRAINING⁽ᵇ⁾ | · | DRAINING⁽ˢ⁾ | TERMINATED |
| **DRAINING** | · | · | · | · | · | · | · | · | LISTENING⁽ᵇ⁾ / TERMINATED⁽ˢ⁾ | DRAINING⁽ˢ⁾ | TERMINATED |
| **TERMINATED** | · | · | · | · | · | · | · | · | · | · | · |

⁽ᵇ⁾ drain reason `BARGE_IN`; ⁽ˢ⁾ drain reason `SHUTDOWN`. `DRAIN_COMPLETE` is the only
guarded transition: it resolves by the drain reason. `CLOSE` during a barge-in drain
escalates the reason to `SHUTDOWN` (an explicit self-transition), so a hang-up can never
be "undone" into `LISTENING`.

`tests/domain/test_states.py` checks all 8 × 11 × 3 `(state, trigger, reason)` triples
against an independently written oracle. `tests/domain/test_machine_fuzz.py` drives
random trigger streams with random remote Lamport stamps against a second reference model.

### Lamport timestamps

Every committed transition consumes exactly one tick: `L := max(L, observed) + 1`, where
`observed` is the stamp carried by a remote message (client signalling, model server)
that caused the trigger. Within a session, stamps are strictly increasing. Across peers
they respect happened-before. Rejected triggers do not tick.

## 3. Barge-in

```
VAD onset ──► on_user_speech_started(at_ns)
               │  state ∈ {PROCESSING_THOUGHT, SPEAKING}?
               ▼
      ┌──────── synchronous critical section (no await) ─────────┐
      │ fire(BARGE_IN) → supervisor.cancel() → ring.abort()      │
      │ → first_token.disarm() → fire(DRAIN)        ⇒ DRAINING   │
      └──────────────────────────────────────────────────────────┘
               │  latency = now − at_ns   (budget 15 ms)
               ▼
      await supervisor.join(drain_timeout_s)       (bounded reap)
               │ stragglers? ──► FAULT ⇒ TERMINATED   (fail closed)
               ▼
      fire(DRAIN_COMPLETE) ⇒ LISTENING
               │ deferred VAD offset? ──► USER_TURN_ENDED ⇒ PROCESSING_THOUGHT
               ▼
      emit BargeInHandled, return BargeInReport
```

* **< 15 ms to `DRAINING`.** Nothing between detection and `DRAINING` yields to the loop,
  so the cost is O(turn tasks + queued frames) of pure Python. Measured on Apple
  Silicon / CPython 3.12 with 16 turn tasks and a full 256-frame ring (2,000 runs):
  **p50 22 µs, p99 27 µs, max 97 µs**. `within_budget` is reported on every barge-in.
* **No stale audio.** `ring.abort()` bumps the ring epoch. The pump (blocked in `get`)
  and any producer blocked in `put` are released, and a waiter from an old epoch can
  never insert or consume a frame of the next one. `push_audio` also fences on the
  response span ID (`StaleSpanError`).
* **No task leaks.** Every task is owned by `TaskSupervisor`. Its done-callback retrieves
  every outcome (no "exception was never retrieved"), and `join` is deadline-bounded.
  Tasks that swallow cancellation are reported as stragglers and fault the session.
* **No unhandled cancellations.** If the caller of the barge-in is itself cancelled
  while reaping, `CancelledError` propagates untouched. The session stays `DRAINING`
  and `aclose()` completes it.
* **No lost turns.** A VAD offset that arrives mid-drain is deferred and replayed as
  `USER_TURN_ENDED` once back in `LISTENING`. A fresh onset cancels the deferral.

## 4. Audio ring

* Preallocated power-of-two slot array; head/tail by bit mask. Frames are stored **by
  reference**. `AudioFrame.wrap` keeps a read-only `memoryview` over the producer's
  buffer, so no payload byte is ever copied.
* Dual bound: a frame is admitted only if a slot **and** byte budget are free.
* `BACKPRESSURE` (egress default): `await put` suspends the producer, giving TTS flow
  control. `DROP_OLDEST` (live ingress): never blocks and evicts the oldest frames,
  bounding queueing delay under jitter or slow consumers, which prevents bufferbloat.
  Evictions emit `BufferOverflow`.
* Wake-ups are broadcast and every waiter re-checks its predicate, so a cancelled waiter
  can never swallow a wake-up. Model-based fuzzing compares the ring against a
  bounded-deque reference.

## 5. Telemetry

* Correlation: `X-Session-ID` (128-bit, 32 hex) and `X-Audio-Span-ID` (64-bit, 16 hex),
  W3C-style, with all-zero reserved. Every `AudioFrame` and every `Transition` carries a
  `TraceContext`. A new span opens per agent response (`USER_TURN_ENDED`).
* Events are frozen, `slots` dataclasses (~80–100 B each, one allocation), emitted
  synchronously to a non-blocking `TelemetrySink`: VAD start/stop, first-token latency,
  per-packet glass-to-glass latency, state transitions and rejections, barge-in
  outcomes, buffer overflow, task failures.
* `LatencyWindow` records in O(1) into a preallocated `array('q')`. Percentiles
  (nearest rank) are computed on demand, off the hot path.

## 6. Memory budget model

Measured with `sys.getsizeof` / `tracemalloc` on CPython 3.12 (64-bit):

| Component | Size | Bound |
|---|---|---|
| `AudioFrame` metadata | 96 B + 184 B read-only `memoryview` ≈ **280 B** | per queued frame |
| Ring slot | 8 B pointer | `capacity_frames` (preallocated) |
| Ring payload | producer buffer (not copied) | `capacity_bytes` |
| `Transition` history | ≈ 100 B | `history_size` (default 64) |
| `LatencyWindow` | 8 B × capacity | 1024 → 8 KiB, two per session |
| `FlightRecorder` (optional) | ≈ 80–100 B / event | `capacity` |
| Idle `DuplexSession` | **≈ 23 KiB** (measured) | fixed |

**Per-session worst case** (default egress ring, 256 frames / 512 KiB):

```
fixed                          ≈  23 KiB
transition history (64)        ≈   6 KiB
ring payload                   ≤ 512 KiB      (capacity_bytes)
ring frame metadata  256 × 280 ≈  70 KiB      (capacity_frames)
------------------------------------------------
                               ≈ 611 KiB   → 1,000 sessions ≈ 0.6 GiB
```

With Opus at 64 kbit/s (160 B per 20 ms frame) the frame bound binds first:
256 × (160 + 280) B ≈ 110 KiB per session, which is 5.12 s of buffered speech.
`capacity_frames` bounds latency (time in queue), `capacity_bytes` bounds memory, and
both are hard limits: no code path can grow beyond them.

## 7. Verification matrix

| Property | Evidence |
|---|---|
| Matrix correctness | exhaustive triple enumeration vs. oracle (`test_states.py`) |
| Machine invariants | Hypothesis stateful fuzzing, 300 × 80 steps (`test_machine_fuzz.py`) |
| Ring ≡ bounded deque | Hypothesis stateful model (`test_ring_model.py`) |
| Exactly-once, per-producer FIFO under contention | fuzzed N×M producer/consumer schedules (`test_ring_concurrency.py`) |
| Barge-in budget, leaks, cancellation safety | `test_barge_in.py` (+ fuzzed barge-in point) |
| No leaked tasks / unretrieved errors | `strict_loop` fixture on every async test |
| Network-free unit tests | `pytest-socket --disable-socket` (AF_UNIX allowed only for the asyncio self-pipe) |
| 100.00 % line + branch coverage | `scripts/coverage-gate.sh` (per-file, no rounding, no pragmas) |
