# Phase 3: real-time AI inference pipeline

Phase 3 turns the gateway into a voice agent:
`Transport → VAD → ASR → LLM → SentenceChunker → TTS → Transport`. All of it streams,
everything is bounded, and conversation history stays truthful when the user
interrupts.

## 1. Layering and ports

```
inference/   VoiceAgentPipeline, SpeechCapture, SentenceChunker, ConversationState,
             SpeechMap, PlaybackLedger, ports (ASR/LLM/TTS), deterministic doubles
pipeline/    GatewaySessionRunner (Phase 2) + egress / ingress-tap / services hooks
domain/      DuplexSession (+ wait_playback_complete) · core/ · telemetry/ · transport/
```

| Port | Contract |
|---|---|
| `StreamingASR.transcribe(audio)` | `AsyncGenerator[TranscriptChunk]`: interim hypotheses, then one final result (text, confidence, word timings, utterance span) |
| `StreamingLLM.stream(history, user_text)` | `AsyncGenerator[TokenDelta]` |
| `StreamingTTS.synthesize(text, stamper)` | `AsyncGenerator[SynthesizedAudio]`: each frame plus how many characters of the unit it has voiced |

Every port is an async generator, so each stage consumes it under
`contextlib.aclosing`. On cancellation the provider stream's `finally` runs right away,
not whenever the garbage collector finds a suspended generator. `FrameStamper` wraps
synthesized PCM in read-only `memoryview` frames without copying, stamped with the
active response span and a continuous sequence.

`ToneASR`, `ScriptedLLM` and `ToneTTS` are deterministic doubles. They encode words as
tone amplitudes, stream scripted tokens and synthesize square waves with exact
alignment, and they count open generators so tests can prove nothing was orphaned. The
suite stays network-free.

## 2. Data flow and supervision

```
client audio ─► transport ─► ingress loop ─► VAD bridge ─► session (barge-in, turns)
                                   │ tap
                                   ▼
                            SpeechCapture: 300 ms pre-roll → per-utterance ring (bounded)
                                   │
 runner service "recognise" ◄──────┘  StreamingASR ─► final transcript ─► UserTurn ─► history
                                                                         └► utterance future
 USER_TURN_ENDED ─► on_user_turn ─► session.spawn(turn)
   turn:     await utterance future ─► spawn stages ─► wait(stages) ─► wait_playback_complete
   generate: StreamingLLM ─► SentenceChunker ─► unit queue (≤ 4 units ahead)
   speak:    unit queue ─► StreamingTTS ─► SpeechMap ─► session.push_audio ─► egress ring
                                                       egress pump ─► PlaybackLedger ─► transport
   commit:   AgentTurn (full or truncated) + ResponseTimeline
```

* The **turn and both stages are session tasks**, so the barge-in critical section
  cancels them synchronously, in the same synchronous step as the egress-ring abort
  and before `DRAINING`. A cancelled task receives `CancelledError` at its next
  resumption, so no stage code can act on an interrupted session. An earlier draft
  nested the stages in a `TaskGroup` inside the turn: a TTS stage already scheduled
  to resume then pushed audio into a `DRAINING` session and turned the barge-in into a
  fault. The race tests caught it.
* **Fault barrier:** a failing stage is reaped once by the session supervisor, which
  faults the call (`FAULT`, socket `1011`) and cancels the other stage and the turn. The
  turn waits on its stages with `asyncio.wait`, which never re-raises their errors, so
  each failure is reported exactly once. A failing ASR fails its runner service, which
  faults the call the same way.
* **Backpressure end to end:** the unit queue bounds how far the LLM can run ahead of
  TTS; the egress ring and the transport bound TTS. Each utterance ring is bounded and
  drop-oldest, so a slow recogniser never slows realtime ingress.

## 3. Turn truncation on barge-in

The history must hold what the user **heard**, not what the LLM **generated**.

1. **Alignment.** For every synthesized frame, `SpeechMap` records the cumulative media
   position and the character offset (`text_offset`) reached in the current unit, in
   preallocated `array('q')` columns (24 B per frame).
2. **Playback model.** `PlaybackLedger` wraps the egress. Per response span it records
   when the first frame left the gateway (`first_sent`) and the media duration sent.
   The client plays in real time from its first packet, so at time `t` it has played at
   most `heard = min(media_sent, t − first_sent)`.
3. **Cut.** On interruption, binary-search `heard` in the alignment, keep the units
   fully played plus the prefix of the current unit, then snap back to a whole word
   (`word_prefix`). A word cut in half is dropped: the history may under-report what
   was said, but never contains words the user did not hear.
4. **Commit timing.** The agent turn is committed only after
   `DuplexSession.wait_playback_complete()`, when the response has left `SPEAKING`.
   TTS usually runs faster than real time, so it can finish *generating* a sentence
   that is still queued for playback; committing then would store the whole sentence
   even if the user interrupts a second later.

Worked example (tested end to end): the reply is "Sure, the weather is mild. Anything
else?". The first unit is 26 characters, synthesized as 11 frames of 20 ms. The user
barges in 70 ms into playback. That is 3 whole frames, so 26 × 3 // 11 = 7 characters
voiced ("Sure, t"), which snaps back to **"Sure,"**. The stored `AgentTurn` is
`"Sure,"` with `interrupted=True` and `generated_chars=40`. "weather", "mild" and
"Anything" never enter the context.

The **user** turn is recorded by the ASR service as soon as its final transcript exists,
so an utterance the user cuts short still appears in the history.

## 4. Sentence boundary buffering model

`SentenceChunker` sits between token deltas and TTS. Single tokens wreck prosody;
whole responses wreck latency. The chunker emits a unit when a boundary is certain:

| Boundary | Rule |
|---|---|
| Sentence end | `.` `!` `?` `…` and runs (`?!`, `...`), plus closing quotes or brackets, **followed by whitespace** |
| Not a boundary | decimals (`3.14`), dotted forms (`example.com`), abbreviations (`Dr.`, `e.g.`), capital initials (`J.`, but not `I.`) |
| CJK sentence end | ideographic full stop, full-width `!` and `?`; no whitespace needed |
| Clause end | `,` `;` `:` em or en dash plus whitespace, once the unit is at least `first_clause_min_chars` (16) for the **first** unit, for fast first audio, or `clause_min_chars` (60) later |
| Length cap | at most `max_chars` (200): cut at the last whitespace, else hard-cut |

**Deferral:** when a decision needs a character that hasn't arrived (is this `.` a
decimal point?), the scanner stops and resumes on the next delta. It never guesses,
so the output is **identical however the stream is tokenised** (fuzzed). Scanning
never revisits a position, so cost is linear in characters. Buffered text is bounded
by `max_chars` plus one delta. Punctuation-only fragments never form a unit.

**Latency cost:** a sentence is released when the next token arrives, i.e. one token
of lookahead. In the measurement below, the first unit, "Sure, I can help with that.",
was emitted when the 7th token ("␣The") arrived.

## 5. Measured

CPython 3.12, Apple Silicon. Pipeline numbers use instant model doubles, so they show
the gateway's own overhead, which adds to model latency.

| Metric | p50 | p99 |
|---|---|---|
| `SentenceChunker.feed` per token | 0.71 µs | 1.96 µs |
| Glass-to-glass inside the gateway: VAD offset → first agent packet on the socket | 366 µs | 599 µs |
| TTS generation halted after barge-in onset (requirement < 5 ms) | 242 µs | 434 µs (max 473 µs) |

`ResponseTimeline` gives the same breakdown for every production response: speech end
(VAD offset), final transcript, first token, first synthesized frame, first egress, and
the `glass_to_glass_ns` derived from them. `speech_end_ns` is the VAD offset decision,
which includes the configured hangover.

## 6. Memory

| Component | Bound |
|---|---|
| Pre-roll | `preroll_ms` (300 ms ≈ 15 frames) |
| Utterance ring | 512 frames / 2 MiB, drop-oldest |
| Unit queue | 4 units × ≤ 200 chars |
| `SpeechMap` | 24 B per synthesized frame of the current response (1.2 KiB/s) |
| Conversation | `max_turns` (64) and `max_chars` (32,000) window, oldest pruned first |

## 7. Verification

| Property | Evidence |
|---|---|
| Heard text is a monotone, whole-word prefix of the response | Hypothesis property over random units, alignments and cuts (`test_playback.py`) |
| No hallucinated words after a mid-sentence barge-in | end-to-end truncation on a controlled playback clock (`test_agent.py`) |
| Chunker output independent of tokenisation; no speech lost; bounded units | Hypothesis over adversarial token mixes (`test_chunker.py`) |
| TTS halts < 5 ms after barge-in; LLM and TTS streams closed | real-clock end-to-end test, open-generator counters |
| Barge-in at any point leaks nothing and keeps the history truthful | fuzzed barge-in point × capacity × speaking rate under a strict loop |
| Stage or ASR failure faults the call exactly once, with no orphans | fault-barrier tests per stage |
| Concurrent calls stay isolated | 12 simultaneous calls with barge-ins in one event loop |
