# Dexter Voice

Dexter's production speech stack is split into input and output so each side can use the backend best suited to it without competing for the RTX 5050.

## Speech input

- Moonshine Medium Streaming
- CPU inference
- Push-to-talk hotkey: `Ctrl+/`
- `sounddevice` / PortAudio input stream stays open between PTT presses

## Speech output

- NeuTTS-2E speaker: `paul`
- Emotion: `neutral`
- Backbone: `neuphonic/neutts-2e-q4-gguf`
- Backbone device: Intel Arc 140T through llama.cpp Vulkan
- Codec: `neuphonic/neucodec-onnx-decoder-int8` on CPU
- Perth watermarking enabled
- 24 kHz mono output
- Windows default output routing: SoundCard/Core Audio/WASAPI

The RTX 5050 remains available to the main Gemma/Ollama workload.

## Validated production tuning

Live profiling on the target Core Ultra 7 255H machine showed that the original pipeline could fall to roughly 1.7-1.8 RTF while Gemma was generating, producing real PCM starvation and audible stop/start glitches.

The current production settings are:

- Q4 NeuTTS backbone
- 8 NeuTTS generation threads (NeuTTS default on this 16-logical-core machine)
- TTS worker created with Windows `ABOVE_NORMAL_PRIORITY_CLASS`
- NeuCodec ONNX Runtime pool capped to 2 intra-op / 1 inter-op threads
- first stream window: 40 frames
- steady stream window: 50 frames
- 0.5 s playback/rebuffer reserve

The largest win was constraining NeuCodec's ONNX Runtime thread pool. With the final 40/50 stream windows, repeated fully-concurrent runs measured about 0.79-0.81 RTF with no PCM underflow events in the tested responses.

## Windows output-device following

PortAudio's default output device was observed to remain stale while Windows switched between laptop speakers and connected audio devices. TTS playback therefore uses `Runtime/Voice/windows_default_audio_sink.py`, backed by SoundCard/Core Audio/WASAPI, and polls the live Windows default render endpoint.

This means Dexter follows normal Windows behavior:

- plug in headphones/speakers -> TTS moves to that output
- unplug them -> TTS falls back to the new Windows default
- switching can happen while an answer is playing

STT remains on its existing PortAudio microphone stream and is not globally reinitialized when the output changes.

## Voice Output toggle

Voice Output is a resource control, not a mute-only switch. Turning it off stops playback, shuts down the NeuTTS worker/model, and releases its Arc/CPU/RAM resources. Re-enabling Voice Output performs a fresh warmup.

## Production launcher

Normal hidden launch path:

```text
wscript.exe
  -> Runtime/Host/launch_hidden.vbs
  -> .venv/Scripts/pythonw.exe -m Runtime.Host.app
```

The hidden launcher preserves stderr in:

```text
%LOCALAPPDATA%\Dexter\Diagnostics\host-stderr.log
```

This is the first place to inspect if Voice fails during normal shortcut startup.

## Diagnostics

Set these before a diagnostic launch when needed:

```powershell
$env:DEXTER_TTS_TRACE = "1"
$env:DEXTER_TTS_PROFILE_STAGES = "1"
$env:DEXTER_TTS_PROFILE_LLAMA_STAGES = "1"
```

Trace output is written to:

```text
%LOCALAPPDATA%\Dexter\Diagnostics\tts-trace.log
```

Useful production overrides remain available for controlled A/B testing:

- `DEXTER_TTS_VARIANT`
- `DEXTER_TTS_THREADS`
- `DEXTER_TTS_CODEC_THREADS`
- `DEXTER_TTS_FIRST_STREAM_FRAMES`
- `DEXTER_TTS_STEADY_STREAM_FRAMES`
- `DEXTER_TTS_BACKBONE_DEVICE`
- `DEXTER_TTS_VULKAN_DEVICE`

## Known remaining work

The live throughput/choppiness problem is considered solved for the current machine. Remaining TTS work is primarily:

- reduce the long-lived worker's oversized RAM footprint by removing unused eager ML-stack imports
- fix the text-queue overflow policy so queue pressure cannot cancel the rest of a response
- continue broader CPU/NPU experiments only if they offer a concrete resource or latency benefit

See GitHub issue #9 for the current optimization backlog.
