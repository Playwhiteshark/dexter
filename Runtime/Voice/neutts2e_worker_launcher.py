"""Launch Dexter's NeuTTS worker with validated production tuning.

This wrapper exists only around the isolated TTS child process. It applies the
validated llama-cpp-python batch settings and installs Dexter's validated lean
compatibility paths before NeuTTS is imported. No system-wide monkey patches are
installed and Ollama is unaffected.
"""

from __future__ import annotations

import fnmatch
import os
import runpy
import sys
from pathlib import Path

WORKER_PATH = Path(__file__).resolve().parent / "neutts2e_worker.py"


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc
    if value < 1:
        raise SystemExit(f"{name} must be positive")
    return value


def _binary_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip()
    if raw not in {"0", "1"}:
        raise SystemExit(f"{name} must be 0 or 1")
    return raw == "1"


def main() -> int:
    # Target-machine A/B (2026-08-25): 128/128 reduced the Q4 backbone by about
    # 325 MB working set / 644 MB private versus NeuTTS's upstream 512/512,
    # while sustained concurrent Gemma + TTS remained faster than realtime
    # (RTF 0.794 vs 0.807 at 512) with only ~78 ms extra first-audio latency.
    batch = _positive_int("DEXTER_TTS_BACKBONE_BATCH", 128)
    ubatch = _positive_int("DEXTER_TTS_BACKBONE_UBATCH", batch)

    # NeuTTS imports the full neucodec PyTorch implementation even when only the
    # precompiled ONNX decoder is selected. Dexter's compatibility module exposes
    # the ONNX decoder API NeuTTS needs and deliberately rejects unused PyTorch
    # codec paths. Repeated sustained-Gemma A/B tests found effectively unchanged
    # performance while saving ~52.5 MB working set / 48.1 MB private memory.
    lean_neucodec = _binary_flag("DEXTER_TTS_LEAN_NEUCODEC", True)
    if lean_neucodec:
        from neucodec_onnx_compat import install_as_neucodec

        install_as_neucodec()
        print(
            "[TTS] NeuCodec ONNX path=lean-compat",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[TTS] NeuCodec ONNX path=upstream",
            file=sys.stderr,
            flush=True,
        )

    # NeuTTS also imports AutoTokenizer/AutoModelForCausalLM unconditionally,
    # although Dexter's production Q4/GGUF path uses llama.cpp and never touches
    # either Transformers class. The full Transformers import currently pulls
    # scikit-learn -> SciPy on the target machine, and Windows Application Control
    # blocks SciPy's unsigned native extension before Voice can start. Use the
    # previously validated quantized-only compatibility surface by default. It
    # fails loudly if a non-GGUF backbone is ever requested. Earlier sustained
    # Gemma A/B measured a modest ~4% RTF regression versus full Transformers,
    # but both variants remained faster than realtime; the upstream path is now
    # not production-viable under the machine's security policy. Keep an explicit
    # fallback for diagnostics or a future signed dependency stack.
    lean_transformers = _binary_flag("DEXTER_TTS_LEAN_TRANSFORMERS", True)
    if lean_transformers:
        from transformers_quantized_compat import install_as_transformers

        install_as_transformers()
        print(
            "[TTS] Transformers path=quantized-compat",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[TTS] Transformers path=upstream",
            file=sys.stderr,
            flush=True,
        )

    # Perth's apply_watermark() uses only the watermark encoder, but upstream
    # Perth also constructs a three-branch detector network used for watermark
    # detection/training. Dexter never performs detection in its speech worker.
    # The encoder-only path loads the exact same encoder weights and applies the
    # same watermark algorithm. Target-machine A/B saved ~29.1 MB working set /
    # 32.1 MB private and produced byte-identical PCM. Repeated full-pipeline
    # sustained-Gemma medians were also non-regressive (RTF 0.806 vs 0.833,
    # TTFA 1.116 s vs 1.155 s). Keep an upstream fallback for diagnostics.
    encoder_only_perth = _binary_flag("DEXTER_TTS_PERTH_ENCODER_ONLY", True)
    if encoder_only_perth:
        from perth_encoder_only import PerthEncoderOnlyWatermarker
        import perth

        perth.PerthImplicitWatermarker = PerthEncoderOnlyWatermarker
        print(
            "[TTS] Perth path=encoder-only",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[TTS] Perth path=upstream",
            file=sys.stderr,
            flush=True,
        )

    import llama_cpp

    original_init = llama_cpp.Llama.__init__

    def tuned_init(self, *args, **kwargs):
        kwargs["n_batch"] = batch
        kwargs["n_ubatch"] = ubatch
        return original_init(self, *args, **kwargs)

    llama_cpp.Llama.__init__ = tuned_init

    # llama-cpp-python's Llama.from_pretrained() always lists the remote
    # Hugging Face repository before it asks hf_hub_download() for the cached
    # model path. That makes an otherwise fully local Dexter startup fail during
    # transient HF/TLS outages even when the GGUF is already on disk. Prefer the
    # existing HF snapshot without network I/O, then retain upstream behavior as
    # a cache-miss fallback so first-time setup still downloads normally.
    original_from_pretrained = llama_cpp.Llama.from_pretrained

    def local_first_from_pretrained(
        cls,
        repo_id,
        filename,
        additional_files=None,
        local_dir=None,
        local_dir_use_symlinks="auto",
        cache_dir=None,
        **kwargs,
    ):
        if local_dir is None and filename:
            try:
                from huggingface_hub import snapshot_download

                snapshot = Path(
                    snapshot_download(
                        repo_id=repo_id,
                        allow_patterns=[filename],
                        cache_dir=cache_dir,
                        local_files_only=True,
                    )
                )
                matches = [
                    path
                    for path in snapshot.rglob("*")
                    if path.is_file()
                    and fnmatch.fnmatch(
                        path.relative_to(snapshot).as_posix(),
                        filename,
                    )
                ]
                if len(matches) == 1:
                    print(
                        f"[TTS] llama.cpp backbone source=local-cache ({matches[0].name})",
                        file=sys.stderr,
                        flush=True,
                    )
                    return cls(model_path=str(matches[0]), **kwargs)
                if len(matches) > 1:
                    print(
                        f"[TTS] Local HF cache has multiple matches for {repo_id}/{filename}; falling back to upstream resolver",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"[TTS] No usable local GGUF cache entry for {repo_id}: {exc}; falling back to Hugging Face",
                    file=sys.stderr,
                    flush=True,
                )

        return original_from_pretrained(
            repo_id=repo_id,
            filename=filename,
            additional_files=additional_files,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            cache_dir=cache_dir,
            **kwargs,
        )

    llama_cpp.Llama.from_pretrained = classmethod(local_first_from_pretrained)
    print(
        f"[TTS] llama.cpp backbone batch={batch}, ubatch={ubatch}, mlock=upstream-default",
        file=sys.stderr,
        flush=True,
    )

    # The launcher itself is executed from Runtime/Voice, so that directory is
    # already sys.path[0]. This preserves Dexter's local librosa compatibility
    # shim exactly as the previous direct-worker launch did.
    runpy.run_path(str(WORKER_PATH), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
