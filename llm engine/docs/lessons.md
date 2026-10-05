# What I learned / what broke

## Milestone 1 — October 5, 2026

- **Python version matters:** the host defaults to Python 3.14.4. Used an isolated Python 3.11.17 environment with verified torch 2.5.1, transformers 4.48.3, and pytest 8.3.5 instead of assuming compatible wheels for the host default.
- **Project boundaries affect pytest:** without a local configuration, pytest selected the parent repository and attempted to write its cache there. Added project-local `pytest.ini` so this project's tests and caches stay local.
- **Check library APIs:** probing GPT2Attention's old `_attn` method raised AttributeError. Inspected the installed source and executed the current `eager_attention_forward` and tiny model to verify tensor names, shapes, activation, and attention backend. The installed default is SDPA; explicitly chose eager attention as the transparent reference.
- **Conv1D transposes are easy to miss:** square attention projection matrices still need transposition. A tiny reference comparison and an explicit square-weight check cover this. Validation occurs before any parameter copy so a missing late-layer tensor cannot partially update a model.
- **Validation must cover zero work:** a zero-token generation request still needs valid tokens and shape. Reused model input validation before the loop, which avoids a wasted forward just to validate the prompt.
- **Downloads require real execution:** sandboxed package installation could not resolve PyPI. Authorized network execution installed the pinned packages and downloaded actual public GPT-2 without a token. Model files and virtual environments remain ignored.
- **Upstream warnings:** the initial weight download emitted an hf_xet deprecation warning. Cached offline tests run without it. No correctness tolerance was changed.
- **Baseline correctness:** 17 structural checks passed first, then 24 weight/logit checks. After generation, 41 tests passed and one CUDA test skipped in 9.26 seconds. Maximum absolute logit difference was 0 on three public GPT-2 prompts; greedy tokens matched HF exactly for 50 new tokens per prompt.
- **CPU demo:** `python -m engine.generate --prompt 'Hello, world!' --max-new-tokens 50 --device cpu` produced a continuation successfully. An offline run with `OMP_NUM_THREADS=1` also succeeded. Thread tuning here is a practical development setting, not a benchmark claim.

CUDA parity is unverified: this machine exposes no CUDA device. No throughput, latency, or memory benchmark is published yet. KV cache, batching, paging, quantization, serving, Docker, CI, and Colab work remain for subsequent milestones.
