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

### Final independent review

The reviewer independently reran the original suite (41 passed, one CUDA skip) and found two observable bugs:

- **HF can hide missing weights:** `from_pretrained()` fills omitted checkpoint tensors with random values, so validating its resulting `state_dict()` alone is insufficient. A real damaged safetensors checkpoint reproduced the acceptance bug. The loader now inspects `output_loading_info` and rejects missing required keys before copying; diagnostics include expected tensor shapes. A normally omitted duplicate tied output-head weight remains legitimate. The direct mapper also reports the expected shape for missing tensors.
- **Decoding can alter a prompt:** decoding the full sequence with `skip_special_tokens=True` removed literal `<|endoftext|>` text supplied by the user, even for zero new tokens. A real-tokenizer CLI test reproduced this. The CLI now preserves original prompt text and decodes only newly generated IDs. A tiny real-forward check separately verifies the empty-prompt seed.

All regression failures were observed before their fixes. After fixes: **44 passed, one CUDA skip**, all three logit errors remained 0, and the CPU CLI preserved `Hello <|endoftext|> world` with zero new tokens. Safetensors 0.8.0 is now pinned because the on-disk regression directly imports it. The damaged-checkpoint test intentionally triggers HF's missing-weight diagnostic before our rejection.

### Execution decisions

- Kept the explicitly requested directory and used a local feature branch instead of moving to another checkout. A later relocation would need path changes.
- Kept the execution ledger in this project's ignored scratch directory, avoiding writes to unrelated parent-project tooling. Generic parent-workspace tooling will not discover that ledger automatically.
- Scoped pytest to this project; parent repository tests are outside this engine's validation.
- Used HF eager attention as the FP32 oracle; optional fused backends and CUDA still need their own hardware validation.
- Shared token validation between forward and generation so zero-token requests are validated without a redundant forward; this adds one small model method to maintain.

Later milestones and CUDA execution remain outside this review. Full output-budget reservation, the eager reference, legitimate tied-head omission, and preserved download exceptions follow the approved scope. No review findings remain deferred. Work stays on `codex/mini-infer-m1` without merging or pushing.
