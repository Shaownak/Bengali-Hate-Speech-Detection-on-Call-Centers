# ADR-021: LLM judge fine-tuning on a 12 GB consumer GPU with open data (Prompt 9)

- **Status:** Accepted
- **Date:** 2026-11-03
- **Deciders:** Claude Code (Prompt 9), per CLAUDE.md "spec ambiguous → simplest option + ADR";
  owner request "start the LLM part on my RTX 3060 12 GB with open-source data"
- **Spec refs:** docs/03 §7, docs/08, docs/09 §7, ADR-008, ADR-009, ADR-012 (D-20), D-12

## Context
docs/03 §7.3 plans a Qwen3.5-9B bf16 LoRA (≈ 22 GB VRAM) with Unsloth on a 48 GB card and in-house
gold windows. What is available now is a native-Windows PC with an RTX 3060 (12 GB, Ampere, bf16)
and no in-house data. vLLM does not run on native Windows, and Unsloth needs triton-windows, which
is fragile. The existing open-synthetic SFT view (Prompt 4b) has only customer-side text events
from 2 datasets, no provisional decisions, no signals and no agent behaviour, so a judge trained on
it would just be a text classifier.

## Decision
1. **Base model: Qwen3.5-4B, bf16 LoRA** (r 16, α 32, dropout 0.05, all linear layers, lr 1e-4
   cosine, loss on assistant tokens, packing off, thinking off). The 4B model is the largest Qwen3.5
   that fits bf16 on 12 GB with gradient checkpointing. It keeps docs/03's advice against 4-bit
   QLoRA on Qwen3.5. Qwen3.5-0.8B is the smoke model. 9B stays the target on the GPU box; on this
   card it only fits with `--load-4bit` (nf4 QLoRA), an opt-in fallback that is recorded in the
   model card. Pins: `qwen3_5_4b` @ `851bf6e8…`, `qwen3_5_0_8b` @ `2fc06364…` (Apache-2.0).
2. **Plain PyTorch + PEFT trainer** (`ml/llm/train.py`), not Unsloth or TRL:
   - batch 1 with gradient accumulation, so there is no padding;
   - logits only at assistant positions (`logits_to_keep`): with a 150–250k vocabulary,
     full-sequence fp32 logits alone would cost 1.5 GB per window;
   - an eval callback that generates on dev windows and keeps the best adapter by event F1.

   Unsloth remains an option on the Linux GPU box.
3. **Separate GPU venv** `.gale/venv-llm` (`make llm-venv`): torch from the CUDA 12.8 index plus
   `ml/llm/requirements-gpu.txt`, outside the uv workspace lock (D-12), so CI and the CPU venv are
   unchanged. All CUDA, torch and HF caches point into `.gale/` (`CUDA_CACHE_PATH`,
   `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `HF_HOME`) because C: has no space.
4. **More open data.**
   - Four CC BY 4.0 Mendeley datasets are now auto-downloaded with sha256 pins: T3 BanglaCAT,
     T5 Cyberbully-multilabel, T12 BanglaHarassNet and R2 Code-mixed Chaos. They are public
     without login.
   - The downloader unpacks archives, refusing zip-slip members.
   - The xlsx reader skips title rows.
   - `commercial_ok` stays `unknown` until the legal review (Q-04), even though the pages say
     CC BY 4.0.
   - With T1, T6 and T8, 7 text sources feed the judge.
5. **SFT window builder v2** (`ml/llm/sft.py`). It composes call windows around open-data
   utterances, with templated Bangla call-centre turns (`templates_bn.yaml`, hand-written, no
   slurs), simulated fast-path signals and provisional events, rendered by the production
   `render_user_message`. Target conventions match fusion (`apply_judgements`, `llm_events`):
   - a provisional event gets a decision and is not repeated in `events`;
   - `shouting` / `harsh_tone` appear only above the acoustic thresholds (rule 5);
   - reported speech gets `context_flags` and severity 1, and is dismissed when provisional;
   - injected instructions are ignored.

   Scenario mix:

   | Scenario | Share |
   |---|---|
   | Customer abuse | 36 % |
   | Agent abuse | 10 % |
   | Agent register / dismissive / sarcasm | 10 % |
   | Reported speech | 5 % |
   | Hard negatives (open-data negatives, complaints, banter, policy) | 31 % |
   | Prompt injection | 3 % |
   | Two abusive turns | 3 % |
   | Customer register | 2 % |

   Default size: 12k train / 400 dev / 400 test windows, from the unified set's leakage-controlled
   splits.
6. **Everything trained here is `research_only`** (ADR-012 / D-20) and cannot be promoted.

7. **Evidence grounding is exact up to Unicode form** (found at the first dev eval: 71 % of
   events were "hallucinated").
   - `galecore.bangla.normalize()` keeps the precomposed nukta letters ড় / ঢ় / য়, while Qwen's
     tokenizer applies NFC, which decomposes them (they are composition exclusions). Copied
     evidence therefore comes back in another form.
   - `find_canonical()` matches by the Unicode part of `normalize()`, and the grounding check and
     metrics use it. The signal still stores the transcript's own span.
   - Without this, production would have dropped real events as `llm_ungrounded`. 82 of 400 dev
     windows were affected.
   - SFT windows are now built from `normalize()`d text, as the ASR worker produces.

## Consequences
- The judge can be trained and measured on this PC. Training at roughly 1k tokens/window takes
  hours, not days.
- 4B is weaker than 9B. The bake-off on the GPU box (docs/03 §7.1) still decides the production
  base.
- Agent-side behaviour without keywords comes from a small template set, so the model will
  generalise only partly to real agent rudeness. In-house T2/T3 gold (P2) replaces this before any
  promotion.
- Signals and provisional events are simulated: real fast-path score distributions will differ.
  Fusion calibration (Prompt 10) and in-house gold close that gap.
- Serving: vLLM (guided JSON) stays the production path on Linux. On this PC, evaluation uses
  transformers generation, and JSON validity comes from the fine-tune plus the existing
  post-validation, with no guided decoding.
