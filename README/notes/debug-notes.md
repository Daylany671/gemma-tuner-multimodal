# Gemma Fine-Tuning Debug Notes

Institutional memory for Gemma (especially Gemma 4) LoRA fine-tuning on Apple Silicon: MPS memory, text collators, and long-input summarization. Multiple related issues live in this file; each issue is self-contained.

**Quick filter:** `grep -n "— Active" README/notes/debug-notes.md`

---

## Issue: Zero loss from degenerate `assistant_masks` (all zeros) — Resolved

**First spotted:** 2026-04-07
**Resolved:** 2026-04-07
**Status:** Resolved

### Summary

`apply_chat_template(..., return_assistant_tokens_mask=True)` can return `assistant_masks` with **no supervised positions** (often when the chat template lacks the Jinja `generation` keyword; HF logs a warning). The old collator path did `labels[am == 0] = ignore`, which masks **every** token when `am` is all zeros → `loss=0`, `grad_norm=0`. **Fix:** per row, use `assistant_masks` only when `am[i].sum() > 0`; otherwise call `mask_gemma_prompt_tokens` for that row. One-time warning when fallback triggers.

### Symptom

```log
{'loss': '0', 'grad_norm': '0', 'learning_rate': '9.981e-06', 'epoch': '0.01363'}
return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword.
```

### Root Cause

`assistant_masks` present with correct shape but **all zeros**; combined with unconditional `labels[am == 0]` masking, every label became `IGNORE_TOKEN_ID`.

### Related Guides

- [Gemma 4 on Apple Silicon](../guides/apple-silicon/gemma4-guide.md) — Text SFT and label masking.

### Fix

**Files:**

- `gemma_tuner/models/common/collators.py` — `DataCollatorGemmaText._collate_instruction`: per-row degenerate `assistant_masks` handling + `_warned_degenerate_assistant_mask`.
- `tests/test_gemma_text_collator.py` — `test_instruction_gemma4_degenerate_all_zero_assistant_masks_uses_fallback`.

### Verification

```bash
.venv/bin/python -m pytest tests/test_gemma_text_collator.py -q
```

---

## Issue: Long-input summarization — truncation drops assistant span — Resolved

**First spotted:** 2026-04-07
**Resolved:** 2026-04-07
**Status:** Resolved

### Summary

For summarization, **right-side truncation** at `max_length` keeps the **beginning** of the chat; very long user prompts can fill the entire window so the `<|turn>model` / assistant content never appears. That is separate from the degenerate-mask bug. **Fix:** before `apply_chat_template`, measure token length per row and **left-truncate the user prompt** (drop characters from the start) until the chat fits `max_length`, then apply the usual HF truncation as a safety net. Optional `instruction_truncation="none"` restores tokenizer-only truncation. Very long assistant targets still need a smaller response or a larger `max_seq_length`; the collator may **prefix-truncate** the assistant string if the pair still does not fit. Chunking / hierarchical summarization for multi-window documents remains a data-pipeline concern.

### Root Cause

Tokenizer `truncation=True` + `max_length` applies to the full rendered chat; long prompts crowd out the assistant turn.

### Fix

**Files:**

- `gemma_tuner/models/common/collators.py` — `instruction_truncation` (default `"left_user"`), `_instruction_seq_len`, `_fit_instruction_pair_to_max_length`, and integration in `_collate_instruction`.
- `tests/test_gemma_text_collator.py` — `_FakeLongPromptTokenizer` plus tests for `left_user` vs `none`.

### Verification

```bash
.venv/bin/python -m pytest tests/test_gemma_text_collator.py -q
```

---

## Issue: MPS OOM during Gemma 4 multimodal text LoRA training — Resolved

**First spotted:** 2026-04-07
**Resolved:** 2026-04-07
**Status:** Resolved

### Summary

First training step crashed with `RuntimeError: MPS backend out of memory` (~41 GiB allocated) while fine-tuning `gemma-4-e2b-it` with LoRA on text CSV data. Mitigations: **MPS + text** forces `per_device_train_batch_size=1` and `per_device_eval_batch_size=1`, enables **gradient checkpointing** by default for MPS text (opt out via `GEMMA_MPS_ALLOW_NO_GRADIENT_CHECKPOINTING`), passes `gradient_checkpointing` into `TrainingArguments`, calls `enable_input_require_grads()` when checkpointing is on, and caps `max_seq_length` on MPS text using `GEMMA_MPS_MAX_SEQ_LENGTH` (default 2048). Added profile `gemma-csv-gemma4-mps-text` in `config.ini`.

### Symptom

```log
RuntimeError: MPS backend out of memory (MPS allocated: 40.96 GiB, ... max allowed: 41.47 GiB).
```

### Root Cause

Unified memory pressure from **Gemma 4 multimodal** weights + **LoRA** + **eager attention** + **micro-batch > 1** and **no gradient checkpointing** on long sequences. Confirmed by stack trace in attention softmax during the first step.

### Related Guides

- [Gemma 4 on Apple Silicon](../guides/apple-silicon/gemma4-guide.md) — MPS allocator, `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, gradient checkpointing tradeoffs.

### Fix

**Files:**

- `gemma_tuner/models/gemma/finetune.py` — MPS text batch clamp, gradient checkpointing wiring, optional `GEMMA_MPS_ALLOW_NO_GRADIENT_CHECKPOINTING`, `GEMMA_MPS_MAX_SEQ_LENGTH` cap, `enable_input_require_grads`.
- `config.ini` — `[profile:gemma-csv-gemma4-mps-text]` with conservative batch/accumulation settings.

### Verification

```bash
gemma-macos-tuner finetune gemma-csv-gemma4-mps-text --json-logging
# Expect first training steps to progress without immediate MPS OOM (still subject to seq length / data).
```

---

## Appendix: Practical context limits (64 GB unified RAM, this stack)

**Not a bug ticket** — planning guidance from the same investigation.

- Causal LM **training** uses one sequence budget for **prompt + response** (plus chat template overhead). “8192 input” is not a separate bucket from the summary.
- For **`gemma-4-e2b-it` LoRA on MPS** in this repo, treat **~2k–3k total tokens** as a realistic training target unless proven otherwise; **~10k total** is not a safe default for local training on this hardware (attention memory scales roughly with sequence length squared).
- **Data:** median prompt length on sampled `gemma-csv` rows was on the order of **~8–9k tokens** (prompt-only encode); many rows exceed any single-window cap without **chunking**, **hierarchical summarization**, or **left-truncating** the source only.

<!--
USAGE NOTES (from template):

1. Prefer updating this file before creating new note files for the same domain.
2. New issue? Copy the "Active" template and paste at the top (newest first).
3. Fixed: rename "— Active" to "— Resolved" and add Resolved date.
4. Investigation Log: trim ruled-out hypotheses after resolution.
-->
