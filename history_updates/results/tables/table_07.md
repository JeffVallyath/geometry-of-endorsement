# Complete cases under each repair condition

Every target and control must be correct under all four histories; each seed remains separate.

| Model | Condition | Source_endpoint | Complete_cases | Cases |
|---|---|---|---|---|
| Qwen2.5-7B | Full history | HISTORY | 11 | 128 |
| Qwen2.5-7B | Note before question | REFLECT | 12 | 128 |
| Qwen2.5-7B | Corrected facts, text | CURRENT_TEXT | 17 | 128 |
| Qwen2.5-7B | Corrected facts, JSON | CURRENT_JSON | 14 | 128 |
| Qwen2.5-7B | Dictionary, untrained | STATE_BASE | 5 | 128 |
| Qwen2.5-7B | Dictionary, trained seed 0 | STATE_0 | 124 | 128 |
| Qwen2.5-7B | Dictionary, trained seed 1 | STATE_1 | 127 | 128 |
| Qwen2.5-32B | Corrected facts, JSON | CURRENT_JSON | 32 | 128 |
| Qwen2.5-32B | Corrected facts, text | CURRENT_TEXT | 48 | 128 |
| Qwen2.5-32B | Full history | NATIVE32_SAVED | 89 | 128 |
| Qwen2.5-32B | Dictionary, trained seed 0 | STATE_0 | 128 | 128 |
| Qwen2.5-32B | Dictionary, trained seed 1 | STATE_1 | 126 | 128 |
| Qwen2.5-32B | Dictionary, untrained | STATE_BASE | 11 | 128 |
| Gemma-2-27B | Corrected facts, JSON | CURRENT_JSON | 59 | 128 |
| Gemma-2-27B | Corrected facts, text | CURRENT_TEXT | 79 | 128 |
| Gemma-2-27B | Full history | NATIVE_GEMMA | 62 | 128 |
| Gemma-2-27B | Dictionary, trained seed 0 | STATE_0 | 127 | 128 |
| Gemma-2-27B | Dictionary, trained seed 1 | STATE_1 | 127 | 128 |
| Gemma-2-27B | Dictionary, untrained | STATE_BASE | 28 | 128 |
