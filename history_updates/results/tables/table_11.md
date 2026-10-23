# Common-state reader comparisons

Cases /128, targets /384, controls /517, original-snapshot targets /1536. Ranking and PROGRAM share exact execution.

| Reader | Source_endpoint | Complete_cases | Targets | Controls | Original_targets |
|---|---|---|---|---|---|
| Ranking, seed 0 | REPAIR_0 | 124 | 380 | 517 | 1522 |
| Ranking, seed 1 | REPAIR_1 | 127 | 383 | 517 | 1524 |
| Generates operation, seed 0 | PROGRAM_0 | 126 | 383 | 516 | 1532 |
| Generates operation, seed 1 | PROGRAM_1 | 126 | 384 | 515 | 1530 |
| Generates answer, seed 0 | QA_0 | 126 | 383 | 516 | 1266 |
| Generates answer, seed 1 | QA_1 | 122 | 382 | 513 | 1068 |
| MeLLo, adapted | MELLO_CURRENT | 45 | 287 | 466 | 1166 |
