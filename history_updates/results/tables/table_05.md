# Cache refresh accounting

Counts retain both seeds, all scheduled questions and every correction or harm.

| Measurement | Count | Source_field |
|---|---|---|
| Saved score rows | 435200 | raw_rows |
| Eligible witness observations before refresh | 83 | baseline_W |
| Witnesses repaired, blocks 17-24 | 36 | B_repairs |
| Witnesses repaired, control blocks 25-32 | 2 | C_repairs |
| Eligible witnesses after refresh | 75 | treated_qualifying_witnesses |
| Original witnesses still present | 35 | baseline_witnesses_retained |
| New witnesses after refresh | 40 | new_qualifying_witnesses |
| Lost qualification without repair | 12 | baseline_ceased_qualifying_without_repair |
| Direct answers made wrong | 76 | B_direct_damage |
| Direct answers corrected | 84 | B_direct_corrections |
| Combined-answer errors before refresh | 3490 | A_joint_errors |
| Combined-answer errors after refresh | 2893 | B_joint_errors |
| Combined answers made wrong | 347 | B_joint_damage |
| Combined answers corrected | 944 | B_joint_corrections |
| Combined-answer errors after control-region refresh | 3494 | C_joint_errors |
| Combined-answer errors in donor reference | 2000 | D_joint_errors |
| Cases | 256 | cases |
| Cases with witnesses after refresh | 58 | treated_witness_cases |
| Distinct designs with repaired witnesses | 28 | repaired_designs |
