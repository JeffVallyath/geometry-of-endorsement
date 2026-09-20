# Selected-case carrier mediation results

Technical disposition: SCIENTIFIC_COMPLETE. 1422 unique scored forwards retained.

One selected matched witness: SSC1-FINAL-0035, origins 3/5, original six questions, Gemma-2-9B-it and INV_PAIR_NLL_s1. This is a mechanism intervention study, not the previously proposed 12-case validation or a population estimate.

Carrier C stays V-only at layers16–23, positions426–516, all8 native KV groups. Writer component interventions use the separately identified small region426–455. The component scan therefore localizes writing only within that fixed30-position region, not exhaustively across all91 positions of C. ATTN/MLP outputs mean the residual additions after their respective Gemma post-layernorm. Layer/head IDs are zero-based.

## Writer-side findings

Frozen selected writer: L16_ATTN_ONLY; direct preservation 8/8; worst full-carrier-normalized joint effect 0.087708; worst carrier projection 0.097924.

| Intervention | Joint donor answers /4 | Direct preserved /8 | Worst joint fraction of C | Carrier projection R3 / R5 |
|---|---:|---:|---:|---|
| BANK_SELF | 0 | 8 | -0.000000 | 0.000000 / 0.000000 |
| COMPONENT_BANK | 2 | 8 | 0.246996 | 0.488653 / 0.489521 |
| L16_ATTN_LEAVE_OUT | 2 | 8 | 0.079260 | 0.442883 / 0.440837 |
| L16_ATTN_ONLY | 0 | 8 | 0.087708 | 0.100155 / 0.097924 |
| L16_MLP_LEAVE_OUT | 2 | 8 | 0.152978 | 0.441110 / 0.451674 |
| L16_MLP_ONLY | 0 | 8 | 0.054266 | 0.114348 / 0.115937 |
| L17_ATTN_LEAVE_OUT | 2 | 8 | 0.112429 | 0.444934 / 0.448866 |
| L17_ATTN_ONLY | 1 | 8 | 0.037458 | 0.089935 / 0.082226 |
| L17_MLP_LEAVE_OUT | 2 | 8 | 0.283346 | 0.460478 / 0.462862 |
| L17_MLP_ONLY | 0 | 8 | -0.081208 | 0.093712 / 0.088227 |
| L18_ATTN_LEAVE_OUT | 2 | 8 | 0.171396 | 0.461391 / 0.463238 |
| L18_ATTN_ONLY | 0 | 8 | -0.102051 | 0.068975 / 0.059164 |
| L18_MLP_LEAVE_OUT | 2 | 8 | 0.152762 | 0.470323 / 0.469715 |
| L18_MLP_ONLY | 0 | 8 | -0.013599 | 0.064550 / 0.072060 |
| L19_ATTN_LEAVE_OUT | 2 | 8 | 0.146243 | 0.473335 / 0.474468 |
| L19_ATTN_ONLY | 0 | 8 | -0.048815 | 0.063687 / 0.055175 |
| L19_MLP_LEAVE_OUT | 2 | 8 | 0.231095 | 0.478897 / 0.479130 |
| L19_MLP_ONLY | 0 | 8 | 0.028694 | 0.063398 / 0.059953 |
| L20_ATTN_LEAVE_OUT | 2 | 8 | 0.179783 | 0.476840 / 0.476131 |
| L20_ATTN_ONLY | 0 | 8 | 0.002920 | 0.049850 / 0.048452 |
| L20_MLP_LEAVE_OUT | 2 | 8 | 0.172123 | 0.483628 / 0.483487 |
| L20_MLP_ONLY | 0 | 8 | 0.001087 | 0.044461 / 0.048297 |
| L21_ATTN_LEAVE_OUT | 2 | 8 | 0.104855 | 0.482298 / 0.480203 |
| L21_ATTN_ONLY | 0 | 8 | 0.026614 | 0.032713 / 0.029102 |
| L21_MLP_LEAVE_OUT | 2 | 8 | 0.217290 | 0.487528 / 0.486720 |
| L21_MLP_ONLY | 0 | 8 | -0.013895 | 0.030205 / 0.032482 |
| L22_ATTN_LEAVE_OUT | 2 | 8 | 0.204002 | 0.487200 / 0.487508 |
| L22_ATTN_ONLY | 0 | 8 | 0.002954 | 0.010102 / 0.010345 |
| L22_MLP_LEAVE_OUT | 2 | 8 | 0.142760 | 0.487863 / 0.488142 |
| L22_MLP_ONLY | 0 | 8 | -0.041670 | 0.014159 / 0.016092 |
| L23_ATTN_LEAVE_OUT | 2 | 8 | 0.251192 | 0.488653 / 0.489521 |
| L23_ATTN_ONLY | 0 | 8 | -0.002449 | 0.000000 / 0.000000 |
| L23_MLP_LEAVE_OUT | 2 | 8 | 0.247654 | 0.488653 / 0.489521 |
| L23_MLP_ONLY | 0 | 8 | -0.010329 | 0.000000 / 0.000000 |
| POST15_COMPLEMENT | 0 | 8 | -0.004296 | 0.162489 / 0.153539 |
| POST15_DIFFERENT | 0 | 8 | -0.205946 | 0.172806 / 0.163133 |
| POST15_FULL | 4 | 8 | 0.938392 | 1.000000 / 1.000000 |
| POST15_SELF | 0 | 8 | -0.000000 | 0.000000 / 0.000000 |
| POST15_SMALL_SHARED | 4 | 8 | 1.085854 | 0.681906 / 0.667893 |

Carrier projections are descriptive tensor-direction alignment, not fractions of a unique causal mechanism. Every direction/format score, per-layer carrier change and small-region change is retained in the machine-readable tables. WRITER_COMPONENT_MEDIATION.json additionally reports each LEAVE_OUT loss relative to the full component BANK, with signed and undefined fractions retained. ONLY layer23 is a structural negative control for carrierV16–23, not for later caches or answers.

## Question-side reader

Frozen set: TOP8 = [[19, 15], [18, 3], [23, 12], [20, 11], [21, 0], [23, 14], [17, 0], [21, 9]]. Training disposition: NO_QUALIFIED_READER_SET.

All128 individual query heads were ranked using only literal recipient3<-5 joint results: min(BLOCK loss, INJECT gain), normalized by the original carrier effect. Each native KV group feeds two query heads; individual-head clamps preserve its sibling. Sets and the training choice were frozen before coded/reverse intervention checks. Those checks are not fresh population confirmation: the case, baselines and writer outcomes were already exposed.

| Set | Recipient / format | Split | BLOCK fraction | INJECT fraction | Min |
|---|---|---|---:|---:|---:|
| TOP1 | 3 / d0 | TRAINING | 0.324499 | 0.118557 | 0.118557 |
| TOP1 | 3 / d1 | CODED_OR_REVERSE_CHECK | 0.298485 | 0.082206 | 0.082206 |
| TOP1 | 5 / d0 | CODED_OR_REVERSE_CHECK | 0.279429 | 0.437717 | 0.279429 |
| TOP1 | 5 / d1 | CODED_OR_REVERSE_CHECK | 0.181769 | 0.360736 | 0.181769 |
| TOP2 | 3 / d0 | TRAINING | 0.480724 | 0.212784 | 0.212784 |
| TOP2 | 3 / d1 | CODED_OR_REVERSE_CHECK | 0.466152 | 0.134264 | 0.134264 |
| TOP2 | 5 / d0 | CODED_OR_REVERSE_CHECK | 0.446316 | 0.640667 | 0.446316 |
| TOP2 | 5 / d1 | CODED_OR_REVERSE_CHECK | 0.330068 | 0.631494 | 0.330068 |
| TOP4 | 3 / d0 | TRAINING | 0.743562 | 0.563267 | 0.563267 |
| TOP4 | 3 / d1 | CODED_OR_REVERSE_CHECK | 0.857469 | 0.609085 | 0.609085 |
| TOP4 | 5 / d0 | CODED_OR_REVERSE_CHECK | 0.547622 | 0.681122 | 0.547622 |
| TOP4 | 5 / d1 | CODED_OR_REVERSE_CHECK | 0.608049 | 0.771157 | 0.608049 |
| TOP8 | 3 / d0 | TRAINING | 0.888235 | 0.777757 | 0.777757 |
| TOP8 | 3 / d1 | CODED_OR_REVERSE_CHECK | 0.950477 | 0.751592 | 0.751592 |
| TOP8 | 5 / d0 | CODED_OR_REVERSE_CHECK | 0.712414 | 0.833726 | 0.712414 |
| TOP8 | 5 / d1 | CODED_OR_REVERSE_CHECK | 0.726941 | 0.894122 | 0.726941 |

The0.8 criterion selects a reader set using training only; it is not a post-hoc overall mechanism-success rule. All direct checks, including failures, remain in ALL_CONDITION_RESULTS.json and ADVERSE_AND_OMISSIONS.json.

## Loss and rescue chain

A=native; B=selected writerONLY; C=B with recipientC restored; D=same B background with ORIGINAL donorC re-injected; E=original donorC on native recipient with selected reader blocked; F=same-site carrier-active reader output restored. D is not B-produced-carrier reconstruction. F functionally undoes a clamp at the same site; it is an expected restoration control, not independent natural-path proof.

| Recipient / format | Writer B-A | Carrier loss B-C | Loss / writer | Rescue D-C | D-B | Reader RC-E / RC-A | Restore F-E | F-RC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 / d0 | 1.159842 | 0.721357 | 0.621945 | 9.572723 | 8.851366 | 0.888235 | 8.870285 | -0.000000 |
| 3 / d1 | 0.686537 | 0.285986 | 0.416563 | 7.767263 | 7.481277 | 0.950477 | 7.439846 | -0.000000 |
| 5 / d0 | 1.989258 | 1.685175 | 0.847137 | 8.083461 | 6.398286 | 0.712414 | 5.604391 | 0.000000 |
| 5 / d1 | 2.194122 | 1.778799 | 0.810711 | 7.312862 | 5.534063 | 0.726941 | 5.240269 | 0.000000 |

Signed margins, negative effects, overshoot and zero-denominator undefined fractions are retained without clipping. Direct-row donor direction can be small and is not treated as a meaningful mediation target; direct preservation is assessed against the same recipient.

| Chain condition | Joint donor answers /4 | Direct preserved /8 |
|---|---:|---:|
| A | 0 | 8 |
| B | 0 | 8 |
| C | 0 | 8 |
| D | 4 | 8 |
| E | 0 | 8 |
| F | 4 | 8 |

## Scope, adverse outcomes and reproducibility

0 of 948 unique direct score rows fail the full preservation rule; 0 scores are invalid. These counts include fixed scans and controls, not independent samples. Every adverse cell is retained.
Direct preservation requires valid/correct, P(correct)>=0.95, candidate mass>=0.90 and absolute same-recipient probability drift<=0.02. P(Yes) is normalized between the two fixed candidate labels, not a calibrated belief. An intended donor transfer can create an incorrect answer.

No universal SHS mechanism, old-fact neuron, unique/minimum circuit, natural-path necessity, or generality to Ripple/CSP80 is established. No training, new cases, alternate model, scientific retry, push, publication or automatic successor is part of this study.

Raw results and actual native tensors remain unchanged. Tensor companions contain lossless safetensors with shape/dtype/native-byte indexes. NUMERICAL_AUDIT.json records independent score/ranking/tensor recomputation; lifecycle closure and cost are recorded separately.
