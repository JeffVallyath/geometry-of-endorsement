# Traceable saved-output examples

These are selected illustrations, not prevalence estimates. Each example is selected solely from the independent calculation, without consulting expected outputs. `examples.json` contains full-precision scores, token bindings and journal line references. Manifest entries identify each original source archive/member.

Relation tables use 1 for supports and 0 for opposes. Actor and project order is explicit. Candidate log probabilities are ordered by the displayed answer mapping. All displayed semantic Yes/No answers are decoded from that mapping.

## repeat_checked_positive

qwen / `FREE_PAIR_CONSISTENCY_s0` / `SSC1-FINAL-0006` / `SSC1-FINAL-0006-d0-02`

Actors: Dion, Zara, Mina, Wren, Orla, Luca. Projects: Garden, Meadow.

Commands (actor/project indices and assigned values): `[{"actor": 3, "project": 0, "value": 1}, {"actor": 4, "project": 0, "value": 0}, {"actor": 5, "project": 0, "value": 0}]`.

Complete common final table: `[[0, 1], [1, 1], [0, 1], [1, 0], [0, 0], [0, 1]]`.

```text
Do Wren and Orla take the same side on Garden?
Answer with exactly one of No, Yes.
```

Candidate order: `['No', 'Yes']`.

| History | Complete starting table | Direct operand answers; P(correct) | Joint answer | Joint candidate logps |
|---|---|---|---|---|
| 0 | `[[0, 1], [1, 1], [0, 1], [1, 0], [0, 0], [0, 1]]` | Wren: Yes; 1.0000000000 / Orla: No; 0.9999999987 | No | `[-0.00011681827891152352, -9.055376052856445]` |
| 1 | `[[0, 1], [1, 1], [0, 1], [1, 0], [0, 0], [1, 1]]` | Wren: Yes; 1.0000000000 / Orla: No; 0.9999999985 | No | `[-0.0002549561613705009, -8.274744987487793]` |
| 2 | `[[0, 1], [1, 1], [0, 1], [1, 0], [1, 0], [0, 1]]` | Wren: Yes; 1.0000000000 / Orla: No; 0.9999999818 | No | `[-1.9073468138230965e-06, -13.148210525512695]` |
| 3 | `[[0, 1], [1, 1], [0, 1], [1, 0], [1, 0], [1, 1]]` | Wren: Yes; 1.0000000000 / Orla: No; 0.9999999950 | No | `[-3.4570634852570947e-06, -12.580806732177734]` |
| 4 | `[[0, 1], [1, 1], [0, 1], [0, 0], [0, 0], [0, 1]]` | Wren: Yes; 0.9999998154 / Orla: No; 0.9999999881 | Yes | `[-4.4778571128845215, -0.011422710493206978]` |
| 5 | `[[0, 1], [1, 1], [0, 1], [0, 0], [0, 0], [1, 1]]` | Wren: Yes; 0.9994579196 / Orla: No; 0.9999999784 | Yes | `[-9.162813186645508, -0.00010489867418073118]` |
| 6 | `[[0, 1], [1, 1], [0, 1], [0, 0], [1, 0], [0, 1]]` | Wren: Yes; 0.9999999928 / Orla: No; 0.9999999511 | No | `[-0.11165868490934372, -2.2476181983947754]` |
| 7 | `[[0, 1], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1]]` | Wren: Yes; 0.9999910957 / Orla: No; 0.9999999639 | Yes | `[-5.230648994445801, -0.005364428274333477]` |

Direct questions: Does Wren favor the Garden proposal? / Does Orla favor the Garden proposal?

Native-final: No; logps `[0.0, -19.984516143798828]`; source journal line 15, physical ID `NATIVE_FINAL|SSC1-FINAL-0006-d0-02`.
Same-method no-op: No; logps `[-0.00011681827891152352, -9.055376052856445]`; source journal line 117, physical ID `o0/FREE_PAIR_CONSISTENCY_s0|SSC1-FINAL-0006-d0-02`.

Independently computed flags: `{"A": true, "C": true, "D": true, "W": true, "strong": true, "native_correct": true, "noop_correct": true}`.

Inputs: [inputs/designs/SSC1-FINAL-0006.json](../inputs/designs/SSC1-FINAL-0006.json) and the response shard named in `examples.json`.

The exact joint query has two saved fresh passes across all histories.

## textual_positive

gemma / `EXISTING_CORRECTION` / `SSC1-FINAL-0000` / `SSC1-FINAL-0000-d0-02`

Actors: Hana, Zara, Ari, Ada. Projects: Library, Harbor.

Commands (actor/project indices and assigned values): `[{"actor": 2, "project": 0, "value": 0}, {"actor": 3, "project": 0, "value": 0}, {"actor": 0, "project": 0, "value": 0}]`.

Complete common final table: `[[0, 1], [1, 0], [0, 0], [0, 0]]`.

```text
Do Ari and Ada take the same side on Library?
Answer with exactly one of No, Yes.
```

Candidate order: `['No', 'Yes']`.

| History | Complete starting table | Direct operand answers; P(correct) | Joint answer | Joint candidate logps |
|---|---|---|---|---|
| 0 | `[[0, 1], [1, 0], [0, 0], [0, 0]]` | Ari: No; 0.9998822392 / Ada: No; 0.9998937945 | Yes | `[-8.036063194274902, -0.0005173536483198404]` |
| 1 | `[[1, 1], [1, 0], [0, 0], [0, 0]]` | Ari: No; 0.9999273059 / Ada: No; 0.9999155480 | Yes | `[-7.159038543701172, -0.0010833829874172807]` |
| 2 | `[[0, 1], [1, 0], [0, 0], [1, 0]]` | Ari: No; 0.9998616051 / Ada: No; 0.9999301417 | Yes | `[-2.6085126399993896, -0.07801783829927444]` |
| 3 | `[[1, 1], [1, 0], [0, 0], [1, 0]]` | Ari: No; 0.9998858569 / Ada: No; 0.9999426865 | Yes | `[-3.0904746055603027, -0.0478663295507431]` |
| 4 | `[[0, 1], [1, 0], [1, 0], [0, 0]]` | Ari: No; 0.9999479744 / Ada: No; 0.9982873334 | Yes | `[-7.383187770843506, -0.0009112972766160965]` |
| 5 | `[[1, 1], [1, 0], [1, 0], [0, 0]]` | Ari: No; 0.9999524567 / Ada: No; 0.9988506901 | Yes | `[-6.765629291534424, -0.0015236446633934975]` |
| 6 | `[[0, 1], [1, 0], [1, 0], [1, 0]]` | Ari: No; 0.9999675169 / Ada: No; 0.9999578040 | No | `[-0.02088320627808571, -3.918302297592163]` |
| 7 | `[[1, 1], [1, 0], [1, 0], [1, 0]]` | Ari: No; 0.9999734218 / Ada: No; 0.9999674590 | No | `[-0.04993171989917755, -3.038806200027466]` |

Direct questions: Does Ari favor the Library proposal? / Does Ada favor the Library proposal?

Native-final: Yes; logps `[-8.6873140335083, -0.00024828212917782366]`; source journal line 9, physical ID `NATIVE_FINAL|SSC1-FINAL-0000-d0-02`.
Same-method no-op: Yes; logps `[-8.036063194274902, -0.0005173536483198404]`; source journal line 179, physical ID `o0/EXISTING_CORRECTION|SSC1-FINAL-0000-d0-02`.

Independently computed flags: `{"A": true, "C": true, "D": true, "W": true, "strong": true, "native_correct": true, "noop_correct": true}`.

Inputs: [inputs/designs/SSC1-FINAL-0000.json](../inputs/designs/SSC1-FINAL-0000.json) and the response shard named in `examples.json`.

No repeat claim is made for this particular example.

## eligible_non_witness

gemma / `INV_PAIR_NLL_s0` / `SSC1-FINAL-0000` / `SSC1-FINAL-0000-d0-02`

Actors: Hana, Zara, Ari, Ada. Projects: Library, Harbor.

Commands (actor/project indices and assigned values): `[{"actor": 2, "project": 0, "value": 0}, {"actor": 3, "project": 0, "value": 0}, {"actor": 0, "project": 0, "value": 0}]`.

Complete common final table: `[[0, 1], [1, 0], [0, 0], [0, 0]]`.

```text
Do Ari and Ada take the same side on Library?
Answer with exactly one of No, Yes.
```

Candidate order: `['No', 'Yes']`.

| History | Complete starting table | Direct operand answers; P(correct) | Joint answer | Joint candidate logps |
|---|---|---|---|---|
| 0 | `[[0, 1], [1, 0], [0, 0], [0, 0]]` | Ari: No; 0.9999925125 / Ada: No; 0.9999714683 | Yes | `[-7.419730186462402, -0.0007695574313402176]` |
| 1 | `[[1, 1], [1, 0], [0, 0], [0, 0]]` | Ari: No; 0.9999921439 / Ada: No; 0.9999587883 | Yes | `[-7.317154884338379, -0.0008509114268235862]` |
| 2 | `[[0, 1], [1, 0], [0, 0], [1, 0]]` | Ari: No; 0.9999931377 / Ada: No; 0.9999885504 | Yes | `[-5.690515995025635, -0.00370352272875607]` |
| 3 | `[[1, 1], [1, 0], [0, 0], [1, 0]]` | Ari: No; 0.9999928445 / Ada: No; 0.9999887401 | Yes | `[-5.373970985412598, -0.005021816119551659]` |
| 4 | `[[0, 1], [1, 0], [1, 0], [0, 0]]` | Ari: No; 0.9999910611 / Ada: No; 0.9997642134 | Yes | `[-5.860389232635498, -0.0031657125800848007]` |
| 5 | `[[1, 1], [1, 0], [1, 0], [0, 0]]` | Ari: No; 0.9999922268 / Ada: No; 0.9999317136 | Yes | `[-5.821439266204834, -0.0032955880742520094]` |
| 6 | `[[0, 1], [1, 0], [1, 0], [1, 0]]` | Ari: No; 0.9999906102 / Ada: No; 0.9999716701 | Yes | `[-5.192517280578613, -0.0060508400201797485]` |
| 7 | `[[1, 1], [1, 0], [1, 0], [1, 0]]` | Ari: No; 0.9999917963 / Ada: No; 0.9999778727 | Yes | `[-5.367076873779297, -0.0051108901388943195]` |

Direct questions: Does Ari favor the Library proposal? / Does Ada favor the Library proposal?

Native-final: Yes; logps `[-8.6873140335083, -0.00024828212917782366]`; source journal line 9, physical ID `NATIVE_FINAL|SSC1-FINAL-0000-d0-02`.
Same-method no-op: Yes; logps `[-7.419730186462402, -0.0007695574313402176]`; source journal line 43, physical ID `o0/INV_PAIR_NLL_s0|SSC1-FINAL-0000-d0-02`.

Independently computed flags: `{"A": true, "C": true, "D": false, "W": false, "strong": false, "native_correct": true, "noop_correct": true}`.

Inputs: [inputs/designs/SSC1-FINAL-0000.json](../inputs/designs/SSC1-FINAL-0000.json) and the response shard named in `examples.json`.

No repeat claim is made for this particular example.

## excluded_direct_operand

qwen / `EXISTING_CORRECTION` / `SSC1-FINAL-0004` / `SSC1-FINAL-0004-d1-02`

Actors: Juno, Eli, Faye, Tavi. Projects: Meadow, Museum.

Commands (actor/project indices and assigned values): `[{"actor": 3, "project": 0, "value": 1}, {"actor": 0, "project": 0, "value": 0}, {"actor": 1, "project": 0, "value": 0}]`.

Complete common final table: `[[0, 0], [0, 0], [1, 0], [1, 1]]`.

```text
Do Tavi and Juno take the same side on Meadow?
Use A for yes and B for no.
Answer with exactly one of B, A.
```

Candidate order: `['B', 'A']`.

| History | Complete starting table | Direct operand answers; P(correct) | Joint answer | Joint candidate logps |
|---|---|---|---|---|
| 0 | `[[0, 0], [0, 0], [1, 0], [1, 1]]` | Tavi: Yes; 1.0000000000 / Juno: No; 0.9999385166 | No | `[-0.013738477602601051, -4.294416427612305]` |
| 1 | `[[0, 0], [1, 0], [1, 0], [1, 1]]` | Tavi: Yes; 1.0000000000 / Juno: No; 0.9999998820 | No | `[-0.002282991772517562, -6.083391189575195]` |
| 2 | `[[1, 0], [0, 0], [1, 0], [1, 1]]` | Tavi: Yes; 1.0000000000 / Juno: Yes; 0.0323432544 | Yes | `[-2.2158851623535156, -0.11547470837831497]` |
| 3 | `[[1, 0], [1, 0], [1, 0], [1, 1]]` | Tavi: Yes; 1.0000000000 / Juno: No; 0.9999982318 | No | `[-0.1557314395904541, -1.9364778995513916]` |
| 4 | `[[0, 0], [0, 0], [1, 0], [0, 1]]` | Tavi: Yes; 1.0000000000 / Juno: No; 0.9999991829 | Yes | `[-1.9143595695495605, -0.15950722992420197]` |
| 5 | `[[0, 0], [1, 0], [1, 0], [0, 1]]` | Tavi: Yes; 1.0000000000 / Juno: No; 0.9999999959 | No | `[-0.04603429138660431, -3.1012978553771973]` |
| 6 | `[[1, 0], [0, 0], [1, 0], [0, 1]]` | Tavi: Yes; 1.0000000000 / Juno: No; 0.9985076317 | Yes | `[-6.04090690612793, -0.002382180653512478]` |
| 7 | `[[1, 0], [1, 0], [1, 0], [0, 1]]` | Tavi: Yes; 1.0000000000 / Juno: No; 0.9999970061 | Yes | `[-4.535604476928711, -0.010778306052088737]` |

Direct questions: Does Tavi favor the Meadow proposal? / Does Juno favor the Meadow proposal?

Native-final: No; logps `[-3.6238969187252223e-05, -10.226373672485352]`; source journal line 32, physical ID `NATIVE_FINAL|SSC1-FINAL-0004-d1-02`.
Same-method no-op: No; logps `[-0.013738477602601051, -4.294416427612305]`; source journal line 202, physical ID `o0/EXISTING_CORRECTION|SSC1-FINAL-0004-d1-02`.

Independently computed flags: `{"A": false, "C": true, "D": true, "W": false, "strong": false, "native_correct": true, "noop_correct": true}`.

Inputs: [inputs/designs/SSC1-FINAL-0004.json](../inputs/designs/SSC1-FINAL-0004.json) and the response shard named in `examples.json`.

No repeat claim is made for this particular example.

## excluded_native_reference

gemma / `INV_PAIR_NLL_s0` / `SSC1-FINAL-0000` / `SSC1-FINAL-0000-d0-08`

Actors: Hana, Zara, Ari, Ada. Projects: Library, Harbor.

Commands (actor/project indices and assigned values): `[{"actor": 2, "project": 0, "value": 0}, {"actor": 3, "project": 0, "value": 0}, {"actor": 0, "project": 0, "value": 0}]`.

Complete common final table: `[[0, 1], [1, 0], [0, 0], [0, 0]]`.

```text
Do Hana and Ari take the same side on Library?
Answer with exactly one of No, Yes.
```

Candidate order: `['No', 'Yes']`.

| History | Complete starting table | Direct operand answers; P(correct) | Joint answer | Joint candidate logps |
|---|---|---|---|---|
| 0 | `[[0, 1], [1, 0], [0, 0], [0, 0]]` | Hana: No; 0.9999877940 / Ari: No; 0.9999925125 | Yes | `[-6.481772422790527, -0.0017814256716519594]` |
| 1 | `[[1, 1], [1, 0], [0, 0], [0, 0]]` | Hana: No; 0.9999786699 / Ari: No; 0.9999921439 | Yes | `[-6.00304651260376, -0.002796669490635395]` |
| 2 | `[[0, 1], [1, 0], [0, 0], [1, 0]]` | Hana: No; 0.9999816831 / Ari: No; 0.9999931377 | Yes | `[-6.3299560546875, -0.002038188511505723]` |
| 3 | `[[1, 1], [1, 0], [0, 0], [1, 0]]` | Hana: No; 0.9999790433 / Ari: No; 0.9999928445 | Yes | `[-5.750472545623779, -0.003565385239198804]` |
| 4 | `[[0, 1], [1, 0], [1, 0], [0, 0]]` | Hana: No; 0.9999678093 / Ari: No; 0.9999910611 | Yes | `[-5.602263927459717, -0.0040604774840176105]` |
| 5 | `[[1, 1], [1, 0], [1, 0], [0, 0]]` | Hana: No; 0.9999786519 / Ari: No; 0.9999922268 | Yes | `[-5.049747943878174, -0.006815520115196705]` |
| 6 | `[[0, 1], [1, 0], [1, 0], [1, 0]]` | Hana: No; 0.9999750819 / Ari: No; 0.9999906102 | Yes | `[-4.3559722900390625, -0.013501990586519241]` |
| 7 | `[[1, 1], [1, 0], [1, 0], [1, 0]]` | Hana: No; 0.9999763258 / Ari: No; 0.9999917963 | Yes | `[-4.456789493560791, -0.01218459103256464]` |

Direct questions: Does Hana favor the Library proposal? / Does Ari favor the Library proposal?

Native-final: No; logps `[-0.023454023525118828, -3.7733852863311768]`; source journal line 12, physical ID `NATIVE_FINAL|SSC1-FINAL-0000-d0-08`.
Same-method no-op: Yes; logps `[-6.481772422790527, -0.0017814256716519594]`; source journal line 46, physical ID `o0/INV_PAIR_NLL_s0|SSC1-FINAL-0000-d0-08`.

Independently computed flags: `{"A": true, "C": false, "D": false, "W": false, "strong": false, "native_correct": false, "noop_correct": true}`.

Inputs: [inputs/designs/SSC1-FINAL-0000.json](../inputs/designs/SSC1-FINAL-0000.json) and the response shard named in `examples.json`.

No repeat claim is made for this particular example.

## excluded_noop_reference

qwen / `EXISTING_CORRECTION` / `SSC1-FINAL-0003` / `SSC1-FINAL-0003-d0-08`

Actors: Cato, Sora, Ada, Wren, Ivo, Mina. Projects: Gallery, Meadow, Canal.

Commands (actor/project indices and assigned values): `[{"actor": 3, "project": 0, "value": 0}, {"actor": 4, "project": 0, "value": 0}, {"actor": 5, "project": 0, "value": 0}]`.

Complete common final table: `[[0, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]`.

```text
Do Mina and Wren take the same side on Gallery?
Answer with exactly one of No, Yes.
```

Candidate order: `['No', 'Yes']`.

| History | Complete starting table | Direct operand answers; P(correct) | Joint answer | Joint candidate logps |
|---|---|---|---|---|
| 0 | `[[0, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]` | Mina: No; 1.0000000000 / Wren: No; 0.9999999999 | No | `[-0.03699471801519394, -3.3154218196868896]` |
| 1 | `[[0, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1]]` | Mina: No; 0.9999999872 / Wren: No; 0.9999999976 | No | `[-0.004812681116163731, -5.338907718658447]` |
| 2 | `[[0, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1]]` | Mina: No; 1.0000000000 / Wren: No; 0.9999999996 | No | `[-0.0036227568052709103, -5.622316837310791]` |
| 3 | `[[0, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]]` | Mina: No; 0.9999999976 / Wren: No; 0.9999999966 | No | `[-0.005364783573895693, -5.230576992034912]` |
| 4 | `[[0, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1]]` | Mina: No; 1.0000000000 / Wren: No; 0.9999999553 | No | `[-1.1920922133867862e-06, -13.600804328918457]` |
| 5 | `[[0, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1]]` | Mina: No; 0.9999999998 / Wren: No; 0.9999999932 | No | `[-1.1920922133867862e-06, -13.605820655822754]` |
| 6 | `[[0, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]]` | Mina: No; 1.0000000000 / Wren: No; 0.9999999951 | No | `[-3.576278118089249e-07, -14.734565734863281]` |
| 7 | `[[0, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]]` | Mina: No; 1.0000000000 / Wren: No; 0.9999999987 | No | `[-4.768370445162873e-07, -14.6282958984375]` |

Direct questions: Does Mina favor the Gallery proposal? / Does Wren favor the Gallery proposal?

Native-final: Yes; logps `[-1.7311108112335205, -0.19490548968315125]`; source journal line 12, physical ID `NATIVE_FINAL|SSC1-FINAL-0003-d0-08`.
Same-method no-op: No; logps `[-0.03699471801519394, -3.3154218196868896]`; source journal line 182, physical ID `o0/EXISTING_CORRECTION|SSC1-FINAL-0003-d0-08`.

Independently computed flags: `{"A": true, "C": false, "D": false, "W": false, "strong": false, "native_correct": true, "noop_correct": false}`.

Inputs: [inputs/designs/SSC1-FINAL-0003.json](../inputs/designs/SSC1-FINAL-0003.json) and the response shard named in `examples.json`.

No repeat claim is made for this particular example.
