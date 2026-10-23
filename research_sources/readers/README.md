# Reader training and serving

| Work on | File and entry point |
|---|---|
| QA/PROGRAM training and serving | [model_runtime.py](model_runtime.py): `train`, `Engine` |
| Ranking-interpreter training and scoring | [qsa_model.py](qsa_model.py): `training`, `Scorer`, `score_arm` |
| Evaluate a case | [worker.py](worker.py): `evaluate_case` |
| Parse a generated operation | [baseline_core.py](baseline_core.py): `parse_program` |

These implementations require model files, adapters and their relative runtime
dependencies before execution. Original container paths must also be configured.

The GMeLLo code in `baseline_core.py` is historical. Use the
[corrected comparison](../../history_updates/results/tables/gmello_corrected.md)
for the current 91/89/46 results.

## Adapters

The six 7B adapters are stored separately in `SAME_CASE_BASELINES_V1_RESULTS.zip`:
QA and PROGRAM seeds 0/1, plus the original ranking-interpreter seeds 0/1.
Obtain these before setting up a fresh run. Base-model weights are separate;
the final 32B/27B repair adapters still need to be supplied.

<details>
<summary>Exact archive members and hashes</summary>

Archive SHA-256: `fd5ba7f79443377b18558fdf0f942a380c6c0e7039c6a5e570f596ad5dacc3ad`.

| Original member | Source file | SHA-256 |
|---|---|---|
| `runtime_source/baseline_core.py` | [baseline_core.py](baseline_core.py) | `e515508cdf5c8b2ffb7a5ec556520051e503a69b68e8040b1609cb7128d49ef2` |
| `runtime_source/model_runtime.py` | [model_runtime.py](model_runtime.py) | `4e79b4bb96f75eaccb71041ddf5f1b7ef437daf487770b59da4f9474df40efa9` |
| `runtime_source/worker.py` | [worker.py](worker.py) | `f645e66f6f3165ab1e511bd853db797cbf70b826297b78a222298d377e6626b0` |
| `historical_training/payload/qsa_model.py` | [qsa_model.py](qsa_model.py) | `6ebf0a78d0ff42eef19622ded4c6fe4cd8de4cdb0b42244f97cb987d621c5f25` |

Each adapter below is 40,400,200 bytes. Canary adapters are not included.

| Adapter member | SHA-256 |
|---|---|
| `raw/adapters/PROGRAM_0/adapter_model.safetensors` | `58e0fb0e19c26b02dd9a7e1ecb58f3e974189a7a6754032eaeb101a516d5caf4` |
| `raw/adapters/PROGRAM_1/adapter_model.safetensors` | `34681a0f45f08b7b6e869312774b78bdcaf7083235ff3b62659c1a5fb5a49f58` |
| `raw/adapters/QA_0/adapter_model.safetensors` | `7d8068fca1abe1f4974e99d30bc526aa7955c68cb9206e273b44c87f0a09bf8a` |
| `raw/adapters/QA_1/adapter_model.safetensors` | `3f5a94e02e12580e5551f35308ab7d07c4a117d021b652d7f6f4f1fb8d9273ef` |
| `original_adapters/seed0/adapter_model.safetensors` | `abdfc30d4ad24cedad04d99a06797b854115bccdc0853dd240e133f8dd3119c3` |
| `original_adapters/seed1/adapter_model.safetensors` | `b9b84813e04f39d47e55a6bb8be25f34b8921970e11cd3e707dbd357b8cc6ce1` |

</details>
