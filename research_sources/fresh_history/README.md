# Synthetic case generation

Start with `build_root` and `gate_mapping` in [prepare_data.py](prepare_data.py)
for case construction and the direct/combined checks. The [canonical helpers](canonical)
define scenes, histories, demonstrations and question forms.

These files can be read and extended here. Running the original generator also
requires its historical inventory, path bindings and model metadata; do not invoke
the entry point as though those dependencies were included.

<details>
<summary>Original source paths and hashes</summary>

Source capsule SHA-256: `8f401b0582990390c0d9b675e95aded75fea0806edb1994df8955968a3789477`.

| Source member | Delivered file | SHA-256 |
|---|---|---|
| `src/prepare_data.py` | [prepare_data.py](prepare_data.py) | `33dd415aa5a4d896bf6aa336f51c85fca0f68f705ac88fe3027be4c00dc94f69` |
| `handoff/plan_checks.py` | [plan_checks.py](plan_checks.py) | `aa93fbbbb6bd8a6ee8acf613f0af9b132f9c798c601cde1957effe05861e6cef` |
| `reproduction/STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS/canonical/candidate.py` | [canonical/candidate.py](canonical/candidate.py) | `f63ef54e7a9d1069f794aa5e197f9b57d56732353221c85f68007374663aa040` |
| `reproduction/STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS/canonical/origins.py` | [canonical/origins.py](canonical/origins.py) | `eba4044ec0872d43777838efb4b2a58ff09a1944130ce7f4e2a4774658746f1c` |
| `reproduction/STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS/canonical/readers.py` | [canonical/readers.py](canonical/readers.py) | `58181aaad07c5673cea764a423b7df9f8630b3bb78d98eec5d153715c456a7e6` |
| `reproduction/STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS/canonical/scenes.py` | [canonical/scenes.py](canonical/scenes.py) | `40c941d56f6b640ff7a5fc5c95d245b5fe0ec86e5764198aa46c1b4a827ec4bb` |
| `reproduction/STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS/canonical/tasks.py` | [canonical/tasks.py](canonical/tasks.py) | `6e3b73b99b4c8c220e6eea2d54cb0830c8bf2efd83ac20850e274b66d1f211bc` |

</details>
