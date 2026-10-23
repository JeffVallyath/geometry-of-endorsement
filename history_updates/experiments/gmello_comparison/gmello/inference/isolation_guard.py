"""Process boundary: no sibling repair/evaluator files, adapters, or network."""
from pathlib import Path
import os
import sys

def install(root):
    root = Path(root).resolve(); task_root = root.parent
    opened = set(); rejected = []
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            rejected.append({"kind": "network"})
            raise PermissionError("COMPARATOR_NETWORK_FORBIDDEN")
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name == "adapter_model.safetensors" or path.name == "adapter_config.json":
            rejected.append({"kind": "adapter"})
            raise PermissionError("COMPARATOR_ADAPTER_FORBIDDEN")
        try: relative = path.relative_to(task_root)
        except ValueError: return
        if not path.is_relative_to(root):
            rejected.append({"kind": "sibling_task_file", "relative": str(relative)})
            raise PermissionError("COMPARATOR_SIBLING_INPUT_FORBIDDEN")
        opened.add(str(path.relative_to(root)))
    sys.addaudithook(audit)
    def report():
        return {"network_forbidden": True, "sibling_repair_and_evaluator_files_forbidden": True, "adapter_files_forbidden": True,
                "task_files_opened": sorted(opened), "rejected_attempts": rejected}
    return report
