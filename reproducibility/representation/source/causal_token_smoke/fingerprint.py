from __future__ import annotations

import hashlib
from typing import Any


def parameter_sha256(model: Any, chunk_elements: int = 8 * 1024 * 1024) -> str:
    """Stream a deterministic full-parameter fingerprint without one giant CPU copy."""
    import torch

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = parameter.detach().contiguous().view(-1)
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(str(parameter.dtype).encode("ascii") + b"\0")
            digest.update(str(tuple(parameter.shape)).encode("ascii") + b"\0")
            for start in range(0, value.numel(), chunk_elements):
                byte_view = value[start : start + chunk_elements].view(torch.uint8)
                digest.update(byte_view.cpu().numpy().tobytes())
    return digest.hexdigest()
