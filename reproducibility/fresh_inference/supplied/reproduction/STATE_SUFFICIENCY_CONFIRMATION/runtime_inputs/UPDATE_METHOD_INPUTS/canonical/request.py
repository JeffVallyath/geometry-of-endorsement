from dataclasses import dataclass

@dataclass(frozen=True)
class WriterInput:
    """The complete inference access boundary; no evaluator fields allowed."""
    prefix_ids: tuple[int, ...]
    addresses: tuple[tuple[int, ...], ...]
    desired_values: tuple[int, ...]

    def __post_init__(self):
        if not self.prefix_ids or len(self.addresses) != len(self.desired_values):
            raise ValueError('Invalid writer input')
        for address, value in zip(self.addresses, self.desired_values, strict=True):
            if not address or len(set(address)) != len(address) or value not in (0, 1):
                raise ValueError('Invalid address/value')
            if min(address) < 0 or max(address) >= len(self.prefix_ids):
                raise ValueError('Address outside prefix')

