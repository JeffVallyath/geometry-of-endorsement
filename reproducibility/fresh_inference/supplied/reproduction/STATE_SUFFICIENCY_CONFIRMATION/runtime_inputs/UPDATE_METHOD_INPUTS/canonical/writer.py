import torch
from . import request as V

def writer_plan(bank, request):
    """No query, mapping, gold, donor, source backup or origin enters the writer."""
    if type(request) is not V.WriterInput:
        raise TypeError('Writer requires the exact narrow immutable schema')
    maps, union = [], set()
    for positions, value in zip(request.addresses, request.desired_values, strict=True):
        index = (torch.zeros(len(positions), dtype=torch.long, device=bank.bb.device),
                 torch.tensor(positions, dtype=torch.long, device=bank.bb.device))
        maps.append({'kind': 'setter', 'fn': bank.fn(value, 1., record=False),
                     'index': index, 'clause_positions': list(positions)})
        union.update(positions)
    return {'maps': maps, 'index': sorted(union), 'site': bank.site, 'workspace': 'fp32'}

