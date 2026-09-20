"""Verify source-bound completed follow-up reports; not a tensor/model replay."""
import json

from .common import ROOT
from .compact_delivery import verify

SOURCE = ROOT / 'reproducibility/completed_followups'
ARCHIVES = {
    'carrier_compact': '5a7fe38d4b86e523b1a164ca01b05a133606318d900be933ebb6ce4b66060ccb',
    'mediation': '13b1e4ec671ffd99855dc67ff5a27e9aac8a704ff169622e8494eab74612311c',
    'method_selection': 'cbecf068c0a5d3ca6731fe1eb6eb2baabcef64a611cd77f87bada69f34a80bb6',
    'valueprism_status': '674d279a75b26928bfdab3510b9c92f326866b71a8a1e40211f954409a9d4214',
}


def check():
    return dict(status='PASS', deliveries={label: len(verify(SOURCE, label, digest)['files'])
                                          for label, digest in ARCHIVES.items()},
                scope='Original report/data integrity; not a fresh tensor replay or runtime attestation',
                active_outcomes_read=False, inference=False)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
