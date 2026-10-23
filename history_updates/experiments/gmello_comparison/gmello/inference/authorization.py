"""Hash-bound amended main authorization; no numerical performance veto."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
PARENT_SHA='e45cd10e44d982e10187542565c1717770c8544b5949608120e3c12c337d82ed'
AMENDMENT_SHA='24244494e139d64fb08993de11f80d69e37171117aa95a63bca646cca7ddefd9'
LOCK_SHA='95f203f7ace167a8f1686843f9aff1f5979928e5f1ad9a35264ca884fcfa261f'
def read(path):return json.loads(Path(path).read_bytes())
def sha_file(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate_fields(authorization,lock,freeze=None,manifest=None):
    assert authorization['authorization_sha256']==digest({k:v for k,v in authorization.items() if k!='authorization_sha256'}),'AUTHORIZATION_DIGEST'
    assert authorization['schema']=='GMELLO_R0_MAIN_V3_AUTHORIZATION'
    assert authorization['parent_archive_sha256']==lock['parent_archive']['sha256']==PARENT_SHA
    assert authorization['amendment_sha256']==AMENDMENT_SHA and authorization['input_lock_sha256']==LOCK_SHA
    assert authorization['parent_graph_gate_status']=='UNQUALIFIED'
    assert authorization['parent_graph_gate_summary']['status']=='UNQUALIFIED' and authorization['parent_graph_gate_summary']['revision']==0
    assert authorization['selected_revision']==lock['selected_revision']==0
    assert authorization['explicit_amended_main_evaluation'] is True and authorization['no_retuning'] is True
    assert authorization['selected_after_development_inspection'] is True and authorization['new_main_outcomes_used_for_selection'] is False
    assert authorization['new_full_dev_rounds']==0
    if freeze is not None:
        assert freeze['public_projection_sha256']==digest({k:v for k,v in freeze.items() if k!='public_projection_sha256'}),'PUBLIC_PROJECTION_DIGEST'
        assert freeze['schema']=='GMELLO_R0_MAIN_V3_SCIENTIFIC_FREEZE'
        assert freeze['main_authorization_sha256']==authorization['authorization_sha256']
        assert freeze['main_authorization']==authorization and freeze['selected_revision']==0
        assert freeze['graph_gate']==authorization['parent_graph_gate_summary']
        assert freeze['parent_graph_gate_status']=='UNQUALIFIED' and freeze['amendment_sha256']==AMENDMENT_SHA
        assert freeze['parent_archive_sha256']==PARENT_SHA
        if manifest is not None:assert freeze['inference_manifest']==manifest,'INFERENCE_MANIFEST_IDENTITY'
    return authorization

def validate_authorization(freeze=None,manifest=None):
    # Original amendment/lock identities remain source identifiers. The public
    # science-only projection is covered by the union manifest; no runtime authority is granted.
    lock=read(ROOT/'INPUT_LOCK.json');authorization=read(ROOT/'MAIN_AUTHORIZATION.json')
    validate_fields(authorization,lock,freeze,manifest)
    for name,expected in lock['parent_inference_file_hashes'].items():
        if name=='inference/run.py':continue
        path=(ROOT.parent/name).resolve();assert path.is_relative_to(ROOT.resolve())
        assert sha_file(path)==expected,('INHERITED_SCIENCE_CHANGED',name)
    assert read(ROOT/'CONFIG.json')==lock['configuration_from_completed_parent']
    return authorization
