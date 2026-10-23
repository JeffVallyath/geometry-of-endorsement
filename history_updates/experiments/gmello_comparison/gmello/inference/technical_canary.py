"""Six predetermined parent DEV requests for transport/timing, never a DEV sweep."""
from pathlib import Path
from isolation_guard import install
ACCESS_REPORT=install(Path(__file__).resolve().parent)
from base_runtime import ROOT,BaseEngine,Journal,read,publish,verify_capsule
from authorization import validate_authorization

def main():
    verify_capsule();validate_authorization()
    target=ROOT/'out/TECHNICAL_CANARY.json'
    if target.exists():
        assert read(target)['technical_status']=='COMPLETE';return
    requests=read(ROOT/'data/TECHNICAL_CANARY_REQUESTS.json');assert len(requests)==6
    journal=Journal();engine=BaseEngine(journal)
    try:
        for row in requests:assert engine.tokens(row['request'])==row['input_ids'],'PARENT_TOKENIZATION_CHANGED'
        outputs=engine.calls([row['request'] for row in requests],nonce='GMELLO_R0_MAIN_V3_FIXED_TECHNICAL_CANARY')
        report={'technical_status':'COMPLETE','requests':6,'full_dev_sweep':False,'selected_revision':0,
                'quality_threshold_used':False,'parent_request_and_token_identity_verified':True,
                'measurements':[{'kind':r['kind'],'length':r['length'],'parent_call_id':r['parent_call_id'],'generation':g} for r,g in zip(requests,outputs)]}
        publish(target,report)
    finally:
        engine.close();publish(ROOT/'out/technical/ACCESS_AUDIT.json',ACCESS_REPORT())

if __name__=='__main__':main()
