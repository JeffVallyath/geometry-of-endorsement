"""Safe single-case wrapper. Model inference must be explicitly requested."""
import argparse
import json
from pathlib import Path
import os

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',default='cases/fixed/example_input.json')
    p.add_argument('--method',default='REBUILD')
    p.add_argument('--output',default='example_output')
    p.add_argument('--question',default='cases/fixed/example_question.json')
    p.add_argument('--model-inference',action='store_true')
    p.add_argument('--cache',help='Local Hugging Face hub cache with exact pinned snapshot')
    p.add_argument('--seed',type=int,choices=(0,1),default=0)
    args=p.parse_args()
    from verify_package import verify
    verify()  # Hash and source-collision gate before any optional model load.
    if not args.model_inference:
        os.environ['CUDA_VISIBLE_DEVICES']=''
    from inputs import prepare, answer, commands_from_json
    source=json.loads(Path(args.case).read_text(encoding='utf8'))
    if set(source)!={'input_id','actor','source_text','commands'}:
        raise ValueError('Input file must contain only the supplied updater-side schema')
    dest=Path(args.output)
    if dest.exists():
        raise FileExistsError('Choose a new output directory; existing artifacts are immutable')
    bb=None
    if args.model_inference:
        if not args.cache:
            raise ValueError('--cache is required; automatic model downloads are disabled')
        from canonical.adapter import Backbone
        from canonical.config import MODELS
        bb=Backbone(cache=args.cache,cfg=MODELS[source['actor']])
    try:
        context=prepare(source['source_text'],commands_from_json(source['commands']),method=args.method,
                        backbone=bb,actor=source['actor'],seed=args.seed)
        dest.mkdir(parents=True)
        (dest/'updated_context.txt').write_text(context.text,encoding='utf8',newline='\n')
        result=context.describe()
        (dest/'updated_context.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        # Question file is not opened until the updated context is complete.
        q=json.loads(Path(args.question).read_text(encoding='utf8'))
        if set(q)!={'text','labels'}:
            raise ValueError('Question file may contain only text and public answer symbols')
        if bb is None:
            scored=dict(schema='MODEL_CHECK_PENDING_V1',question=q['text'],status='PENDING_NO_MODEL',
                        explanation='Text update exercised. No answer or model score was fabricated.')
        else:
            scored=answer(context,q['text'],q['labels'],backbone=bb)
        (dest/'answer.json').write_text(json.dumps(scored,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(context=result,answer_status=scored.get('status','SCORED'))))
    finally:
        if bb is not None:
            bb.finish()

if __name__=='__main__':
    main()
