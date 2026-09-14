"""Original no-training backbone guard; provider lifecycle omitted."""
import importlib.abc
import rfr5_common as C

class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.',1)[-1] in {'rcc4_train','rso3_train','srs2_train','qbrc_train'}:
            raise ImportError('Training entry point prohibited in RFR5')
        return None

def create_backbone(cache, actor):
    import torch
    from rso3_adapter import Backbone
    class FrozenBackbone(Backbone):
        def backward(self, *args, **kwargs): raise RuntimeError('No backward passes in RFR5')
        def train_forward(self, *args, **kwargs): raise RuntimeError('No training path in RFR5')
        def full_forward(self, *args, **kwargs): raise RuntimeError('No training-capable full_forward path in RFR5')
        def compile(self, prefix_ids, maps=None, index=None, site=None, capture=(), grad=False, replay_positions=None, batch=1, rows=None, capture_next=(), workspace='fp32'):
            if grad: raise RuntimeError('No gradient compilation in RFR5')
            return super().compile(prefix_ids,maps,index,site,capture,False,replay_positions,batch,rows,capture_next,workspace)
    torch.set_grad_enabled(False)
    return FrozenBackbone(cache=cache,cfg=C.V4.MODELS[actor])
