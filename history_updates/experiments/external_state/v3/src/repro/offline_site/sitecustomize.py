"""Process-local guards for offline replay; no system settings are changed."""
import datetime
import importlib.abc
import importlib.machinery
import os
import subprocess
import sys
import threading
import time

if os.environ.get('REPRO_OFFLINE')=='1':
    def offline(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo',
                     'socket.gethostbyname','socket.gethostbyaddr'}:
            raise RuntimeError('NETWORK_DISABLED_DURING_REPLAY: '+event)
    sys.addaudithook(offline)

    class DisabledServiceLoader(importlib.abc.Loader):
        def create_module(self,spec):
            return None
        def exec_module(self,module):
            raise ImportError('SERVICE_MODULE_DISABLED_DURING_REPLAY: '+module.__name__)

    class NoControlServices(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in {'runpod','boto3','paramiko','openai','cloud','watchdog','deploy'}:
                # Optional dependency discovery is read-only. Deny execution
                # at the loader boundary without breaking find_spec callers.
                return importlib.machinery.ModuleSpec(fullname,DisabledServiceLoader())
            return None
    sys.meta_path.insert(0,NoControlServices())

    # Some unchanged historical runners omit flags at their child boundary.
    # This guard affects only descendants in this explicitly offline invocation.
    original_popen=subprocess.Popen
    inherited_deadline=os.environ.get('REPRO_DEADLINE_UTC')
    class GuardedPopen(original_popen):
        def __init__(self,*args,**kwargs):
            # Historical wrappers sometimes replace PYTHONPATH for their own
            # modules. Preserve the offline/deadline guard at that boundary.
            env=dict(os.environ if kwargs.get('env') is None else kwargs['env'])
            guard=os.path.dirname(__file__)
            paths=env.get('PYTHONPATH','').split(os.pathsep)
            env['PYTHONPATH']=os.pathsep.join([guard,*[p for p in paths if p and p!=guard]])
            for key in ('REPRO_OFFLINE','CUDA_VISIBLE_DEVICES',
                        'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE',
                        'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
                if key in os.environ:env[key]=os.environ[key]
            if inherited_deadline:
                requested=env.get('REPRO_DEADLINE_UTC',inherited_deadline)
                def instant(value):
                    parsed=datetime.datetime.fromisoformat(value)
                    if parsed.tzinfo is None:raise ValueError('Replay deadline requires a timezone')
                    return parsed.timestamp()
                # A nested command may narrow its deadline, never extend it.
                env['REPRO_DEADLINE_UTC']=min((inherited_deadline,requested),key=instant)
            kwargs['env']=env
            if os.name=='nt':
                flags=kwargs.get('creationflags',0)
                if flags & (subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_CONSOLE):
                    raise RuntimeError('Visible or detached child launch forbidden in replay')
                kwargs['creationflags']=flags|subprocess.CREATE_NO_WINDOW
            super().__init__(*args,**kwargs)
    subprocess.Popen=GuardedPopen

    if os.environ.get('REPRO_DEADLINE_UTC'):
        deadline=datetime.datetime.fromisoformat(os.environ['REPRO_DEADLINE_UTC']).timestamp()
        def deadline_guard():
            while True:
                remaining=deadline-time.time()
                if remaining<=0:os._exit(124)
                time.sleep(min(remaining,30))
        threading.Thread(target=deadline_guard,daemon=True).start()
