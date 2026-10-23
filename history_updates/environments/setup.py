"""Explicit pinned-environment installation, separate from offline replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[1]

class QuietEnvironment(venv.EnvBuilder):
    def _call_new_python(self,context,*args,**kwargs):
        kwargs['creationflags']=getattr(subprocess,'CREATE_NO_WINDOW',0)
        kwargs['timeout']=600
        return super()._call_new_python(context,*args,**kwargs)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',type=Path,default=ROOT.parent)
    parser.add_argument('--plan',action='store_true',help='Print paths and installation commands without creating environments')
    args=parser.parse_args();prefix=args.prefix.resolve()
    if prefix.is_relative_to(ROOT):parser.error('Environment prefix must be outside the immutable release')
    if sys.version_info[:2]!=(3,13):parser.error('Use Python 3.13 for these tested dependency pins')
    jobs=[]
    for name in ('core','historical','gmello'):
        folder=prefix/('.venv-'+name)
        python=folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        command=[str(python),'-m','pip','install','--disable-pip-version-check','--require-virtualenv','-r',str(ROOT/'environments'/(name+'.lock'))]
        jobs.append((folder,command))
    if args.plan:
        print(json.dumps([{'directory':str(folder),'install':command} for folder,command in jobs],indent=2));return
    if any(folder.exists() for folder,_ in jobs):parser.error('An environment already exists; choose a fresh prefix or use your existing interpreters')
    for folder,command in jobs:
        QuietEnvironment(with_pip=True).create(folder)
        subprocess.run(command,check=True,timeout=3600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print('Activate .venv-core or run its Python. Default sibling environments are discovered by reproduce.py.')

if __name__=='__main__':main()
