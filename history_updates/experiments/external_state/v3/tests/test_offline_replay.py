"""Exercise the actual child guard, including harmless package discovery."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import datetime
import time

from repro.__main__ import environment


class OfflineReplayTests(unittest.TestCase):
    def test_absolute_deadline_stops_a_headless_python_without_the_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=environment(Path(tmp))
            env['REPRO_DEADLINE_UTC']=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=2)).isoformat()
            started=time.monotonic()
            result=subprocess.run([getattr(sys,'_base_executable',sys.executable),'-B','-c','import time;time.sleep(60)'],env=env,
                capture_output=True,text=True,timeout=6,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.assertEqual(result.returncode,124,result.stderr)
            self.assertLess(time.monotonic()-started,6)

    def test_optional_discovery_services_network_and_hidden_descendants(self):
        program=r'''
import importlib.util,os,socket,subprocess,sys
assert importlib.util.find_spec('openai') is not None
try:
    import openai
except ImportError as error:
    assert 'SERVICE_MODULE_DISABLED_DURING_REPLAY' in str(error)
else:
    raise AssertionError('Service import was allowed')
try:
    socket.getaddrinfo('example.invalid',443)
except RuntimeError as error:
    assert 'NETWORK_DISABLED_DURING_REPLAY' in str(error)
else:
    raise AssertionError('Network lookup was allowed')
if os.name=='nt':
    import ctypes
    assert ctypes.windll.kernel32.GetConsoleWindow()==0
    child=subprocess.run([sys.executable,'-c','import ctypes;assert ctypes.windll.kernel32.GetConsoleWindow()==0'],check=True)
    try:
        subprocess.Popen([sys.executable,'-c','pass'],creationflags=subprocess.CREATE_NEW_CONSOLE)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Visible child launch was allowed')
print('PASS')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-B','-c',program],env=environment(Path(tmp)),
                capture_output=True,text=True,timeout=30,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS',result.stdout)

    def test_child_environment_override_keeps_offline_guard(self):
        child="import os,socket;assert os.environ['REPRO_OFFLINE']=='1';socket.getaddrinfo('example.invalid',443)"
        parent=("import os,subprocess,sys;env=dict(os.environ,PYTHONPATH='',REPRO_OFFLINE='0');"
                "p=subprocess.run([sys.executable,'-c',"+repr(child)+"],env=env,capture_output=True,text=True);"
                "assert p.returncode!=0 and 'NETWORK_DISABLED_DURING_REPLAY' in p.stderr,p.stderr")
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-B','-c',parent],env=environment(Path(tmp)),
                capture_output=True,text=True,timeout=30,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
