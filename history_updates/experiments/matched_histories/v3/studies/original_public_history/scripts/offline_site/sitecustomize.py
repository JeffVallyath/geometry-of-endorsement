"""Prevent accidental network or pretrained-client use in CPU review commands."""
import importlib.abc
import sys


class NoPretrainedClients(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'transformers', 'huggingface_hub', 'datasets', 'sentence_transformers',
            'openai', 'anthropic', 'boto3', 'lambda_api', 'operator_adapter',
        }:
            raise ImportError('Offline review forbids pretrained-model and service clients')


def no_network(event, args):
    if event.startswith('socket.') and event != 'socket.gethostname':
        raise RuntimeError('Offline review forbids network access')


sys.meta_path.insert(0, NoPretrainedClients())
sys.addaudithook(no_network)
