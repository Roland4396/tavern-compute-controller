"""Validate one boot inventory for scheduler addresses and transport processes.
Does not open a connection, select a GPU, or change any SSH configuration.
"""
import argparse
import json
import re
from pathlib import Path

REMOTE_CALLBACK = '/home/worker/.local/state/gpu-runtime/orchestrator.sock'
ORDER = ('agent', 'glm', 'h3', 'h3_ref', 'graph', 'tts', 'seedvr3', 'qwen38_native', 'qwen38_yarn')
NAME = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,126}\Z')


def rows(config):
    nodes = config.get('nodes')
    if not isinstance(nodes, list) or not nodes:
        raise ValueError('GPU controller inventory is empty')
    addresses = {8080}  # controller API itself
    names = set()
    result = []
    for n in nodes:
        name = n['name']
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in names:
            raise ValueError('Invalid or duplicate node alias')
        names.add(name)
        ports = []
        for key in ORDER:
            port = n[key+'_port']
            if type(port) is not int or not 1024 <= port <= 65535 or port in addresses:
                raise ValueError('Invalid or overlapping tunnel port: '+key)
            ports.append(str(port)); addresses.add(port)
        # These alternative model servers intentionally use one local forward.
        if n['deepseek_port'] != n['glm_port']:
            raise ValueError('DeepSeek and GLM require the shared model forward')
        result.append([name,*ports[:7],REMOTE_CALLBACK,*ports[7:]])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('--snapshot',type=Path)
    a=p.parse_args()
    raw=a.config.read_bytes(); result=rows(json.loads(raw))
    if a.snapshot:
        # Caller supplied a private mktemp file. Both app.py and the forwards
        # will see exactly these validated bytes even if a mount later changes.
        a.snapshot.write_bytes(raw)
    for row in result:print('\t'.join(row))

if __name__=='__main__': main()
