"""Small JSON inheritance helper; lists replace, dictionaries merge."""
import copy
import json
from pathlib import Path


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError('Circular config inheritance: '+str(path))
    seen.add(path)
    config = json.loads(path.read_text())
    parent = config.pop('extends', None)
    if parent:
        config = merge(load_config(path.parent/parent, seen), config)
    seen.remove(path)
    return config
