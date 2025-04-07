import json
from typing import List, Dict
from pathlib import Path

def read_jsonl(path: Path) -> List[Dict]:
    """Read a jsonl file line by line and parse each line as JSON."""
    if not path.exists():
        return []
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def append_jsonl(path: Path, data: Dict):
    """Append a single dictionary as a JSON line to a jsonl file."""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def update_jsonl(path: Path, updated_data: Dict, key: str):
    """Update a specific record in a jsonl file based on a key."""
    items = read_jsonl(path)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            if item[key] == updated_data[key]:
                f.write(json.dumps(updated_data, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps(item, ensure_ascii=False) + '\n') 