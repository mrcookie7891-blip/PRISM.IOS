"""Deduplicate normalized lines before stable, deterministic held-out splitting."""
import hashlib
from pathlib import Path
import random
import unicodedata


def split_documents(paths, seed=42):
    files = set()
    for path in paths:
        path = Path(path)
        files.update(path.glob('*.txt') if path.is_dir() else [path])
    documents, seen, sources = [], set(), []
    for path in sorted(files):
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        sources.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        for line in text.splitlines():
            line = unicodedata.normalize('NFC', line).strip()
            if line and line not in seen:
                seen.add(line)
                documents.append(line + '\n')
    if len(documents) < 10:
        raise ValueError('Need at least 10 unique nonempty lines for train/validation/test splits')
    # Shuffle sorted content so file ordering does not change assignments.
    documents.sort()
    random.Random(seed).shuffle(documents)
    n = max(1, len(documents) // 10)
    splits = {'train': documents[2*n:], 'validation': documents[n:2*n], 'test': documents[:n]}
    hashes = {k: hashlib.sha256(''.join(v).encode()).hexdigest() for k, v in splits.items()}
    return splits, {'seed': seed, 'sources': sources, 'split_sha256': hashes,
                    'unique_lines': len(documents), 'split_documents': {k: len(v) for k, v in splits.items()}}
