"""Compare actual consumed sample streams from two completed training runs."""
import argparse
import json
from pathlib import Path


def load_run(path):
    metadata = json.loads((path / 'data_reuse_metadata.json').read_text(encoding='utf-8'))
    audit = {}
    for line in (path / 'data_order_audit.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        step = row['update_step']
        if step in audit:
            raise ValueError(f'duplicate audit step {step}: {path}')
        audit[step] = row
    if not audit:
        raise ValueError(f'empty audit: {path}')
    hp = metadata['hyperparameters']
    final = audit.get(hp['num_iterations'])
    if final is None or final['consumed_samples'] != hp['num_iterations'] * hp['batch_size']:
        raise ValueError(f'run is incomplete or sample counts are inconsistent: {path}')
    for step, row in audit.items():
        if row['consumed_samples'] != step * hp['batch_size'] or row['consumed_tokens'] != row['consumed_samples'] * hp['sequence_length']:
            raise ValueError(f'inconsistent counts at step {step}: {path}')
    return metadata, audit


def compare(a_path, b_path):
    a, aa = load_run(a_path)
    b, bb = load_run(b_path)
    for field in ['shard_content_fingerprint', 'sequence_length', 'samples_per_epoch',
                  'data_seed', 'shuffle', 'numpy_version', 'algorithm', 'epoch0_order_sha256']:
        if a['data_order'][field] != b['data_order'][field]:
            raise ValueError(f'data configuration differs: {field}')
    for field in ['batch_size', 'num_iterations']:
        if a['hyperparameters'][field] != b['hyperparameters'][field]:
            raise ValueError(f'{field} differs; this verifier requires equal training budgets')
    if aa.keys() != bb.keys():
        raise ValueError('audit checkpoints differ')
    for step in sorted(aa):
        if aa[step] != bb[step]:
            raise ValueError(f'actual consumed sample order differs by step {step}')
    final = aa[max(aa)]
    return f"PASS: actual sample order matches at {len(aa)} audit checkpoints; {final['consumed_samples']:,} samples / {final['consumed_tokens']:,} tokens."


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_a', type=Path)
    parser.add_argument('run_b', type=Path)
    args = parser.parse_args()
    try:
        print(compare(args.run_a, args.run_b))
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(1, f'FAIL: {exc}\n')
