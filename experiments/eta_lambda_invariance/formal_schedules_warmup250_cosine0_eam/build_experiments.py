"""Materialize eight E/A/M schedules; the trainer reads explicit per-step numbers."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ('C', 'E', 'A', 'M', 'EA', 'EM', 'AM', 'EAM')
PATTERNS = {'E': 'transformer.wte.weight', 'A': 'transformer.h.*.attn.c_*.weight',
            'M': 'transformer.h.*.mlp.c_*.weight'}

def make_schedule(case, total=5100, warmup=250):
    assert case in CASES and 0 < warmup < total-1
    rows = []
    for step in range(1, total+1):
        base = .03 * min(step/warmup, 1.0)
        overrides = {}
        if step > warmup:
            q = (step-warmup-1)/(total-warmup-1)
            cosine = .03 * (1.0+math.cos(math.pi*q))/2.0
            overrides = {PATTERNS[g]: cosine for g in case if g in PATTERNS}
        rows.append(dict(update_step=step, default_elr=base, overrides=overrides))
    return dict(version=1, steps=rows)

def main():
    manifest = dict(start_mode='fresh', seed=0, warmup_completed_step=250, total_updates=5100, peak_rms_elr=.03,
                    cosine_final_rms_elr=0.0, norm_gamma='constant after warmup', cases=[])
    for case in CASES:
        path = HERE/f'{case}.json'
        text = json.dumps(make_schedule(case), indent=2, allow_nan=False)+'\n'
        path.write_text(text, encoding='utf-8', newline='\n')
        manifest['cases'].append(dict(case=case, schedule=path.name,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            cosine_groups=[g for g in case if g in PATTERNS]))
        wrapper = '#!/usr/bin/env bash\nset -euo pipefail\nHERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)\n'
        wrapper += f'exec bash "$HERE/run_case.sh" {case} "$@"\n'
        (HERE/f'run_{case}.sh').write_text(wrapper, encoding='utf-8', newline='\n')
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Created eight full schedules, eight launchers and manifest in', HERE)

if __name__ == '__main__': main()
