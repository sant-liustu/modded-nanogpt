"""Add260 pairs (seeds2140-2399) to the previous140 mixed-context pairs."""
import sys
from run_mix2 import main

if __name__ == '__main__':
    if any(arg.split('=', 1)[0] in ('--pairs', '--seed-start') for arg in sys.argv[1:]):
        raise SystemExit('Top-up is fixed at260 pairs/seeds2140-2399; use run_mix2.py for custom ranges.')
    main(default_pairs=260, default_seed_start=2140)
