"""Width 1536 (=768*2) entry point sharing the fixed-norm ELR training code."""
from pathlib import Path
import runpy
import sys


def main():
    here = Path(__file__).resolve().parent
    options = sys.argv[1:]
    for option in options:
        if option.startswith('--model_width=') and option != '--model_width=1536':
            raise ValueError('this entry is fixed to model_width=1536; use train.py for other widths')
    if not any(option.startswith('--model_width=') for option in options):
        options = ['--model_width=1536'] + options
    if not any(option.startswith('--run_name=') for option in options):
        options = ['--run_name=w1536'] + options
    sys.argv = [str(here / 'train.py')] + options
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
