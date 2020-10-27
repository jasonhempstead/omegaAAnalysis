# testing plotting.py

from precessionlib.plotting import make_systematics_plots
import sys


def main():
    filename = sys.argv[1]
    outdir = sys.argv[2]
    prefix = ' '.join(sys.argv[3:])
    make_systematics_plots(filename, outdir, prefix)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('must provide filename and outdir!')
        sys.exit(0)
    sys.exit(main())
