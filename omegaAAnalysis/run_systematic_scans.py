# testing systematics.py

from precessionlib import systematics
import sys


def main():
    systematics.run_systematic_sweeps(sys.argv[1])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('must provide config!')
        sys.exit(0)
    sys.exit(main())
