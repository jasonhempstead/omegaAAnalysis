# driver file to run the omega_a analysis
# configured using config files
#
# Aaron Fienberg
# September 2018

from precessionlib import analysis
import json
import sys


def main():
    # with open('confArtificialGainCor.json') as file:
    with open(sys.argv[1]) as file:
        config = json.load(file)

    analysis.run_analysis(config)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('must provide config!')
        sys.exit(0)
    sys.exit(main())
