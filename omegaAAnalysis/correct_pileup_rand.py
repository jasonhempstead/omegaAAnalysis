# creates the pileup corrected histogram
# also copies the lost muon histogram
#
# For now has hardcoded histogram names and rebin factors
#
# Takes a couple minutes
#
# Aaron Fienberg
# September 2018

import sys
import ROOT as r
from precessionlib.calospectra import CaloSpectra
from precessionlib.util import rebinned_last_axis


def build_root_hists(filename, histname, n_fills,
                     nonrand_name, rebin_factor):
    '''
    use CaloSpectra to build pileup corrected and non corrected 3d hists
    returns uncorrected, corrected, pileup_normalizations
    '''
    par_guess = 2 * 1.25 / n_fills / 25
    spec = CaloSpectra.from_root_file(filename, histname, do_triple=True,
                                      single_param=True,
                                      param_guess=par_guess)

    rebinned_axes = list(spec.axes)
    rebinned_axes[-1] = rebinned_axes[-1][::rebin_factor]

    rebinned_spec = rebinned_last_axis(spec.array, rebin_factor)

    uncorrected_hist = spec.build_root_hist(
        rebinned_spec, rebinned_axes, 'uncorrected')
    uncorrected_hist.SetDirectory(0)

    # build pileup correction from nonrandomized hist
    nonrand_spec = CaloSpectra.from_root_file(filename, nonrand_name,
                                              do_triple=True,
                                              single_param=True,
                                              param_guess=par_guess)

    rebinned_corrected = rebinned_spec - \
        rebinned_last_axis(nonrand_spec.pu_spectrum, rebin_factor)

    corrected_hist = spec.build_root_hist(
        rebinned_corrected, rebinned_axes, 'corrected')

    corrected_hist.SetDirectory(0)

    return uncorrected_hist, corrected_hist


def main():
    if len(sys.argv) < 2:
        print('Usage: correct_pileup.py <input root file>')
        return 0

    infile_name = sys.argv[1]
    file = r.TFile(infile_name)

    dirs = ['clustersAndCoincidences']

    hists = []

    for dir_name in dirs:
        inf = r.TFile(infile_name)
        ctag_hist = inf.Get(f'{dir_name}/ctag')
        n_fills = ctag_hist.GetEntries()

        print(f'{n_fills} fills')

        uncorrected, corrected = build_root_hists(
            infile_name, nonrand_name=f'{dir_name}/clusters',
            n_fills=n_fills,
            histname='clustersAndCoincidencesRand/clusters',
            rebin_factor=6)

        trip_hist = file.Get(f'{dir_name}/triples')
        quad_hist = file.Get(f'{dir_name}/quadruples')
        ctag_hist = file.Get(f'{dir_name}/ctag')

        hists.append([uncorrected, corrected, trip_hist, quad_hist, ctag_hist])

    outfile_name = infile_name.replace(
        '.root', '') + '_pileup_corrected_rand.root'

    outf = r.TFile(outfile_name, 'recreate')

    for dir_name, hist_list in zip(dirs, hists):
        out_dir = outf.mkdir(dir_name)
        out_dir.cd()

        for hist in hist_list:
            hist.SetDirectory(r.gDirectory)

    outf.Write()


if __name__ == '__main__':
    sys.exit(main())
