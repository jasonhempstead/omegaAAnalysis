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
import numpy as np
from precessionlib.calospectra import CaloSpectra
from precessionlib.util import rebinned_last_axis

gain_amp = 1.1e-3
gain_tau = 64.44
gain_asym = 0.13


def build_root_hists(filename, histname,
                     model_file,
                     n_fills, rebin_factor):
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

    rebinned_corrected = rebinned_spec - \
        rebinned_last_axis(spec.pu_spectrum, rebin_factor)

    rebinned_times = rebinned_last_axis(
        spec.time_centers, rebin_factor) / rebin_factor

    # get phase/freq params from model file
    f = r.TFile(model_file)
    tHist = f.Get('T-Method/tMethodHist')
    tFit = tHist.GetFunction('tMethodFit')
    oma = tFit.GetParameter(5) * (1 + tFit.GetParameter(4) * 1e-6)
    phi = tFit.GetParameter(3)

    # wiggling gain
    eps = gain_amp * np.exp(-rebinned_times / gain_tau) * \
        (1 + gain_asym * np.cos(oma * rebinned_times - phi))
    for i in range(rebinned_corrected.shape[0]):
        calo_spec = rebinned_corrected[i]
        energies = spec.energy_centers

        rebinned_corrected[i] = CaloSpectra.gain_perturb(
            calo_spec, energies, -eps)

    corrected_hist = spec.build_root_hist(
        rebinned_corrected, rebinned_axes, 'corrected')

    corrected_hist.SetDirectory(0)

    return uncorrected_hist, corrected_hist


def main():
    if len(sys.argv) < 3:
        print('Usage: correct_pileup.py <input root file> <model fit file>')
        return 0

    infile_name = sys.argv[1]
    file = r.TFile(infile_name)

    model_file = sys.argv[2]

    dirs = ['clustersAndCoincidences',
            'clustersAndCoincidencesNoIFG',
            'clustersAndCoincidencesNoSTDP']

    hists = []

    found_dirs = []

    for dir_name in dirs:
        inf = r.TFile(infile_name)
        if not inf.Get(dir_name):
            print(f'directory "{dir_name}" not present')
            continue

        found_dirs.append(dir_name)

        ctag_hist = inf.Get(f'{dir_name}/ctag')

        n_fills = ctag_hist.GetEntries()

        print(f'{n_fills} fills')

        uncorrected, corrected = build_root_hists(
            infile_name,
            histname=f'{dir_name}/clusters',
            n_fills=n_fills, rebin_factor=6,
            model_file=model_file)

        trip_hist = file.Get(f'{dir_name}/triples')
        quad_hist = file.Get(f'{dir_name}/quadruples')
        ctag_hist = file.Get(f'{dir_name}/ctag')

        hists.append([uncorrected, corrected, trip_hist, quad_hist, ctag_hist])

    outfile_name = infile_name.replace(
        '.root', '') + '_pileup_corrected_fgain.root'

    outf = r.TFile(outfile_name, 'recreate')

    for dir_name, hist_list in zip(found_dirs, hists):
        out_dir = outf.mkdir(dir_name)
        out_dir.cd()

        for hist in hist_list:
            hist.SetDirectory(r.gDirectory)

    outf.Write()


if __name__ == '__main__':
    sys.exit(main())
