# Take a config file and build the A-Weighted and T-Method
# pu enhancement factors based on the analysis results
# saves the results in a text file

import sys
import json
import numpy as np
from precessionlib.calospectra import CaloSpectra
from matplotlib import pyplot as plt
import ROOT as r


def get_E_thresh(root_f):
    T_hist = root_f.Get('T-Method/tMethodHist')
    name = T_hist.GetTitle()
    words = name.split()

    energy_index = words.index('GeV') - 1

    return 1000 * float(words[energy_index])


def get_A_model(root_f):
    return root_f.Get('A-Weighted/aVsESpline')


def main():
    with open(sys.argv[1]) as file:
        config = json.load(file)

    # get the output file (the one with the fitted histograms)
    out_dir = config['out_dir']
    out_name = config['outfile_name']
    if not out_name.endswith('.root'):
        out_name += '.root'
    full_f_name = f'{out_dir}/{out_name}'
    output_root_f = r.TFile(full_f_name)

    # get the input file (the one with the 3d hist)
    input_f_name = config['file_name']
    uncor_name = config['uncor_hist_name']
    input_f = r.TFile(input_f_name)

    # get the number of fills in the dataset
    dir_name = uncor_name.split('/')[0]
    ctag_hist = input_f.Get(f'{dir_name}/ctag')
    n_fills = ctag_hist.GetEntries()
    print(f'{n_fills:.0f} fills')

    # build CaloSpectra
    # assumes cycltron binning
    spec = CaloSpectra.from_root_file(input_f_name,
                                      histname=uncor_name, do_triple=True,
                                      single_param=True,
                                      param_guess=2 * 1.25 / n_fills / 150,
                                      pu_time_max=650)

    # get T-Method energy threshold
    E_thresh = get_E_thresh(output_root_f)
    # get A vs E model
    A_model = get_A_model(output_root_f)

    T_meth_facs = spec.T_method_unc_facs(E_thresh)
    # assumes 1.0 - 3.0 GeV range for a-weighted hist
    A_weight_facs = spec.A_weighted_unc_facs(A_model, 1000, 3000)

    avg_dt = np.average(spec.estimated_deadtimes(n_fills))
    print(f'Average deadtime: {avg_dt*1000:.2f} ns')

    # make a plot to ensure everything looks ok
    times = spec.time_centers
    plt.plot(times, T_meth_facs)
    plt.plot(times, A_weight_facs)
    plt.xlim(30.2, 100)
    plt.ylim(1, 1.01)

    plt.show()

    # write the results to a file
    out_name = out_name.replace('.root', '')
    txt_out_name = f'{out_dir}/{out_name}_pu_enhancements.txt'
    print(f'Saving result to {txt_out_name}')
    stacked = np.hstack(
        (times[:, None], T_meth_facs[:, None], A_weight_facs[:, None]))

    np.savetxt(txt_out_name, stacked,
               header='time, T-Meth, A-Weight')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('must provide config!')
        sys.exit(0)
    sys.exit(main())
