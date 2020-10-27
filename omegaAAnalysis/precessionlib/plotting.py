# functions for making plots from the output files
# produced by analysis.py and systematics.py
#
# Aaron Fienberg
# June 2019

import subprocess
import ROOT as r
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


#
# plotting code for systematic scan results
#


def plot_sweep(r_file, dir_name, xlabel, prefix, fit=True):
    ''' plot a systematic sweep,
    R and chi2 versus sweep parameter for T-Method and A-Weighted'''

    plt.rcParams['figure.figsize'] = 12, 9

    plt.subplots_adjust(wspace=0.25)
    plt.subplots_adjust(hspace=0.25)

    font = {'weight': 'bold', 'size': 16}

    for offset, subdir_name in enumerate(['T-Method', 'A-Weighted']):
        plt.subplot(221 + 2 * offset)

        params, rs = graph_to_arrays(
            r_file.Get(f'{dir_name}/{subdir_name}/sweepGraphs/R'))
        params, chi2s = graph_to_arrays(
            r_file.Get(f'{dir_name}/{subdir_name}/sweepGraphs/chi2'))

        plt.plot(params, rs, 'o')

        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel('$R$', fontsize=16)
        plt.gca().tick_params(labelsize=12)

        if fit:
            r_fit = np.polyfit(params, rs, 1)
            r_poly = np.poly1d(r_fit)
            fine_params = np.linspace(params[0], params[-1], 1000)

            plt.plot(fine_params, r_poly(fine_params), 'k--')

            raw_xlab = xlabel.replace('$', '')
            annot = f'$\\frac{{dR}}{{d({raw_xlab})}} = {r_fit[0]:.3f}$'

            x_annot = 0.05 if r_fit[0] > 0 else 0.55

            plt.text(x_annot, 0.75, annot,
                     transform=plt.gca().transAxes, fontdict=font)

        plt.subplot(222 + 2 * offset)

        plt.plot(params, chi2s, 'o')
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(r'$\chi^2$', fontsize=16)

        plt.gca().tick_params(labelsize=12)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        if fit:
            chi2_fit = np.polyfit(params, chi2s, 2)
            chi2_poly = np.poly1d(chi2_fit)

            chi2_width = 1 / np.sqrt(chi2_fit[0])
            chi2_min = -chi2_fit[1] / 2 / chi2_fit[0]

            plt.plot(fine_params, chi2_poly(fine_params), 'k--')
            annot = f'$\mathrm{{min}} = {chi2_min:.2f} \pm {chi2_width:.2f}$'
            plt.text(0.5, 0.75, annot,
                     transform=plt.gca().transAxes, fontdict=font,
                     horizontalalignment='center')

        figtext_y = 0.9 if offset == 0 else 0.48

        plt.figtext(0.5, figtext_y, f'{prefix} {subdir_name}',
                    ha='center', va='center', fontsize=16)

#
# Plot results of the seed scan
#


def plot_seed_scan(r_file, prefix):
    plt.rcParams['figure.figsize'] = 12, 5

    plt.subplots_adjust(wspace=0.25)
    plt.subplots_adjust(top=0.8)

    title = ''

    subdir_names = ['T-Method', 'A-Weighted']

    plt.subplot(121)
    for offset, subdir_name in enumerate(subdir_names):
        params, rs = graph_to_arrays(
            r_file.Get(f'seedScan/{subdir_name}/sweepGraphs/R'))

        mean_r = np.average(rs)
        r_std = np.std(rs)

        fmt = 'ko' if subdir_name == 'T-Method' else 'bo'
        color = 'black' if subdir_name == 'T-Method' else 'blue'

        plt.plot(params, rs, fmt, label=subdir_name)
        plt.axhline(mean_r, color=color, linestyle='-')
        plt.axhline(mean_r + r_std, color=color, linestyle='--')
        plt.axhline(mean_r - r_std, color=color, linestyle='--')

        print(f'{subdir_name}: mean = {mean_r:.3f}, std = {r_std:.3f}')

        title += f'\n{subdir_name}: {mean_r:.3f} +/- {r_std:.3f}'

        plt.legend(fontsize=16)
        plt.xlabel('seed number', fontsize=16)
        plt.ylabel('$R$', fontsize=16)

    plt.subplot(122)
    for offset, subdir_name in enumerate(subdir_names):
        params, chi2s = graph_to_arrays(
            r_file.Get(f'seedScan/{subdir_name}/sweepGraphs/chi2'))

        fmt = 'ko' if subdir_name == 'T-Method' else 'bo'

        plt.plot(params, chi2s, fmt, label=subdir_name)
        plt.xlabel('seed number', fontsize=16)
        plt.ylabel(r'$\chi^2$', fontsize=16)

    ndf = int(r_file.Get('seedScan/T-Method/TFit_seed_1').GetNDF())

    plt.axhline(ndf, linestyle='-', color='black')

    x_range = [0, params[-1] + 1]
    plt.xlim(*x_range)
    plt.fill_between(x_range, 2 * [ndf + np.sqrt(2.0 * ndf)],
                     2 * [ndf - np.sqrt(2.0 * ndf)], color='black',
                     alpha=0.5)

    # plt.axhline(ndf + np.sqrt(2.0 * ndf), linestyle='--', color='black')
    # plt.axhline(ndf - np.sqrt(2.0 * ndf), linestyle='--', color='black')

    plt.suptitle(f'{prefix}{title}', fontsize=16)
    plt.legend(fontsize=16)
    plt.gca().tick_params(labelsize=12)


def graph_to_arrays(graph):
    ''' extract x and y arrays from a ROOT TGraph'''

    xs = np.array([float(x) for x in graph.GetX()])
    ys = np.array([float(y) for y in graph.GetY()])

    return xs, ys


#
# Driver function
#

def make_systematics_plots(file_name, outdir, prefix):
    r_file = r.TFile(file_name)

    subprocess.call(f'mkdir -p {outdir}'.split())

    dir_names = ['pileupPhaseSweep', 'ifgAmpSweep',
                 'residualGainSweeps/ampSweep', 'residualGainSweeps/tauSweep',
                 'residualGainSweeps/asymmetrySweep',
                 'residualGainSweeps/phaseSweep']

    xlabels = [r'$\Delta t_{pu}$ [$\mu s$]', r'$A_{IFG}$', '$\delta_{g}$',
               r'$\tau_g$ [$\mu s$]', r'$A_{g}$', '$\phi_{g}$']

    if (r_file.Get('seedScan')):
        plot_seed_scan(r_file, prefix)
        plt.savefig(f'{outdir}/seedScan.pdf', bbox='tight')
        plt.savefig(f'{outdir}/seedScan.png', bbox='tight')
        plt.show()
    else:
        print(f'Seed scan not present in {file_name}')

    for dir_name, xlabel in zip(dir_names, xlabels):
        if not r_file.Get(dir_name):
            print(f'{dir_name} not present in {file_name}')
            continue

        do_fit = dir_name not in [
            'pileupPhaseSweep', 'residualGainSweeps/tauSweep',
            'residualGainSweeps/phaseSweep']

        plot_sweep(r_file, dir_name, xlabel, prefix, fit=do_fit)

        plot_name = dir_name.replace('/', '_')

        plt.savefig(f'{outdir}/{plot_name}.pdf', bbox='tight')
        plt.savefig(f'{outdir}/{plot_name}.png', bbox='tight')
        plt.show()
