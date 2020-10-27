# omega_a analysis functions
#
# also contains the run_analysis driver function
#
# Aaron Fienberg
# September 2018

import math
import subprocess
import ROOT as r
import numpy as np
from BlindersPy3 import Blinders, FitType
from .util import *
from .fitmodels import *

r.gStyle.SetOptStat(0)
r.gStyle.SetOptFit(1111)

calo_dirname = 'perCaloPlots'
e_sweep_dirname = 'energyBinnedPlots'


def do_threshold_sweep(all_calo_2d, fit_function, fit_start, fit_end,
                       start_thresh=1800, twoDHist = True):
    ''' finds the optimal T-Method threshold
        returns:
        (best_threshold, [(thresh1, r_precision1, thresh2, preceision2),...])

        thresholds are in units of bin number from all_calo_2d's y axis
    '''

    thresh_bin = all_calo_2d.GetYaxis().FindBin(start_thresh)

    r_precisions = []
    best_thresh = None
    for e_bin in range(thresh_bin - 50, thresh_bin + 30):
        if twoDHist:
            this_proj = all_calo_2d.ProjectionX(f'proj{e_bin}', e_bin, -1)
        else:
            this_proj = all_calo_2d

        if fit_function.GetParameter(0) == 0:
            fit_function.SetParameter(
                0, this_proj.GetBinContent(this_proj.FindBin(30)) * 1.6)

        this_proj.Fit(fit_function, '0qEM', '', fit_start, fit_end)

        r_precisions.append((e_bin,
                             fit_function.GetParError(4)))

        if best_thresh is None or best_thresh[-1] > r_precisions[-1][-1]:
            best_thresh = r_precisions[-1]

    return best_thresh[0], r_precisions


def find_cbo_freq(fft_hist):
    return fft_hist.GetBinCenter(fft_hist.GetMaximumBin())


def fit_and_fft(hist, func, fit_name, fit_options,
                fit_start, fit_end, find_cbo=False, double_fit=False):
    ''' fit hist to a function and FFT the residuals
        returns the residuals hist and the FFT hist

        if find_cbo is true,
        cbo_freq will be estimated from the largest FFT peak
        and returned after the FFT hist as a third return value
    '''

    hist.Fit(func, fit_options, '', fit_start, fit_end)
    if double_fit:
        hist.Fit(func, fit_options, '', fit_start, fit_end)

    if adjust_phase_parameters(func):
        # adjusted some phase parameters, try fitting again
        hist.Fit(func, fit_options, '', fit_start, fit_end)

    resids = build_residuals_hist(hist, func)
    resid_fft = fft_histogram(after_t(resids, fit_start), f'{fit_name}FFT')
    resid_fft.SetTitle(f'{hist.GetTitle()};f [MHz];fft mag')

    retvals = (resids, resid_fft)

    if find_cbo:
        cbo_freq = find_cbo_freq(resid_fft)
        retvals = retvals + (cbo_freq,)

    return retvals


def fit_slice(master_3d, name, model_fit,
              fit_options, fit_range, energy_bins, calo_bins,
              adjust_N=False):
    ''' do a projected T-Method fit
        starting fit_guesses are based on model_fit
        projects in region defined by energy_bins and calo_bins
        returns
        energy_bins are in bin numbers, not energies

        if adjust_N is true, sets the N0 guess based on
        bin content at 30 microseconds

        returns fit, resids, fft
    '''

    master_3d.GetXaxis().SetRange(0, master_3d.GetNbinsX())
    master_3d.GetYaxis().SetRange(*energy_bins)
    master_3d.GetZaxis().SetRange(*calo_bins)

    hist = master_3d.Project3D('x')
    hist.SetName(name)

    # hist = calo_2d.ProjectionX(f'{name}', energy_bins[0], energy_bins[1])

    bin_width = hist.GetBinWidth(1)
    hist.SetTitle(f';time [#mus]; N / {bin_width:.3f} #mus')

    if adjust_N:
        model_fit.SetParameter(0,
                               hist.GetBinContent(hist.FindBin(30)) * 1.6)

    # find a reasonable guess for the asymmetry
    # by comparing max, min, avg over one period
    if is_free_param(model_fit, 2):
        start_bin = hist.FindBin(fit_range[0])
        bins_per_period = int(approx_oma_period / hist.GetBinWidth(1))
        vals = [hist.GetBinContent(i)
                for i in range(start_bin, start_bin + bins_per_period)]
        a_guess = (max(vals) - min(vals)) * len(vals) / sum(vals) / 2
        model_fit.SetParameter(2, a_guess)

    resids, fft = fit_and_fft(
        hist, model_fit, name, fit_options, fit_range[0], fit_range[1],
        double_fit=True)

    master_3d.GetXaxis().SetRange(1, master_3d.GetNbinsX())
    master_3d.GetYaxis().SetRange(1, master_3d.GetNbinsY())
    master_3d.GetZaxis().SetRange(1, master_3d.GetNbinsZ())

    return hist, resids, fft


def T_method_analysis(all_calo_2d, blinder, config, pu_unc_factors=[]):
    ''' do a full calo T-Method analysis
    returns T-Method hist, fit function, TFitResult, threshold_bin
    '''

    # where to put the plots
    pdf_dir = f'{config["out_dir"]}/plots'

    # start with a simple all calo analysis

    # sweep energy threshold to find the best energy cut
    omega_a_ref = blinder.paramToFreq(0)
    five_param_tf1 = build_5_param_func(config)
    five_param_tf1.SetParameters(0, 64.4, 0.2, 0.2, 0, omega_a_ref)
    five_param_tf1.FixParameter(5, omega_a_ref)
    print('doing intial threshold sweep...')

    optimal_thresh_bin, sweep_res = do_threshold_sweep(
        all_calo_2d, five_param_tf1, config['fit_start'], config['fit_end'])
    c, _ = plot_threshold_sweep(all_calo_2d, sweep_res, optimal_thresh_bin)
    c.Print(f'{pdf_dir}/optimalThreshold.pdf')

    print(f'best threshold bin: {optimal_thresh_bin}')

    if config['fix_thresh_bin']:
        optimal_thresh_bin = config['thresh_bin']
        print(f'forcing thresh bin to {optimal_thresh_bin}')

    best_thresh = all_calo_2d.GetYaxis().GetBinCenter(optimal_thresh_bin)
    print(f'threshold: {best_thresh}')
    print('')

    # now do a five parameter fit with the optimal threshold
    best_T_hist = all_calo_2d.ProjectionX(
        'tMethodHist', optimal_thresh_bin, -1)
    five_param_tf1.SetParameter(0,
                                best_T_hist.GetBinContent(
                                    best_T_hist.FindBin(30)) * 1.6)
    bin_width = best_T_hist.GetBinWidth(1)

    calo_range = config['calo_range']
    calo_str = f'calos {calo_range[0]} to {calo_range[1]}'
    best_T_hist.SetTitle(
        f'T-Method, {calo_str}, {best_thresh/1000:.2f} GeV threshold; ' +
        f'time [#mus]; N / {bin_width:.3f} #mus')

    print('five parameter fit to T Method histogram...')

    resids, fft, cbo_freq = fit_and_fft(
        best_T_hist, five_param_tf1, 'fiveParamAllCalos',
        config['fit_options'],
        config['fit_start'], config['fit_end'], True)
    print_fit_plots(best_T_hist, fft, cbo_freq, 'fiveParamAllCalos', pdf_dir)

    print('')

    # adjust manually if we got a strange cbo frequency
    if not 0.2 < cbo_freq < 0.5:
        cbo_freq = config['cbo_freq_guess']

    # adjust fit start time to closest zero crossing, if requested
    if config['closest_zero_crossing']:
        Ac = five_param_tf1.GetParameter(2)
        As = five_param_tf1.GetParameter(3)
        phi = math.atan2(As, Ac)
        omega_guess = blinder.paramToFreq(five_param_tf1.GetParameter(4))

        old_start = config['fit_start']
        config['fit_start'] = closest_zero_crossing(config['fit_start'],
                                                    omega_guess,
                                                    phi)

        adjustment = (config['fit_start'] - old_start) * 1000

        print(f'adjusted fit start time by {adjustment} ns')
        print(f"new start time: {config['fit_start']}")
    else:
        print('did not adjust the fit start time')

    print(f'estimated CBO frequency is {cbo_freq:.2f} MHz')
    print(f'this correspons to n = {n_of_CBO_freq(cbo_freq):.3f}')

    print('\nfitting with cbo N term...')
    with_cbo_tf1 = build_CBO_only_func(five_param_tf1, cbo_freq, config)

    resids, fft = fit_and_fft(
        best_T_hist, with_cbo_tf1, 'cboFitAllCalos',
        config['fit_options'],
        config['fit_start'], config['fit_end'])
    print_fit_plots(best_T_hist, fft, cbo_freq, 'cboFitAllCalos', pdf_dir)

    cbo_freq = with_cbo_tf1.GetParameter(9) / 2 / math.pi

    print(f'\nfitted CBO frequency is {cbo_freq: .3f} MHz')
    print(f'this correspons to n = {n_of_CBO_freq(cbo_freq):.3f}')

    print('\nfitting with VW N term...')

    vw_tf1 = build_CBO_VW_func(with_cbo_tf1, cbo_freq, config)

    resids, fft = fit_and_fft(
        best_T_hist, vw_tf1, 'vwFitAllCalos',
        config['fit_options'],
        config['fit_start'], config['fit_end'])
    print_fit_plots(best_T_hist, fft, cbo_freq, 'vwFitAllCalos', pdf_dir)

    print('\npreparing muon loss histograms...')
    muon_hists = prepare_loss_hist(config, best_T_hist)
    c, _ = plot_loss_hists(*muon_hists[1:])
    c.Print(f'{pdf_dir}/lostMuonPlot.pdf')

    print('\nfitting with muon loss term included...')
    loss_tf1 = build_losses_func(vw_tf1, config)

    resids, fft = fit_and_fft(
        best_T_hist, loss_tf1, 'lossFitAllCalos',
        config['fit_options'],
        config['fit_start'], config['fit_end'])
    print_fit_plots(best_T_hist, fft, cbo_freq, 'lossFitAllCalos', pdf_dir)

    print('fitting with full model over full range...')

    full_fit_tf1 = build_full_fit_tf1(loss_tf1, config)
    full_fit_tf1.SetName('tMethodFit')

    if len(pu_unc_factors):
        print('adjusting T-Method hist with pileup-corrected bin errors')
        # one more fit using pu_uncertainty factors
        if len(pu_unc_factors) != best_T_hist.GetNbinsX():
            raise ValueError('Number of T-Method pileup uncertainty factors'
                             ' does not match the number of bins'
                             ' in the T-Method histogram!')

        for i_bin in range(1, best_T_hist.GetNbinsX() + 1):
            content = best_T_hist.GetBinContent(i_bin)
            old_err = np.sqrt(content) if content >= 0 else 0
            new_err = old_err * pu_unc_factors[i_bin - 1]

            best_T_hist.SetBinError(i_bin, new_err)

    resids, fft = fit_and_fft(
        best_T_hist, full_fit_tf1, 'fullFitAllCalos',
        config['fit_options'] + 'EM',
        config['fit_start'], config['extended_fit_end'],
        double_fit=True)

    print_fit_plots(best_T_hist, fft, cbo_freq, 'fullFitAllCalos', pdf_dir)

    # grab the covariance matrix from the full fit
    fr = best_T_hist.Fit(full_fit_tf1, 'EMS', '',
                         config['fit_start'], config['extended_fit_end'])

    corr_mat = fr.GetCorrelationMatrix()
    print('')
    print('parameter correlations with R: ')
    for i in range(full_fit_tf1.GetNpar()):
        corr = corr_mat(i, 4)
        if corr != 0:
            print(f'R-{full_fit_tf1.GetParName(i)} correlation: ' +
                  f'{corr:.2f}')

    print('\nfinished basic T-Method analysis\n')

    return best_T_hist, full_fit_tf1, fft, fr, optimal_thresh_bin, muon_hists


def T_method_calo_sweep(master_3d, model_fit, thresh_bin, config):
    ''' sweep over all calos, fitting each one
    returns a list of (hist, model_fit, resids, fft)) for each calo
    uses model_fit to determine guesses for first calo
    after that, it uses the results from the previous calo
    '''

    model_fit = configure_model_fit(model_fit,
                                    'TMethCaloModelFit',
                                    config['calo_sweep'])

    print('T Method calo sweep...')
    results = []
    for i in range(1, master_3d.GetNbinsZ() + 1):
        model_fit = clone_full_fit_tf1(model_fit, f'Calo{i}TFit')

        print(f'calo {i}')

        # adjust N only for the first calo
        adjust_N = True if i == 1 else False

        hist, resids, fft = fit_slice(master_3d, f'Calo{i}THist', model_fit,
                                      fit_options=config['fit_options'],
                                      fit_range=(config['fit_start'],
                                                 config['fit_end']),
                                      energy_bins=(
                                          thresh_bin, master_3d.GetNbinsY()),
                                      calo_bins=(i, i),
                                      adjust_N=adjust_N)

        hist.SetTitle(f'calo {i} T-Method')
        fft.SetTitle(f'calo {i} T-Method')

        results.append((hist, model_fit, resids, fft))

    return results


def A_weighted_calo_sweep(master_3d, model_fit, a_vs_e_spline, config,
                          min_e=1000, max_e=3000):
    ''' sweep over all calos, apply A-weighted fit to each one
    returns a list of (hist, model_fit, resids, fft)) for each calo
    uses model_fit to determine guesses for first calo
    after that, it uses the results from the previous calo
    '''

    model_fit = configure_model_fit(model_fit,
                                    'AWeightCaloModelFit',
                                    config['calo_sweep'])

    print('A-Weighted calo sweep...')
    results = []
    for i in range(1, master_3d.GetNbinsZ() + 1):
        model_fit = clone_full_fit_tf1(model_fit, f'Calo{i}AWeightFit')

        print(f'calo {i}')

        master_3d.GetZaxis().SetRange(i, i)

        calo_2d = master_3d.Project3D('yx')
        calo_2d.SetName(f'aWeightCalo{i}')

        hist = build_a_weight_hist(
            calo_2d, a_vs_e_spline, f'Calo{i}AWeightHist')

        time_axis = calo_2d.GetXaxis()
        bin_width = time_axis.GetBinWidth(1)
        hist.SetTitle(f'Calo {i} A-Weighted;' +
                      f't [#mus]; N / {bin_width:.3f} #mus')

        if i == 1:
            model_fit.SetParameter(
                0, hist.GetBinContent(hist.FindBin(30)) * 1.6)

        resids, fft = fit_and_fft(hist, model_fit, hist.GetName(),
                                  config['fit_options'],
                                  config['fit_start'],
                                  config['fit_end'],
                                  double_fit=True)

        fft.SetTitle(f'calo {i} A-Weighted')

        results.append((hist, model_fit, resids, fft))

    master_3d.GetZaxis().SetRange(1, master_3d.GetNbinsZ())
    return results


def energy_sweep(master_3d, model_fit, config):
    ''' sweep over energy bins, fitting each one
    returns a list of (hist, [e_low, e_high], model_fit, resids, fft))
    for each energy
    uses model_fit as starting guess for the first energy bin
    after that, it uses the results from the previous bin

    n_slices is approximate, the actual number will be such that
    each histogram has the same energy width,
    and all histograms span the range from min_e to max_e
    '''

    model_fit = configure_model_fit(model_fit,
                                    'energySweepModelFit',
                                    config['E_binned_ana'])

    print('energy binned sweep...')

    bin_ranges = calculate_E_bin_ranges(master_3d, config)

    results = []

    r_guess = model_fit.GetParameter(4)

    for i, (low_bin, high_bin) in enumerate(bin_ranges):
        energy_axis = master_3d.GetYaxis()
        low_e = energy_axis.GetBinLowEdge(low_bin)
        high_e = energy_axis.GetBinLowEdge(high_bin + 1)
        avg_e = 0.5 * (low_e + high_e)

        model_fit = clone_full_fit_tf1(model_fit,
                                       f'eFit{avg_e:.0f}')
        model_fit.SetParameter(4, r_guess)

        print(f'slice from {low_e:.0f} to {high_e:.0f} MeV')

        # # adjust N only for the first calo
        adjust_N = True if i == 0 else False

        calo_range = config['calo_range']

        hist, resids, fft = \
            fit_slice(master_3d, f'eBin{avg_e:.0f}', model_fit,
                      fit_options=config['fit_options'],
                      fit_range=(config['fit_start'],
                                 config['fit_end']),
                      energy_bins=(low_bin, high_bin),
                      calo_bins=(calo_range[0], calo_range[1]),
                      adjust_N=adjust_N)

        hist.SetTitle(
            f'calos {calo_range[0]} through {calo_range[1]}, ' +
            f'{low_e: .0f} to {high_e: .0f} MeV')
        fft.SetTitle(
            f'calos {calo_range[0]} through {calo_range[1]}, ' +
            f'{low_e:.0f} to {high_e:.0f} MeV')

        results.append((hist, (low_e, high_e), model_fit, resids, fft))

    return results


def calculate_E_bin_ranges(master_3d, config):
    ''' calculates energy bin ranges for each energy binned hist in the
    energy binned analysis defined by config, the configuration dictionary
    returns a list of tuples, each containing the
    low bin and high bin for an energy binned histogram,
    [(low1, high1), (low2, high2), ..., (lown, highn)]
    '''

    e_conf = config['E_binned_ana']
    min_e = e_conf['min_E']
    max_e = e_conf['max_E']
    n_slices = e_conf['n_bins']

    # convert energy range into bin index ranges
    energy_axis = master_3d.GetYaxis()

    low_bin = energy_axis.FindBin(min_e)
    high_bin = energy_axis.FindBin(max_e) + 1

    bins_per_slice = math.ceil((high_bin - low_bin) / n_slices)
    bin_ranges = [(start, start + bins_per_slice - 1)
                  for start in range(low_bin, high_bin, bins_per_slice)]

    return bin_ranges


def T_meth_pu_mult_scan(corrected_2d, uncorrected_2d,
                        thresh_bin, model_fit, scales, config,
                        pu_unc_facs=None, max_thresh_bin=-1):
    ''' vary the pileup multiplier and fit

    corrected_2d: pileup corrected 2d histogram
    uncorrected_2d: pileup uncorrected 2d histogram
    thresh_bin: T-Method threshold energy bin
    model_fit: fit function to use, should be a "full fit"
    scales: a list of scale factors to use
    config: analysis config dictionary

    returns list of fit functions at each scale factor
    '''

    uncorrected_T_hist = uncorrected_2d.ProjectionX(
        'uncorrectedT', thresh_bin, max_thresh_bin)

    # pu perturbation is uncorrected T hist minus corrected T hist
    pu_pert = uncorrected_T_hist.Clone()
    pu_pert.SetName('puPertTMeth')
    corrected_T_hist = corrected_2d.ProjectionX('correctedTMeth',
                                                thresh_bin, max_thresh_bin)
    pu_pert.Add(corrected_T_hist, -1)

    pu_scan_fit = clone_full_fit_tf1(model_fit,
                                     f'tFitScaled{scales[0]}')

    fits = []
    for scale_factor in scales:
        fit_name = f'tFitScaled{scale_factor}'
        if len(fits) == 0:
            fit = pu_scan_fit
        else:
            fit = clone_full_fit_tf1(fits[-1], fit_name)

        fit_hist = uncorrected_T_hist.Clone()
        fit_hist.Add(pu_pert, -1 * scale_factor)

        fit_hist.ResetStats()

        for i_bin in range(1, corrected_T_hist.GetNbinsX() + 1):
            # if no pu_unc factors, use corrected bin errors
            # else use the bin_error and a scaled version of the pu_unc_factors
            if pu_unc_facs is None:
                fit_hist.SetBinError(i_bin,
                                     corrected_T_hist.GetBinError(i_bin))
            else:
                scaled_unc_fac = scale_factor * \
                    (pu_unc_facs[i_bin - 1] - 1) + 1

                new_err = fit_hist.GetBinError(i_bin) * scaled_unc_fac

                fit_hist.SetBinError(i_bin, new_err)

        fit_hist.Fit(fit, config['fit_options'] + '0', ' ',
                     config['fit_start'], config['extended_fit_end'])

        fits.append(fit)

    return fits


def make_pu_scan_graphs(scales, fits):
    ''' returns chi_2 vs scan_x
    and also parameter_graphs dictionary for all
    non fixed fit parameters '''

    chi2_g = r.TGraph()

    par_gs = {}

    for scale, fit in zip(scales, fits):
        chi2_g.SetPoint(chi2_g.GetN(), scale,
                        fit.GetChisquare())

        update_par_graphs(scale, fit, par_gs)

    return chi2_g, par_gs


def make_calo_sweep_graphs(calo_sweep_res):
    ''' returns chi2_g, par_gs '''
    chi2_g = r.TGraphErrors()
    chi2_g.SetName('calo_chi2_g')
    chi2_g.SetTitle(';calo num; #chi^{2}/ndf')

    par_gs = {}

    for calo_num, (hist, fit, _, fft) in \
            enumerate(calo_sweep_res, 1):

        pt_num = calo_num - 1

        chi2_g.SetPoint(pt_num, calo_num,
                        fit.GetChisquare() / fit.GetNDF())
        chi2_g.SetPointError(pt_num, 0, math.sqrt(2 / fit.GetNDF()))

        update_par_graphs(calo_num, fit, par_gs)

    return chi2_g, par_gs


def make_E_sweep_graphs(energy_sweep_res):
    ''' returns chi2_g, par_gs '''
    chi2_g = r.TGraphErrors()
    chi2_g.SetName('energy_chi2_g')
    chi2_g.SetTitle(';energy [MeV]; #chi^{2}/ndf')

    par_gs = {}

    for pt_num, (hist, (low_e, high_e), fit, _, fft) in \
            enumerate(energy_sweep_res):

        energy = 0.5 * (low_e + high_e)

        chi2_g.SetPoint(pt_num, energy,
                        fit.GetChisquare() / fit.GetNDF())
        chi2_g.SetPointError(pt_num, 0, math.sqrt(2 / fit.GetNDF()))

        update_par_graphs(energy, fit, par_gs)

    return chi2_g, par_gs


def make_loss_correction_hists(cumu_loss, K_loss, config):
    ''' make scaled loss correction histograms
    both starting from t = 0 and starting from fit_start
    returns (from_zero, from_t_start)'''
    from_zero = cumu_loss.Clone()
    from_zero.Scale(100 * K_loss)
    from_zero.SetName('lossCorrectionFromZero')
    from_zero.SetTitle('loss correction from t = 0;time [#mus];' +
                       ' fractional loss correction [%]')

    t_start = config['fit_start']
    from_t_start = from_zero.Clone()
    from_t_start.SetName(f'lossCorrectionFrom{t_start:.1f}')
    from_t_start.SetTitle(f'loss correction from {t_start:.1f} #mus ;' +
                          'time [#mus]; fractional loss correction [%]')

    start_frac = from_zero.GetBinContent(from_zero.FindBin(t_start))

    for i_bin in range(1, from_t_start.GetNbinsX() + 1):
        content = from_t_start.GetBinContent(i_bin)

        if from_t_start.GetBinCenter(i_bin) < t_start:
            from_t_start.SetBinContent(i_bin, 0)
        else:
            new_content = (content - start_frac) / (1 - start_frac / 100.0)
            from_t_start.SetBinContent(i_bin, new_content)

    t_max = from_t_start.GetBinLowEdge(from_t_start.GetNbinsX() + 1)
    from_t_start.GetXaxis().SetRangeUser(t_start, t_max)

    return from_zero, from_t_start


#
# Stuff for A-Weighted analysis
#


def build_A_vs_E_spline(a_vs_e, phi_vs_e, deriv_cut=0.01):
    '''
    builds a signed A vs E spline out of
    an unsigned a_vs_e graph and a phi_vs_e graph
    returns a_vs_e_spline, signed_a_vs_e_graph
    '''

    # find energy at which asymmetry changes sign
    # base this on a large slope of phi versus E
    x, y = r.Double(), r.Double()
    phi_vs_e.GetPoint(0, x, y)
    last_x, last_y = float(x), float(y)
    inversion_e = 0
    for pt_num in range(1, phi_vs_e.GetN()):
        phi_vs_e.GetPoint(pt_num, x, y)
        abs_deriv = abs((y - last_y) / (x - last_x))
        if abs_deriv > deriv_cut:
            inversion_e = float(x)
            break
        last_x, last_y = float(x), float(y)

    # build signed version of A versus E
    signed_a_vs_e = r.TGraph()
    for pt_num in range(a_vs_e.GetN()):
        a_vs_e.GetPoint(pt_num, x, y)
        adjusted_y = y
        if x < inversion_e:
            adjusted_y *= -1

        signed_a_vs_e.SetPoint(pt_num, x, adjusted_y)

    # build an interpolating spline
    a_vs_e_spline = r.TSpline3('A vs E spline', signed_a_vs_e)
    a_vs_e_spline.SetName('aVsESpline')

    signed_a_vs_e.SetName('aVsEGraph')
    signed_a_vs_e.SetTitle(';energy [MeV]; A')

    return signed_a_vs_e, a_vs_e_spline


def build_a_weight_hist(spec_2d, a_vs_e_spline, name,
                        min_e=1000, max_e=3000, energyWeighted = False):
    ''' builds an asymmetry weighted histogram,
    pretty self explanatory
    '''

    # don't go above the max energy available from the a_vs_e_spline
    if max_e > a_vs_e_spline.GetXmax():
        max_e = a_vs_e_spline.GetXmax()

    time_axis = spec_2d.GetXaxis()
    bin_width = time_axis.GetBinWidth(1)
    a_weight_hist = r.TH1D(name, 'A-Weighted;' +
                           f' t [#mus]; N / {bin_width:.3f} #mus',
                           time_axis.GetNbins(),
                           time_axis.GetBinLowEdge(1),
                           time_axis.GetBinUpEdge(time_axis.GetNbins()))
    a_weight_hist.Sumw2()

    start_bin = spec_2d.GetYaxis().FindBin(min_e)
    end_bin = spec_2d.GetYaxis().FindBin(max_e) + 1

    for e_bin in range(start_bin, end_bin):
        energy_slice = spec_2d.ProjectionX(
            f'{name}e_slice{e_bin}', e_bin, e_bin)
        r.SetOwnership(energy_slice, True)

        energy = spec_2d.GetYaxis().GetBinCenter(e_bin)
        if energyWeighted:
            a_weight_hist.Add(energy_slice, energy)
        else:
            a_weight_hist.Add(energy_slice, a_vs_e_spline.Eval(energy))

    return a_weight_hist


def A_weight_pu_mult_scan(corrected_2d, uncorrected_2d,
                          a_vs_e_spline,
                          model_fit, scales, config,
                          pu_unc_facs=None,
                          min_e=1000, max_e=3000):
    ''' vary the pileup multiplier and fit, for A_Weighted analysis

    corrected_2d: pileup corrected 2d histogram
    uncorrected_2d: pileup uncorrected 2d histogramx
    a_vs_e_spline: asymmetry versus energy model
    model_fit: fit function to use, should be a "full fit"
    scales: a list of scale factors to use
    config: analysis config dictionary
    returns list of fit functions at each scale factor
    min_e: minimum energy to include
    max_e: maximum energy to include
    '''

    # 2d pileup perturbation is uncorrected minus corrected
    pu_pert_2d = uncorrected_2d.Clone()
    pu_pert_2d.SetName('2d_pu_pert')
    pu_pert_2d.Add(corrected_2d, -1)
    corrected_a_hist = build_a_weight_hist(
        corrected_2d, a_vs_e_spline,
        f'corrected_aweight', min_e=min_e, max_e=max_e)

    pu_scan_fit = clone_full_fit_tf1(model_fit,
                                     f'aWeightScaled{scales[0]}')

    fits = []
    for scale_factor in scales:
        fit_name = f'aWeightFitScaled{scale_factor}'
        if len(fits) == 0:
            fit = pu_scan_fit
        else:
            fit = clone_full_fit_tf1(fits[-1], fit_name)

        scaled_2d = uncorrected_2d.Clone()
        scaled_2d.Add(pu_pert_2d, -1 * scale_factor)

        scaled_a = build_a_weight_hist(scaled_2d, a_vs_e_spline,
                                       f'scaled_aweight_{scale_factor}',
                                       min_e=min_e,
                                       max_e=max_e)

        for i_bin in range(1, corrected_a_hist.GetNbinsX() + 1):
            # if no pu_unc factors, use corrected bin errors
            # else use the bin_error * scale_factor * uncertainty factor
            if pu_unc_facs is None:
                scaled_a.SetBinError(i_bin,
                                     corrected_a_hist.GetBinError(i_bin))
            else:
                scaled_unc_fac = scale_factor * \
                    (pu_unc_facs[i_bin - 1] - 1) + 1

                new_err = scaled_a.GetBinError(i_bin) * scaled_unc_fac

                scaled_a.SetBinError(i_bin, new_err)

        scaled_a.Fit(fit, config['fit_options'] + '0', ' ',
                     config['fit_start'], config['extended_fit_end'])

        fits.append(fit)

    return fits


def get_residuals_distribution(residuals_hist, name, config,
                               n_bins=100, range_min=-5, range_max=5):
    ''' histogram the residuals within the fit window '''
    resid_dist = r.TH1D(f'{residuals_hist.GetName()}_distro',
                        f'{name}; pull; n bins', 100, -5, 5)

    for i_bin in range(1, residuals_hist.GetNbinsX() + 1):
        bin_t = residuals_hist.GetBinCenter(i_bin)
        if config['fit_start'] < bin_t < config['extended_fit_end']:
            resid_dist.Fill(residuals_hist.GetBinContent(i_bin))

    resid_dist.Fit('gaus', 'q', '', -1.5, 1.5)

    return resid_dist


#
# plotting and printing functions
#
# print functions make pdf plots
# plot functions return a canvas (and newly created objects in the canvas)
#


def plot_hist(hist):
    c = r.TCanvas()
    hist.GetYaxis().SetRangeUser(10, hist.GetMaximum())
    hist.Draw()
    c.SetLogy(1)

    return c, []


def plot_fft(fft, cbo_freq):
    c = r.TCanvas()

    fft.Draw()
    lns = plot_expected_freqs(fft, cbo_freq)

    return c, lns


def print_fit_plots(hist, fft, cbo_freq, fit_name, pdf_dir):
    c, _ = plot_hist(hist)
    c.Draw()

    c.Print(f'{pdf_dir}/{fit_name}.pdf')

    c, _ = plot_fft(fft, cbo_freq)
    c.Print(f'{pdf_dir}/{fit_name}FFT.pdf')


def plot_expected_freqs(resid_fft, cbo_freq, f_c=1.0 / 0.149, f_a=0.2291):
    y_min, y_max = 0, resid_fft.GetMaximum() * 1.1
    resid_fft.GetYaxis().SetRangeUser(y_min, y_max)

    n = n_of_CBO_freq(cbo_freq)

    try:
        expected_freqs = {'f_{cbo}': cbo_freq, 'f_{y}': math.sqrt(n) * f_c,
                          'f_{vw}': f_c * (1 - 2 * math.sqrt(n)),
                          'f_{a} - f_{cbo}': cbo_freq - f_a,
                          'f_{a} + f_{cbo}': cbo_freq + f_a,
                          '2*f_{cbo}': cbo_freq * 2}
    except ValueError:
        expected_freqs = []

    lns = []
    for name in expected_freqs:
        freq = expected_freqs[name]
        ln = r.TLine(freq, y_min, freq, y_max)
        ln.SetLineStyle(2)
        ln.SetLineWidth(2)
        ln.Draw()
        lns.append(ln)

    return lns


def plot_threshold_sweep(all_calo_2d, r_precisions, best_bin):
    # make a plot of the threshold sweep result
    precision_vs_threshg = r.TGraph()
    for i, (e_bin, precision) in enumerate(r_precisions):
        precision_vs_threshg.SetPoint(i,
                                      all_calo_2d.GetYaxis().
                                      GetBinCenter(e_bin),
                                      precision)

    best_Ecut = all_calo_2d.GetYaxis().GetBinCenter(best_bin)
    precision_vs_threshg.SetTitle(f'optimal threshold: {best_Ecut:.0f} MeV;'
                                  'energy threshold [MeV];'
                                  ' #omega_{a} uncertainty [ppm]')

    c = r.TCanvas()
    precision_vs_threshg.Draw('ap')
    y_min, y_max = min(precision_vs_threshg.GetY()) * \
        0.9, max(precision_vs_threshg.GetY()) * 1.1
    precision_vs_threshg.GetYaxis().SetRangeUser(y_min, y_max)

    line = r.TLine(best_Ecut, y_min, best_Ecut, y_max)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw()
    c.SetLogy(0)

    return c, [precision_vs_threshg, line]


def plot_loss_hists(lost_muon_rate,
                    lost_muon_prob):
    # scale histograms for plotting
    lost_muon_rate.Scale(1.0 / lost_muon_rate.Integral('width'))
    lost_muon_prob.Scale(1.0 / lost_muon_prob.Integral('width'))
    plot_cumulative = lost_muon_prob.GetCumulative()
    plot_cumulative.Scale(
        lost_muon_rate.GetMaximum() / plot_cumulative.GetMaximum())

    c = r.TCanvas()
    lost_muon_rate.SetLineColor(r.kCyan)
    lost_muon_rate.Draw('hist')
    lost_muon_rate.SetTitle('')
    lost_muon_rate.GetXaxis().SetRangeUser(0, 300)
    lost_muon_rate.GetXaxis().SetTitle('time [#mus]')
    lost_muon_rate.GetYaxis().SetRangeUser(1e-6, 1)

    lost_muon_prob.SetLineColor(r.kRed)
    lost_muon_prob.Draw('hist same')
    plot_cumulative.SetLineColor(r.kBlue)
    plot_cumulative.Draw('hist same')
    c.SetLogy(1)
    c.Draw()
    leg = r.TLegend(0.5, 0.8, 0.8, 0.5)
    leg.SetLineColor(r.kWhite)
    leg.SetFillColor(r.kWhite)
    leg.SetShadowColor(r.kWhite)
    leg.AddEntry(lost_muon_rate, 'L(t)', 'l')
    leg.AddEntry(lost_muon_prob, 'L(t) #bullet exp(t/#tau)', 'l')
    leg.AddEntry(plot_cumulative, '#int_{0}^{t} L(t\')'
                 ' #bullet exp(t\'/#tau) dt\'', 'l')

    leg.Draw()

    return c, [plot_cumulative, leg]


def print_calo_sweep_res(calo_sweep_res, chi2_g, par_gs, pdf_dir, suffix):
    print('making calorimeter sweep plots')

    r.gStyle.SetStatH(0.15)

    for calo_num, (hist, fit, _, fft) in \
            enumerate(calo_sweep_res, 1):

        cbo_freq = fit.GetParameter(9) / 2 / math.pi
        print_fit_plots(hist, fft, cbo_freq,
                        f'{calo_dirname}/calo{calo_num}{suffix}', pdf_dir)

    c = r.TCanvas()
    chi2_g.Draw('ap')
    chi2_g.GetXaxis().SetLimits(0, 25)
    ln = r.TLine(0, 1, 25, 1)
    ln.SetLineWidth(2)
    ln.SetLineStyle(2)
    ln.Draw()

    c.Print(f'{pdf_dir}/{calo_dirname}/chi2VsCalo{suffix}.pdf')

    for par_num in par_gs:
        g = par_gs[par_num]
        g.GetXaxis().SetTitle('calo num')

        g.Draw('ap')

        par_name = g.GetYaxis().GetTitle()
        par_name = strip_par_name(par_name)

        g.GetXaxis().SetLimits(0, 25)

        c.Print(f'{pdf_dir}/{calo_dirname}/{par_name}VsCalo{suffix}.pdf')


def print_energy_sweep_res(energy_sweep_res, chi2_g, par_gs, pdf_dir):
    print('making energy sweep plots')

    r.gStyle.SetStatH(0.2)

    for pt_num, (hist, (low_e, high_e), fit, _, fft) in \
            enumerate(energy_sweep_res):

        energy = 0.5 * (low_e + high_e)

        cbo_freq = fit.GetParameter(9) / 2 / math.pi
        print_fit_plots(hist, fft, cbo_freq,
                        f'{e_sweep_dirname}/eFit{energy:.0f}', pdf_dir)

    c = r.TCanvas()
    chi2_g.SetTitle(';energy [MeV]; #chi^{2}/ndf')
    chi2_g.Draw('ap')

    c.Print(f'{pdf_dir}/{e_sweep_dirname}/chi2VsEnergy.pdf')

    for par_num in par_gs:
        g = par_gs[par_num]
        g.GetXaxis().SetTitle('energy [MeV]')

        g.Draw('ap')

        par_name = g.GetYaxis().GetTitle()
        par_name = strip_par_name(par_name)

        c.Print(f'{pdf_dir}/{e_sweep_dirname}/{par_name}VsEnergy.pdf')


def plot_pu_sweep_R(par_gs, title):
    ''' returns canvas
    '''
    # make R plot
    c1 = r.TCanvas()
    r_g = par_gs[4]
    r_g.SetTitle(f'{title};pu multiplier; R')
    r_g.Fit('pol1', 'EMq')
    r_g.Draw('ap E0X')

    return c1


def plot_pu_sweep_chi2(chi2_g, title):
    ''' returns canvas
    '''
    c1 = r.TCanvas()
    chi2_g.SetTitle(f'{title};pu multiplier; #chi^{{2}}')
    chi2_g.Draw()

    # make chi2 plot

    chi2s = list(chi2_g.GetY())
    min_chi2 = min(chi2s)
    min_chi2_mult = chi2_g.GetX()[chi2s.index(min_chi2)]

    chi2_g.Draw('ap')

    chi2_g_fit = r.TF1('pu_chi2_fit', '[1]*(x-[0])^2 + [2]')
    chi2_g_fit.SetParName(0, 'a')
    chi2_g_fit.SetParName(1, 'b')
    chi2_g_fit.SetParName(2, 'c')
    chi2_g_fit.SetParameter(0, min_chi2_mult)
    chi2_g_fit.SetParameter(1, 0)
    chi2_g_fit.SetParameter(2, min_chi2)
    chi2_g.Fit(chi2_g_fit, 'EMq')

    return c1


def plot_a_vs_e_curve(a_vs_e_g, a_vs_e_spline):
    c = r.TCanvas()

    a_vs_e_g.Draw('ap')
    a_vs_e_g.GetXaxis().SetLimits(0, 3000)
    a_vs_e_spline.Draw('same')
    ln = r.TLine(0, 0, 3000, 0)
    ln.SetLineStyle(2)
    ln.Draw()

    return c, ln


#
# driver function to run the analysis
#


def run_analysis(config):
    #
    # build necessary output directories
    #
    out_dir = config['out_dir']
    subprocess.call(f'mkdir -p {out_dir}'.split())
    pdf_dir = f'{out_dir}/plots'
    subprocess.call(f'mkdir -p {pdf_dir}'.split())
    calo_dir = f'{pdf_dir}/{calo_dirname}'
    subprocess.call(f'mkdir -p {calo_dir}'.split())
    e_bin_dir = f'{pdf_dir}/{e_sweep_dirname}'
    subprocess.call(
        f'mkdir -p {e_bin_dir}'.split())

    #
    # setup blinder object and read histograms from input file
    #

    blinder = Blinders(FitType.Omega_a, config['blinding_phrase'])

    master_3d = get_histogram(config['hist_name'], config['file_name'])
    master_3d.SetName('master_3d')
    uncorrected_3d = get_histogram(config['uncor_hist_name'],
                                   config['file_name'])

    calo_range = config['calo_range']
    uncorrected_3d.GetZaxis().SetRange(calo_range[0], calo_range[1])
    master_3d.GetZaxis().SetRange(calo_range[0], calo_range[1])

    all_calo_2d = master_3d.Project3D('yx')
    all_calo_2d.SetName('all_calo_2d')

    r.gStyle.SetStatW(0.25)
    r.gStyle.SetStatH(0.4)

    pu_unc_file = config.get('pu_uncertainty_file', None)

    if pu_unc_file is not None:
        # load pileup uncertainty factors
        factor_array = np.loadtxt(pu_unc_file, skiprows=1)
        T_meth_unc_facs = factor_array[:, 1]
        A_weight_unc_facs = factor_array[:, 2]
    else:
        T_meth_unc_facs = []
        A_weight_unc_facs = []

    #
    # Start with a T-Method analysis
    #

    T_hist, full_fit, T_fft, T_result, thresh, muon_hists = T_method_analysis(
        all_calo_2d, blinder, config, T_meth_unc_facs)
    T_resids = build_residuals_hist(
        T_hist, full_fit, True, name='TMethodResiduals')
    T_resid_dist = get_residuals_distribution(T_resids, 'T-Method', config)

    # build fractional loss hists
    frac_losses = make_loss_correction_hists(muon_hists[0],
                                             full_fit.GetParameter(14), config)

    print('T-Method pileup multiplier scan...')

    # do T-Method pileup multiplier scan
    uncorrected_2d = uncorrected_3d.Project3D('yx')
    uncorrected_2d.SetName('uncor_2d')

    pu_scan_fit = clone_full_fit_tf1(full_fit, 'pu_scan_fit')
    pu_scan_fit.SetParLimits(6, 100, 400)

    scale_factors = [i / 10 for i in range(15)]
    unc_facs = T_meth_unc_facs if len(T_meth_unc_facs) else None
    pu_scale_fits = T_meth_pu_mult_scan(all_calo_2d, uncorrected_2d,
                                        thresh, pu_scan_fit,
                                        scale_factors, config,
                                        pu_unc_facs=unc_facs)
    t_pu_chi2_g, t_pu_par_gs = make_pu_scan_graphs(
        scale_factors, pu_scale_fits)
    c1 = plot_pu_sweep_R(t_pu_par_gs, 'T-Method')
    c1.Print(f'{pdf_dir}/tMethodPuSweepR.pdf')
    c1 = plot_pu_sweep_chi2(t_pu_chi2_g, 'T-Method')
    c1.Print(f'{pdf_dir}/tMethodPuSweepchi2.pdf')

    #
    # Do a per calo analysis
    #

    print('Per calo analysis...')

    # T-Method fits per calo
    calo_sweep_res = T_method_calo_sweep(
        master_3d, full_fit, thresh, config)
    calo_chi2_g, calo_sweep_par_gs = make_calo_sweep_graphs(calo_sweep_res)
    print_calo_sweep_res(calo_sweep_res, calo_chi2_g,
                         calo_sweep_par_gs, pdf_dir, 'TFit')

    print('energy binned analysis...')

    # do the energy bin sweeps
    e_sweep_res = energy_sweep(master_3d, full_fit, config)
    e_sweep_chi2_g, e_sweep_par_gs = make_E_sweep_graphs(e_sweep_res)
    print_energy_sweep_res(e_sweep_res, e_sweep_chi2_g,
                           e_sweep_par_gs, pdf_dir)

    print('A-Weighted Analysis...')

    signed_a_vs_e, a_vs_e_spline = build_A_vs_E_spline(
        e_sweep_par_gs[2], e_sweep_par_gs[3])
    c, _ = plot_a_vs_e_curve(signed_a_vs_e, a_vs_e_spline)
    c.Print(f'{pdf_dir}/aVsECurve.pdf')

    a_weight_hist = build_a_weight_hist(
        all_calo_2d, a_vs_e_spline, 'aWeightHist')

    a_weight_fit = clone_full_fit_tf1(full_fit, 'aWeightFit')

    # A-Weighted fit sometimes has issues with VW, set some parameter limits
    # (if not already limited)
    if not is_limited_param(a_weight_fit, 10):
        # limit vw lifetime, but don't weaken limits
        a_weight_fit.SetParLimits(10,
                                  0.8 * a_weight_fit.GetParameter(10),
                                  1.2 * a_weight_fit.GetParameter(10))

    if not is_limited_param(a_weight_fit, 13):
        # limit VW frequency parameter
        freq_par_err = full_fit.GetParError(13)
        freq_par = full_fit.GetParameter(13)
        a_weight_fit.SetParLimits(13,
                                  freq_par - 5 * freq_par_err,
                                  freq_par + 5 * freq_par_err)

    a_weight_fit.SetParameter(0,
                              a_weight_hist.GetBinContent(
                                  a_weight_hist.FindBin(30)) * 1.6)

    if len(A_weight_unc_facs):
        print('adjusting A-Weighted hist with pileup-corrected bin errors')
        # one more fit using pu_uncertainty factors
        if len(A_weight_unc_facs) != a_weight_hist.GetNbinsX():
            raise ValueError('Number of A-Weighted pileup uncertainty factors'
                             ' does not match the number of bins'
                             ' in the A-Weighted histogram!')

        for i_bin in range(1, a_weight_hist.GetNbinsX() + 1):
            new_err = a_weight_hist.GetBinError(i_bin) \
                * A_weight_unc_facs[i_bin - 1]

            a_weight_hist.SetBinError(i_bin, new_err)

    resids, A_fft = fit_and_fft(
        a_weight_hist, a_weight_fit, 'fullFitAWeight',
        config['fit_options'] + 'EM',
        config['fit_start'], config['extended_fit_end'], double_fit=True)

    print_fit_plots(a_weight_hist, A_fft,
                    a_weight_fit.GetParameter(9) / 2 / math.pi,
                    'aWeightedFit', pdf_dir)

    # get full A-weighted fit result
    # A_result = a_weight_hist.Fit(a_weight_fit, config['fit_options'] + 'EMS',
    #                              '',
    #                              config['fit_start'],
    #                              config['extended_fit_end'], double_fit=True)

    # print out final A-Weighted fit result to make sure it was successful
    A_result = a_weight_hist.Fit(a_weight_fit, 'EMS',
                                 '',
                                 config['fit_start'],
                                 config['extended_fit_end'])

    A_resids = build_residuals_hist(
        a_weight_hist, a_weight_fit, True, name='AWeightedResiduals')
    A_resid_dist = get_residuals_distribution(A_resids, 'A-Weighted', config)

    print('A-Weighted pileup multiplier scan...')

    a_pu_fit = clone_full_fit_tf1(a_weight_fit, 'a_pu_fit')
    unc_facs = A_weight_unc_facs if len(A_weight_unc_facs) else None
    a_pu_scan_fits = A_weight_pu_mult_scan(all_calo_2d, uncorrected_2d,
                                           a_vs_e_spline, a_pu_fit,
                                           scale_factors, config,
                                           pu_unc_facs=unc_facs)

    a_pu_chi2_g, a_pu_par_gs = make_pu_scan_graphs(
        scale_factors, a_pu_scan_fits)
    c1 = plot_pu_sweep_R(a_pu_par_gs, 'A-Weighted')
    c1.Print(f'{pdf_dir}/aWeightPuSweepR.pdf')
    c1 = plot_pu_sweep_chi2(a_pu_chi2_g, 'A-Weighted')
    c1.Print(f'{pdf_dir}/aWeightPuSweepchi2.pdf')

    print('A-Weighted calo scan')

    # A-Weighted fits per calo
    calo_sweep_a_res = A_weighted_calo_sweep(
        master_3d, a_weight_fit, a_vs_e_spline, config)
    calo_a_chi2_g, calo_sweep_a_par_gs = make_calo_sweep_graphs(
        calo_sweep_a_res)
    print_calo_sweep_res(calo_sweep_a_res, calo_a_chi2_g,
                         calo_sweep_a_par_gs, pdf_dir, 'AWeight')

    if config['do_start_time_scans']:

        print('T-Method start time scan:')

        start_time_conf = config['start_time_scan']
        start_time_fit = configure_model_fit(full_fit,
                                             'start_time_fit',
                                             start_time_conf)

        t_scan_hist = T_hist.Clone()
        t_scan_hist.SetName('t_scan_hist')
        t_scan_chi2, t_scan_res = start_time_scan(
            t_scan_hist, start_time_fit,
            start=config['fit_start'],
            end=config['extended_fit_end'],
            step=start_time_conf['step'],
            n_pts=start_time_conf['n_pts'],
            fit_options=config['fit_options'] + 'E')
        t_scan_chi2.SetName('TMethodChi2VsStartTime')

        t_start_canvs = []
        for i, res in enumerate(t_scan_res):
            name = f'TMethodPar{res.par_num}StartScan'
            t_start_canvs.append(r.TCanvas(name, name))
            res.Draw()
            t_start_canvs[-1].SetName(f'TMethodPar{res.par_num}StartScan')
            t_start_canvs[-1].Print(
                f'{pdf_dir}/TMethodStartScan{res.par_num}.pdf')

        print('A-Weighted start time scan:')

        a_start_time_fit = configure_model_fit(
            a_weight_fit, 'a_start_time_fit', start_time_conf)

        a_scan_hist = a_weight_hist.Clone()
        a_scan_hist.SetName('a_scan_hist')
        a_scan_chi2, a_scan_res = start_time_scan(
            a_scan_hist, a_start_time_fit,
            start=config['fit_start'],
            end=config['extended_fit_end'],
            step=start_time_conf['step'],
            n_pts=start_time_conf['n_pts'],
            fit_options=config['fit_options'] + 'E')
        a_scan_chi2.SetName('AWeightedChi2VsStartTime')

        a_start_canvs = []
        for i, res in enumerate(a_scan_res):
            name = f'AWeightPar{res.par_num}StartScan'
            a_start_canvs.append(r.TCanvas(name, name))
            res.Draw()
            a_start_canvs[-1].Draw()
            a_start_canvs[-1].Print(
                f'{pdf_dir}/AWeightStartScan{res.par_num}.pdf')

    #
    # make an output file with resulting root objects
    #

    out_f_name = config['outfile_name']
    if not out_f_name.endswith('.root'):
        out_f_name = out_f_name + '.root'
    out_f = r.TFile(f'{out_dir}/{out_f_name}', 'recreate')

    # save A weighted and T Method plots
    t_dir = out_f.mkdir('T-Method')
    t_dir.cd()
    T_hist.Write()
    full_fit.Write()
    T_resids.Write()
    T_fft.Write()
    T_resid_dist.Write()
    T_result.SetName('TFitResult')
    T_result.Write()

    a_dir = out_f.mkdir('A-Weighted')
    a_dir.cd()
    a_weight_hist.Write()
    a_weight_fit.Write()
    A_fft.Write()
    A_resids.Write()
    A_resid_dist.Write()
    A_result.SetName('AFitResult')
    A_result.Write()

    # save A vs E model
    signed_a_vs_e.Write()
    a_vs_e_spline.Write()

    # muon loss stuff
    loss_dir = out_f.mkdir('muonLosses')
    loss_dir.cd()
    for hist in muon_hists:
        hist.Write()
    for hist in frac_losses:
        hist.Write()

    # save energy binned plots
    e_sweep_dir = out_f.mkdir('energySweep')
    e_sweep_dir.cd()
    for par_num in e_sweep_par_gs:
        graph = e_sweep_par_gs[par_num]
        graph.SetName(f'par{par_num}VsE')
        graph.Write()
    for result in e_sweep_res:
        result[0].Write()
        result[2].Write()
    e_sweep_chi2_g.Write()

    # save calo sweep plots
    calo_sweep_dir = t_dir.mkdir('caloSweep')
    calo_sweep_dir.cd()
    for par_num in calo_sweep_par_gs:
        graph = calo_sweep_par_gs[par_num]
        graph.SetName(f'par{par_num}VsCalo')
        graph.Write()
    for result in calo_sweep_res:
        result[0].Write()
        result[1].Write()
        result[-1].Write()
    calo_chi2_g.Write()

    calo_sweep_a_dir = a_dir.mkdir('caloSweep')
    calo_sweep_a_dir.cd()
    for par_num in calo_sweep_a_par_gs:
        graph = calo_sweep_a_par_gs[par_num]
        graph.SetName(f'par{par_num}VsCalo')
        graph.Write()
    for result in calo_sweep_a_res:
        result[0].Write()
        result[1].Write()
        result[-1].Write()
    calo_a_chi2_g.Write()

    # save pileup multiplier scan results
    t_pu_dir = t_dir.mkdir('pileupScan')
    t_pu_dir.cd()
    t_pu_chi2_g.SetName('tMethodChi2VsPuScale')
    t_pu_chi2_g.Write()
    for par_num in t_pu_par_gs:
        graph = t_pu_par_gs[par_num]
        graph.SetName(f'tMethodPar{par_num}VsPuScale')
        graph.Write()
    a_pu_dir = a_dir.mkdir('pileupScan')
    a_pu_dir.cd()
    a_pu_chi2_g.SetName('aWeightedChi2VsPuScale')
    a_pu_chi2_g.Write()
    for par_num in a_pu_par_gs:
        graph = a_pu_par_gs[par_num]
        graph.SetName(f'aWeightedPar{par_num}VsPuScale')
        graph.Write()

    if config['do_start_time_scans']:
        # save start time scan results
        t_start_dir = t_dir.mkdir('startTimeScan')
        t_start_dir.cd()
        t_scan_chi2.Write()
        for canv in t_start_canvs:
            canv.Write()

        a_start_dir = a_dir.mkdir('startTimeScan')
        a_start_dir.cd()
        a_scan_chi2.Write()
        for canv in a_start_canvs:
            canv.Write()

    out_f.Write()

    print('Done!')
