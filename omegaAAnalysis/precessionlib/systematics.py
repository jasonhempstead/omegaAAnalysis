# functions for running omega_a systematic sweeps
#
# Aaron Fienberg
# June 2019

import json
import gc
import ROOT as r
import numpy as np
from .calospectra import CaloSpectra
from .util import *
from .fitmodels import *
from .analysis import *

#
# Pileup phase sweep code
#


def pileup_phase_sweep(config, conf_dir):
    ''' run the pileup phase sensitivity scan '''
    print('\n---\nPileup phase sweep\n---')

    raw_f, fit_info = load_raw_and_analyzed_files(config, conf_dir)

    print('Building calo spectrum and pileup correction...')
    hist_name = config['hist_name']
    dir_name = hist_name.split('/')[0]
    ctag = raw_f.Get(f'{dir_name}/ctag')
    n_fills = ctag.GetEntries()
    print(f'{n_fills:.0f} fills')

    spec = CaloSpectra.from_root_file(f'{conf_dir}/{config["raw_file"]}',
                                      config['hist_name'],
                                      do_triple=True,
                                      single_param=True,
                                      param_guess=2 * 1.25 / n_fills / 25,
                                      pu_time_max=650)
    print(f'estimated deadtimes per calo: {spec.estimated_deadtimes(n_fills)}')

    # run the scan
    print('running pileup time shift scan...')
    outs_T = []
    outs_A = []

    shifts = list(range(config['shift_min'], config['shift_max'] + 1))
    for shift in shifts:
        print(f'shift {shift}....')
        out_T, out_A = fit_pu_shifted_T_and_A(shift, spec, fit_info, config)
        outs_T.append(out_T)
        outs_A.append(out_A)

        gc.collect()

    # convert shifts to units of microseconds
    for i, shift in enumerate(shifts):
        shifts[i] = shifts[i] * (spec.time_centers[1] - spec.time_centers[0])

    # print out the sensitivity
    time_range = (shifts[-1] - shifts[0])

    T_R_range = outs_T[-1][1].GetParameter(4) - outs_T[0][1].GetParameter(4)
    T_sens = T_R_range / time_range

    A_R_range = outs_A[-1][1].GetParameter(4) - outs_A[0][1].GetParameter(4)
    A_sens = A_R_range / time_range

    print('\nPhase scan summary:')
    print(f'T-Method sensitivity: {T_sens:.3f} ppb / ns')
    print(f'A-Weighted sensitivity: {A_sens:.3f} ppb / ns')

    return outs_T, outs_A, shifts


def fit_pu_shifted_T_and_A(shift_in_bins, spec, fit_info, config, rebin=6):
    _ = r.TCanvas()

    hist_T, hist_A = build_pu_shifted_T_and_A(
        shift_in_bins, spec, fit_info, rebin)

    return fit_T_and_A(hist_T, hist_A, fit_info,
                       config, f'pu_shift_{shift_in_bins}')


def build_pu_shifted_T_and_A(shift_in_bins, spec, fit_info, rebin=6):
    ''' build T-Method and A-Weighted histograms with the pileup spectrum
    shifted by a certain number of bins'''

    uncor_spec = spec.array
    pu_spec = spec.pu_spectrum

    shifted_pu = np.zeros_like(pu_spec)

    if shift_in_bins > 0:
        shifted_pu[:, :, shift_in_bins:] = pu_spec[:, :, :-shift_in_bins]
    elif shift_in_bins < 0:
        shifted_pu[:, :, :shift_in_bins] = pu_spec[:, :, -shift_in_bins:]
    else:
        shifted_pu[:, :, :] = pu_spec[:, :, :]

    shifted_corrected = rebinned_last_axis(uncor_spec - shifted_pu,
                                           rebin)
    rebinned_axes = list(spec.axes)
    rebinned_axes[-1] = spec.axes[-1][::rebin]

    shifted_3d = CaloSpectra.build_root_hist(shifted_corrected,
                                             rebinned_axes,
                                             f'pushifted3d_{shift_in_bins}')

    master_2d = shifted_3d.Project3D(f'yx_ps{shift_in_bins}')
    r.SetOwnership(master_2d, True)

    return T_and_A_from_2d(master_2d, fit_info, f'PUShifted{shift_in_bins}',
                           f'pu shift {shift_in_bins}')

#
# Approximate IFG sweep code
#


def ifg_amplitude_sweep(config, conf_dir):
    ''' run the IFG amplitude sweep '''
    print('\n---\nApproximate IFG amplitude sweep\n---')

    raw_f, fit_info = load_raw_and_analyzed_files(config, conf_dir)

    print('Building corrected, uncorrected, and difference hists...')
    cor = raw_f.Get(config['cor_hist_name'])
    cor_2d = cor.Project3D('yx_1')
    r.SetOwnership(cor_2d, True)
    cor_2d.SetName('IFGCorrected2D')

    try:
        # check if a separate uncorrected raw filename is specified
        uncor_f = r.TFile(f'{conf_dir}/{config["uncor_raw_f_name"]}')
    except KeyError:
        # use the same raw file for the uncorrected hist
        uncor_f = raw_f

    uncor = uncor_f.Get(config['uncor_hist_name'])
    uncor_2d = uncor.Project3D('yx_2')
    r.SetOwnership(uncor_2d, True)
    uncor_2d.SetName('IFGUncorrected2D')

    delta_rho = uncor_2d.Clone()
    delta_rho.SetName('delta_rho')
    r.SetOwnership(delta_rho, True)
    delta_rho.Add(cor_2d, -1)

    # run the scan
    print('running approximate ifg amplitude scan...')
    if config['scale_max'] < config['scale_min']:
        raise RuntimeError(
            'Problem with IFG scan configuration!'
            '"scale_min" must be less than "scale_max"')

    scales = []
    scale = config['scale_min']
    while scale <= config['scale_max']:
        scales.append(scale)
        scale += config['scale_step']

    outs_T = []
    outs_A = []
    for scale in scales:
        print(f'scale {scale:.3f}....')
        out_T, out_A = fit_ifg_scaled_T_and_A(
            scale, cor_2d, delta_rho, fit_info, config)
        outs_T.append(out_T)
        outs_A.append(out_A)

        gc.collect()

    scale_range = scales[-1] - scales[0]
    T_R_range = outs_T[-1][1].GetParameter(4) - outs_T[0][1].GetParameter(4)
    T_sens = T_R_range / scale_range

    A_R_range = outs_A[-1][1].GetParameter(4) - outs_A[0][1].GetParameter(4)
    A_sens = A_R_range / scale_range

    print('\nIFG amplitude scan summary:')
    print(f'T-Method sensitivity: {T_sens*1000:.1f} ppb / IFG correction')
    print(f'A-Weighted sensitivity: {A_sens*1000:.1f} ppb / IFG correction')

    return outs_T, outs_A, scales


def build_ifg_scaled_T_and_A(scale_factor, cor_2d, delta_rho, fit_info):
    scaled_2d = cor_2d.Clone()
    scaled_2d.SetName(f'scaled2D_{scale_factor}')
    scaled_2d.Add(delta_rho, float(1 - scale_factor))

    return T_and_A_from_2d(scaled_2d, fit_info, f'Scaled{scale_factor:.3f}',
                           f'IFG scale {scale_factor}')


def fit_ifg_scaled_T_and_A(scale_factor, cor_2d, delta_rho, fit_info, config):
    _ = r.TCanvas()

    hist_T, hist_A = build_ifg_scaled_T_and_A(
        scale_factor, cor_2d, delta_rho, fit_info)

    return fit_T_and_A(hist_T, hist_A, fit_info,
                       config, f'ifg_scale_{scale_factor}')


#
# Residual gain correction sweep code
#

def residual_gain_amp_scan(config, conf_dir):
    ''' scan the amplitude of the "residual" gain correction '''
    print('\n---\nResidual gain correction amplitude sweep\n---')

    raw_f, fit_info, spec, oma, phi = prepare_residual_correction_scan(
        config, conf_dir)

    # run the scan
    print('running residual gain correction amplitude scan...')
    if config['amp_max'] < config['amp_min']:
        raise RuntimeError(
            'Problem with residual gain scan configuration!'
            '"amp_min" must be less than "amp_max"')

    amps = []
    amp = config['amp_min']
    while amp <= config['amp_max']:
        amps.append(amp)
        amp += config['amp_step']

    ref_amp = config['reference_amp']
    ref_tau = config['reference_tau']
    ref_A = config['reference_asym']

    outs_T = []
    outs_A = []
    for amp in amps:
        print(f'amplitude {amp:.3f}....')

        name = f'residGainAmp{amp:.3f}'

        out_T, out_A = fit_gaincor_T_and_A(spec, amp * ref_amp,
                                           ref_tau, ref_A, oma, phi,
                                           fit_info, name, config)

        outs_T.append(out_T)
        outs_A.append(out_A)

        gc.collect()

    # get some summary information
    amp_range = amps[-1] - amps[0]
    T_R_range = outs_T[-1][1].GetParameter(4) - outs_T[0][1].GetParameter(4)
    T_sens = T_R_range / amp_range

    A_R_range = outs_A[-1][1].GetParameter(4) - outs_A[0][1].GetParameter(4)
    A_sens = A_R_range / amp_range

    print('\nResidual gain amplitude scan summary:')
    print(
        'T-Method residual gain sensitivity:'
        f' {T_sens*1000:.1f} ppb / {ref_amp*1e3:.1f} x 10^-3')
    print(
        'A-Weighted residual gain sensitivity:'
        f' {A_sens*1000:.1f} ppb / {ref_amp*1e3:.1f} x 10^-3')

    return outs_T, outs_A, amps


def residual_gain_tau_scan(amp, config, conf_dir):
    ''' scan the lifetime of the "residual" gain correction '''
    print('\n---\nResidual gain correction lifetime sweep\n---')

    raw_f, fit_info, spec, oma, phi = prepare_residual_correction_scan(
        config, conf_dir)

    # run the scan
    print('running residual gain correction lifetime scan...')
    if config['tau_max'] < config['tau_min']:
        raise RuntimeError(
            'Problem with residual gain scan configuration!'
            '"tau_min" must be less than "tau_max"')

    taus = []
    tau = config['tau_min']
    while tau <= config['tau_max']:
        taus.append(tau)
        tau += config['tau_step']

    ref_A = config['reference_asym']

    outs_T = []
    outs_A = []
    for tau in taus:
        print(f'lifetime {tau:.3f}....')

        name = f'residGainTau{tau:.3f}'

        out_T, out_A = fit_gaincor_T_and_A(spec, amp,
                                           tau, ref_A, oma, phi,
                                           fit_info, name, config)
        outs_T.append(out_T)
        outs_A.append(out_A)

        gc.collect()

    print('\nLifetime scan completed!\n')

    return outs_T, outs_A, taus


def residual_gain_asym_scan(amp, config, conf_dir):
    ''' scan the asymmetry of the "residual" gain correction '''
    print('\n---\nResidual gain correction asymmetry sweep\n---')

    raw_f, fit_info, spec, oma, phi = prepare_residual_correction_scan(
        config, conf_dir)

    # run the scan
    print('running residual gain correction asymmetry scan...')
    if config['asymmetry_max'] < config['asymmetry_min']:
        raise RuntimeError(
            'Problem with residual gain scan configuration!'
            '"asymmetry_min" must be less than "asymmetry_max"')

    asyms = []
    asym = config['asymmetry_min']
    while asym <= config['asymmetry_max']:
        asyms.append(asym)
        asym += config['asymmetry_step']

    ref_tau = config['reference_tau']

    outs_T = []
    outs_A = []
    for asym in asyms:
        print(f'asymmetry {asym:.3f}....')

        name = f'residGainAsym{asym:.3f}'

        out_T, out_A = fit_gaincor_T_and_A(spec, amp,
                                           ref_tau, asym, oma, phi,
                                           fit_info, name, config)

        outs_T.append(out_T)
        outs_A.append(out_A)

        gc.collect()

    # get some summary information
    # range over which to report the sensitivity
    ref_range = 0.1

    asym_range = asyms[-1] - asyms[0]
    T_R_range = outs_T[-1][1].GetParameter(4) - outs_T[0][1].GetParameter(4)
    T_sens = ref_range * T_R_range / asym_range

    A_R_range = outs_A[-1][1].GetParameter(4) - outs_A[0][1].GetParameter(4)
    A_sens = ref_range * A_R_range / asym_range

    print('\nResidual gain asymlitude scan summary:')
    print(
        'T-Method residual gain asymmetry sensitivity:'
        f' {T_sens*1000:.1f} ppb / {ref_range * 100:.1f}%')
    print(
        'A-Weighted residual gain asymmetry sensitivity:'
        f' {A_sens*1000:.1f} ppb / {ref_range * 100:.1f}%')

    print('\nAsymmetry scan completed!\n')

    return outs_T, outs_A, asyms


def residual_gain_phase_scan(amp, config, conf_dir):
    ''' scan the phase of the "residual" gain correction '''
    print('\n---\nResidual gain correction phase sweep\n---')

    raw_f, fit_info, spec, oma, phi = prepare_residual_correction_scan(
        config, conf_dir)

    # run the scan
    print('running residual gain correction phase scan...')

    phases = [0]
    phase = - np.pi
    while phase <= np.pi:
        phases.append(phase)
        phase += np.abs(config['phi_step'])

    phases = sorted(phases)

    ref_tau = config['reference_tau']
    ref_asym = config['reference_asym']

    outs_T = []
    outs_A = []
    for phase in phases:
        print(f'phase {phase:.3f}....')

        name = f'residGainPhase{phase:.3f}'

        out_T, out_A = fit_gaincor_T_and_A(spec, amp,
                                           ref_tau, ref_asym, oma, phi + phase,
                                           fit_info, name, config)

        outs_T.append(out_T)
        outs_A.append(out_A)

        gc.collect()

    print('\n Phase scan completed!\n')

    return outs_T, outs_A, phases


def prepare_residual_correction_scan(config, conf_dir):
    ''' common preparation for all scans involving the
    residual gain correction '''
    raw_f, fit_info = load_raw_and_analyzed_files(config, conf_dir)

    print('Building CaloSpectra...')
    spec = CaloSpectra.from_root_file(fit_info['infile_name'],
                                      fit_info['ana_hist_name'])

    # oscillation frequency
    fit = fit_info['example_T_fit']
    oma = fit.GetParameter(5) * (1 + fit.GetParameter(4) * 1e-6)
    phi = fit.GetParameter(3)

    return raw_f, fit_info, spec, oma, phi


def build_gain_corrected_T_and_A(
        spec, ampl, tau, asym, oma, phi,
        fit_info, name):
    ''' create "gain corrected" T-Method and A-Weighted histograms'''
    corrected = gain_correct_3d(spec, ampl, tau, asym, oma, phi)

    corrected_hist_3d = CaloSpectra.build_root_hist(
        corrected, spec.axes, f'{name}_3d')
    corrected_2d = corrected_hist_3d.Project3D('yx')
    r.SetOwnership(corrected_2d, True)

    return T_and_A_from_2d(corrected_2d, fit_info, name,
                           name)


def gain_correct_3d(spec, ampl, tau, asym, oma, phi):
    ''' apply the (approximate) gain correction/perturbation
    Returns the gain perturbed 3d numpy array '''

    times = spec.time_centers

    # multiply by -1 to undo the gain perturbation,
    # turning the perturbation into a correction
    pert = -1 * ampl * np.exp(-times / tau) * \
        (1 + asym * np.cos(oma * times - phi))

    corrected = np.empty_like(spec.array)
    for i, calo_spec in enumerate(spec.array):
        corrected[i] = CaloSpectra.gain_perturb(
            calo_spec, spec.energy_centers, pert)

    return corrected


def fit_gaincor_T_and_A(spec, ampl, tau, asym, oma, phi,
                        fit_info, name, config):
    _ = r.TCanvas()

    hist_T, hist_A = build_gain_corrected_T_and_A(
        spec, ampl, tau, asym, oma, phi,
        fit_info, name)

    return fit_T_and_A(hist_T, hist_A, fit_info,
                       config, name)

#
# Seed scan code
#


def seed_scan(config, conf_dir):
    print('\n---\nRunning the seed scan\n---')

    raw_f, fit_info = load_raw_and_analyzed_files(config, conf_dir)

    # get expected ctag
    cor_hist_dir = config['cor_hist_name'].split('/')[0]
    ctag_hist = raw_f.Get(f'{cor_hist_dir}/ctag')
    expected_ctag = ctag_hist.GetEntries()

    print('Building 2d pileup spectrum for the seed scan...')
    pu_2d = get_pileup_2d(fit_info['infile_name'], config)

    print('running the seed scan...')

    outs_T = []
    outs_A = []
    seed_num = 0
    for file_name in config['input_files']:
        r_file = r.TFile(f'{conf_dir}/{file_name}')
        dir_names = get_dir_names(r_file)
        for dir_name in dir_names:
            seed_num += 1

            print(f'seed {seed_num}...')
            hist_name = f'{dir_name}/{config["seed_hist_names"]}'

            out_T, out_A = fit_seed_T_and_A(r_file, hist_name, pu_2d,
                                            expected_ctag, seed_num,
                                            config, fit_info)

            outs_T.append(out_T)
            outs_A.append(out_A)

            gc.collect()

    seed_nums = list(range(1, seed_num + 1))

    # print summary info
    rs_T = np.array([out[1].GetParameter(4) for out in outs_T])
    rs_A = np.array([out[1].GetParameter(4) for out in outs_A])

    print('\nSeed scan summary:')
    print(f'{len(seed_nums)} seeds')
    print(f'T-Method RMS: {np.std(rs_T)*1000:.1f} ppb')
    print(f'A-Weighted RMS: {np.std(rs_A)*1000:.1f} ppb')

    return outs_T, outs_A, seed_nums


def get_dir_names(r_file):
    ''' returns a list of the names of the directories in r_file '''
    next_obj = r.TIter(r_file.GetListOfKeys())

    dir_names = []
    while True:
        obj = next_obj()

        if not obj:
            break

        if obj.GetClassName() == 'TDirectoryFile':
            dir_names.append(obj.GetName())

    return dir_names


def get_pileup_2d(filename, conf, rebin=1):
    ''' return the dataset's pu spectrum as a 2D root histogram'''

    cor = CaloSpectra.from_root_file(filename, conf['cor_hist_name'])
    uncor = CaloSpectra.from_root_file(filename, conf['uncor_hist_name'])

    pu_3d = uncor.array - cor.array
    pu_3d = rebinned_last_axis(pu_3d, rebin)

    rebinned_axes = list(cor.axes)
    rebinned_axes[-1] = rebinned_axes[-1][::rebin]

    pu_hist_3d = CaloSpectra.build_root_hist(
        pu_3d, rebinned_axes, 'seedScanPu3D')

    pu_2d = pu_hist_3d.Project3D('yx')
    r.SetOwnership(pu_2d, True)
    pu_2d.SetName('seedScanPU2D')

    return pu_2d


def build_seed_T_and_A(r_file, hist_name, pu_2d,
                       expected_ctag, seed_num, fit_info):
    ''' build the T and A histograms for this random seed '''

    # ensure the CTAG is correct
    dir_name = hist_name.split('/')[0]
    ctag_hist = r_file.Get(f'{dir_name}/ctag')
    n_ctag = ctag_hist.GetEntries()

    if n_ctag != expected_ctag:
        f_name = r_file.GetName()
        raise RuntimeError(
            f'Incorrect ctag value from {f_name}, {hist_name}: {n_ctag}')

    hist_3d = r_file.Get(hist_name)
    uncor_2d = hist_3d.Project3D(f'yx_{seed_num}')
    r.SetOwnership(uncor_2d, True)
    uncor_2d.SetName(f'uncor2d_{seed_num}')

    cor_2d = uncor_2d.Clone()
    r.SetOwnership(cor_2d, True)
    cor_2d.SetName(f'cor2d_{seed_num}')
    cor_2d.Add(pu_2d, -1)

    return T_and_A_from_2d(cor_2d, fit_info,
                           f'Seed{seed_num}', f'seed {seed_num}')


def fit_seed_T_and_A(r_file, hist_name, pu_2d,
                     expected_ctag, seed_num,
                     config, fit_info):
    _ = r.TCanvas()

    ''' build the T and A histograms for a given random seed and fit them '''
    hist_T, hist_A = build_seed_T_and_A(r_file, hist_name, pu_2d,
                                        expected_ctag, seed_num, fit_info)

    return fit_T_and_A(hist_T, hist_A, fit_info, config, f'seed_{seed_num}')

#
# Common code
#


def T_and_A_from_2d(hist_2d, fit_info, name, title):
    T_hist = hist_2d.ProjectionX(f'T{name}',
                                 fit_info['thresh_bin'], -1)
    T_hist.SetTitle(f'T-Method, {title}')

    T_hist.SetDirectory(0)

    A_hist = build_a_weight_hist(hist_2d, fit_info['a_model'],
                                 f'A{name}')
    A_hist.SetTitle(f'A-Weighted, {title}')

    A_hist.SetDirectory(0)

    apply_pu_unc_factors(T_hist, A_hist, fit_info)

    return T_hist, A_hist


def fit_T_and_A(hist_T, hist_A, fit_info, config, name):
    fit_T = configure_model_fit(
        fit_info['example_T_fit'], f'TFit_{name}', config)

    resids_T, fft_T = fit_and_fft(hist_T, fit_T, f'fullFitScale{name}',
                                  config['fit_options'],
                                  fit_info['fit_start'],
                                  fit_info['fit_end'],
                                  double_fit=True)

    fit_A = configure_model_fit(
        fit_info['example_A_fit'], f'AFit_{name}', config)

    resids_A, fft_A = fit_and_fft(hist_A, fit_A,
                                  f'AFitScale{name}',
                                  config['fit_options'],
                                  fit_info['fit_start'],
                                  fit_info['fit_end'],
                                  double_fit=True)

    return (hist_T, fit_T, fft_T), (hist_A, fit_A, fft_A)


def apply_pu_unc_factors(T_hist, A_hist, fit_info):
    ''' apply pileup uncertainty factors to the T and A hists '''
    T_meth_unc_facs = fit_info['T_meth_unc_facs']
    A_weight_unc_facs = fit_info['A_weight_unc_facs']

    # pu uncertainty factors
    if len(T_meth_unc_facs):
        for i_bin in range(1, T_hist.GetNbinsX() + 1):
            #  don't take sqrt of negative bin contents
            if T_hist.GetBinContent(i_bin) > 0:
                old_err = np.sqrt(T_hist.GetBinContent(i_bin))
                new_err = old_err * T_meth_unc_facs[i_bin - 1]
                T_hist.SetBinError(i_bin, new_err)

    if len(A_weight_unc_facs):
        for i_bin in range(1, A_hist.GetNbinsX() + 1):
            old_err = A_hist.GetBinError(i_bin)
            new_err = old_err * A_weight_unc_facs[i_bin - 1]
            A_hist.SetBinError(i_bin, new_err)


def load_pu_uncertainties(pu_unc_file):
    ''' load pileup uncertainties from the text file '''
    if pu_unc_file is not None:
        factor_array = np.loadtxt(pu_unc_file, skiprows=1)
        T_meth_unc_facs = factor_array[:, 1]
        A_weight_unc_facs = factor_array[:, 2]
    else:
        T_meth_unc_facs = []
        A_weight_unc_facs = []

    return T_meth_unc_facs, A_weight_unc_facs


def load_raw_and_analyzed_files(config, conf_dir):
    ''' common to all systematic sweeps:
    load the raw input file and the associated analysis output file'''

    # parse the fit configuration file
    with open(f'{conf_dir}/{config["fit_conf"]}') as file:
        fit_conf = json.load(file)

    # load the raw file
    try:
        rawf_name = f'{conf_dir}/{config["raw_file"]}'
    except KeyError:
        rawf_name = fit_conf['file_name']

    raw_f = r.TFile(rawf_name)

    ana_fname = f'{conf_dir}/../{fit_conf["out_dir"]}/'\
        f'{fit_conf["outfile_name"]}'
    if not ana_fname.endswith('.root'):
        ana_fname += '.root'

    return raw_f, load_fit_conf(fit_conf, r.TFile(ana_fname))


def load_fit_conf(fit_conf, ana_file):
    ''' load example fits, pileup uncertainties,
    threshold bin, a_models, etc from the fit_conf'''
    thresh_bin = fit_conf['thresh_bin']

    example_T_hist = ana_file.Get('T-Method/tMethodHist')
    example_fit = example_T_hist.GetFunction('tMethodFit')

    example_A_hist = ana_file.Get('A-Weighted/aWeightHist')
    example_A_fit = example_A_hist.GetFunction('aWeightFit')
    a_model = ana_file.Get('A-Weighted/aVsESpline')

    fit_start, fit_end = example_fit.GetXmin(), example_fit.GetXmax()

    prepare_loss_hist(fit_conf, example_T_hist)

    T_meth_unc_facs, A_weight_unc_facs = load_pu_uncertainties(
        fit_conf.get('pu_uncertainty_file'))

    output_dict = {
        'thresh_bin': thresh_bin,
        'example_T_hist': example_T_hist,
        'example_T_fit': example_fit,
        'T_meth_unc_facs': T_meth_unc_facs,
        'example_A_hist': example_A_hist,
        'example_A_fit': example_A_fit,
        'A_weight_unc_facs': A_weight_unc_facs,
        'a_model': a_model,
        'fit_start': fit_start,
        'fit_end': fit_end,
        'infile_name': fit_conf['file_name'],
        'ana_hist_name': fit_conf['hist_name'],
        'root_file': ana_file,
    }

    return output_dict


def make_output_dir(out_f, outs_T, outs_A, x_vals, dir_name, sweep_par_name):
    ''' store the output of a systematic scan in the output root file'''
    super_dir = out_f.mkdir(dir_name)

    sub_names = ['T-Method', 'A-Weighted']
    for outs, sub_name in zip([outs_T, outs_A], sub_names):
        this_dir = super_dir.mkdir(sub_name)
        this_dir.cd()

        for hist, fit, _ in outs:
            hist.Write()
            fit.Write()

        chi2_g = r.TGraph()
        par_gs = {}

        chi2_g.SetName('chi2')

        for x_val, (_, fit, _) in zip(x_vals, outs):
            pt_num = chi2_g.GetN()

            chi2_g.SetPoint(pt_num, x_val, fit.GetChisquare())
            update_par_graphs(x_val, fit, par_gs)

        g_dir = this_dir.mkdir('sweepGraphs')
        g_dir.cd()

        chi2_g.GetXaxis().SetTitle(sweep_par_name)
        chi2_g.Write()

        for g_name in par_gs:
            par_gs[g_name].GetXaxis().SetTitle(sweep_par_name)
            par_gs[g_name].Write()


def get_chi2_min(xs, outs):
    ''' returns the x corresponding to minimum chi2
    and the width of the chi2 minimum.
    This only if chi2 vs x is parabolic'''

    xs = np.array(xs)
    chi2s = np.array([out[1].GetChisquare() for out in outs])

    quad_fit = np.polyfit(xs, chi2s, 2)

    min_x_chi2 = -quad_fit[1] / 2 / quad_fit[0]
    width = 1 / np.sqrt(quad_fit[0])

    return min_x_chi2, width


#
# Driver function
#

def run_systematic_sweeps(conf_name):
    with open(conf_name) as file:
        config = json.load(file)

    config_dir = '/'.join(conf_name.split('/')[:-1])

    # pileup phase sweep
    pu_phase_conf = config.get('pileup_phase_scan')
    pu_sweep_out = None
    if pu_phase_conf is not None and pu_phase_conf['run_scan']:
        pu_sweep_out = pileup_phase_sweep(pu_phase_conf, config_dir)
        print('\n---\nPileup phase sweep done\n---\n')

    # ifg amplitude sweep
    ifg_amp_conf = config.get('ifg_amplitude_scan')
    ifg_amp_out = None
    if ifg_amp_conf is not None and ifg_amp_conf['run_scan']:
        ifg_amp_out = ifg_amplitude_sweep(ifg_amp_conf, config_dir)
        print('\n---\nIFG amplitude sweep done\n---\n')

    # residual gain correction scans
    resid_g_conf = config.get('residual_gain_correction')
    resid_g_amp_out = None
    if resid_g_conf is not None and resid_g_conf['run_scan']:
        resid_g_amp_out = residual_gain_amp_scan(resid_g_conf, config_dir)

        # get T-Method and A-Weighted minimum amplitudes
        T_min, T_width = get_chi2_min(resid_g_amp_out[-1], resid_g_amp_out[0])
        print(f'T-Method minimum amp: {T_min:.2f} +/- {T_width:.2f}')
        A_min, A_width = get_chi2_min(resid_g_amp_out[-1], resid_g_amp_out[1])
        print(f'A-Weighted minimum amp: {A_min:.2f} +/- {A_width:.2f}')

        # take more precise A-Weighted minimum for following scans
        # unless force_amplitude is set to true
        ref_amp = resid_g_conf['reference_amp']
        amp = ref_amp if resid_g_conf['force_amplitude'] else A_min * ref_amp

        print(
            f'Using amp = {amp*1e3:.1f} x 10^-3 for the tau, asymmetry scans')

        resid_g_tau_out = residual_gain_tau_scan(amp, resid_g_conf, config_dir)
        resid_g_asym_out = residual_gain_asym_scan(
            amp, resid_g_conf, config_dir)
        resid_g_phase_out = residual_gain_phase_scan(
            amp, resid_g_conf, config_dir)

        print('\n---\nResidual gain sweeps done\n---\n')

    # seed scan
    seed_conf = config.get('seed_scan')
    seed_out = None
    if seed_conf is not None and seed_conf['run_scan']:
        seed_out = seed_scan(seed_conf, config_dir)
        print('\n---\nSeed scan done\n---\n')

    # make output file
    outf_name = config['outfile_name']
    if not outf_name.endswith('.root'):
        outf_name += '.root'

    outf = r.TFile(outf_name, 'recreate')

    print('\n---\nMaking output file\n---\n')
    if pu_sweep_out is not None:
        make_output_dir(outf, *pu_sweep_out, 'pileupPhaseSweep',
                        'pileup time shift [#mus]')

    if ifg_amp_out is not None:
        make_output_dir(outf, *ifg_amp_out, 'ifgAmpSweep',
                        'IFG amplitude multiplier')

    if resid_g_amp_out is not None:
        resid_g_dir = outf.mkdir('residualGainSweeps')
        ref_amp = resid_g_conf['reference_amp']
        make_output_dir(resid_g_dir, *resid_g_amp_out, 'ampSweep',
                        'residual gain amplitude '
                        f'[{ref_amp*1e3:.1f} #times 10^{{-3}}]')
        make_output_dir(resid_g_dir, *resid_g_tau_out, 'tauSweep',
                        'residual gain lifetime [#mus]')
        make_output_dir(resid_g_dir, *resid_g_asym_out, 'asymmetrySweep',
                        'residual gain asymmetry')
        make_output_dir(resid_g_dir, *resid_g_phase_out, 'phaseSweep',
                        'residual gain phase')

    if seed_out is not None:
        make_output_dir(outf, *seed_out, 'seedScan',
                        'seed number')
