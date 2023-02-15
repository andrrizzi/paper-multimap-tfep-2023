#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Various analysis utilities.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import glob
import itertools
import json
import pathlib
import os

import numpy as np
import pandas as pd
import pint
import torch
from scipy import stats
import scipy.special

import tfep.analysis
import tfep.io
import tfep.utils.plumed.io as plumedio


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# The number of repeats used to compute the standard deviation of the free
# energy in the reference calculations. Used to compute the t-statistic.
N_REPEATS_REF = 8

UNIT = pint.UnitRegistry()
KT = (300.0 * UNIT.kelvin * UNIT.molar_gas_constant).to(UNIT.kcal/UNIT.mol).magnitude

# =============================================================================
# UTILS
# =============================================================================

def read_reference_data(ref_calc='CRO'):
    """Read reference free energies.

    ref_calc can be "CRO", "BAR", or "TFEP".

    For TFEP, we use the predictions computed with Z-matrix coordinates, batch
    size 48, and (when available) OPES reference simulation.
    """
    # Read the reference results (which are stored in kcal/mol).
    with open(os.path.join('..', 'hipen', 'reference.json'), 'r') as f:
        reference_data = json.load(f)

    # Load the amber offsets.
    with open(os.path.join('..', 'amber', 'offsets_3ob.json'), 'r') as f:
        amber_offset = json.load(f)
    for mol_id, mol_data in reference_data.items():
        reference_data[mol_id]['AMBER offset'] = amber_offset[mol_id]

    # Read TFEP results.
    for mol_id in reference_data:
        if os.path.exists(os.path.join('tfep', 'zinc_'+mol_id, 'zmatrix-spline6-b48-r11-opes')):
            analysis_file_path = os.path.join('tfep', 'zinc_'+mol_id, 'zmatrix-spline6-b48-r11-opes', 'analysis.json')
        else:
            analysis_file_path = os.path.join('tfep', 'zinc_'+mol_id, 'zmatrix-spline6-b48', 'analysis.json')

        # Load the analysis.
        with open(analysis_file_path, 'r') as f:
            reference_data[mol_id]['TFEP'] = json.load(f)

        # Apply offset between CHARMM and AMBER.
        if ref_calc == 'TFEP':
            # TFEP analysis files already include the AMBER offset.
            reference_data[mol_id]['DA AMBER'] = reference_data[mol_id][ref_calc]['df']['bootstrap'][-1]['mean']
        else:
            with open(os.path.join('..', 'amber', 'offsets_3ob.json'), 'r') as f:
                amber_offset = json.load(f)
            for mol_id in reference_data.keys():
                reference_data[mol_id]['DA AMBER'] = reference_data[mol_id][ref_calc]['DA'] - amber_offset[mol_id]

    return reference_data


def logmeanexp(d, dim=-1):
    n_samples = torch.tensor(d.shape[dim])
    return torch.logsumexp(d - torch.log(n_samples), dim=dim)


def tfep_estimator(work, dim=-1):
    return - logmeanexp(-work, dim=dim)


def rmse(d, vectorized=False):
    if torch.is_tensor(d):
        return torch.sqrt((d**2).mean(dim=-1))
    return np.sqrt((d**2).mean(axis=-1))


def get_subdir_plot_options(subdir_name, label_replace, color_rule, ls_rule=None):
    # Fix label.
    label = subdir_name
    if label_replace is not None:
        for replaced, replacing in label_replace.items():
            label = label.replace(replaced, replacing)

    # Color.
    color = None
    if color_rule is not None:
        for pattern in color_rule:
            if pattern in subdir_name:
                color = color_rule[pattern]
                break

    # Line style.
    ls = '-'
    if ls_rule is not None:
        for pattern in ls_rule:
            if pattern in subdir_name:
                ls = ls_rule[pattern]
                break

    return label, color, ls


# =============================================================================
# ANALYSIS
# =============================================================================

def plot_all_df(subdir_names, n_samples=None, ref_calc='CRO', mol_ids=None,
                y_lim=None, label_replace=None, ls_rule=None, color_rule=None,
                fig_file_suffix=None, groups=None, print_stats=False):
    """Plot the final free energy prediction of all molecules."""
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set_context('paper', font_scale=1.0)
    sns.set_style('whitegrid')

    # fig, ax = plt.subplots(figsize=(13, 6))
    fig, ax = plt.subplots(figsize=(3.33, 2.8))

    # Read reference data.
    reference_data = read_reference_data(ref_calc=ref_calc)

    # Find all molecules IDS for which we have results and order them.
    if mol_ids is None:
        mol_ids = []
        for subdir_path in sorted(glob.glob(os.path.join('fep', 'zinc_*'))):
            # The second folder of subdir_path is e.g. zinc_00140610.
            mol_ids.append(pathlib.Path(subdir_path).parts[1][5:])
    n_mols = len(mol_ids)

    # We slightly shift data points from different subdir_names so that
    # error bars do not overlap and are clearly visible.
    shifts = np.linspace(-.2, .2, len(subdir_names))

    # Read all TFEP results.
    df_errors = {}
    df_errors_ci = {}
    for subdir_name, shift in zip(subdir_names, shifts):
        # df_errors[idx] is the error w.r.t. the reference data for molecule ID mol_ids[idx].
        df_errors[subdir_name] = []
        df_errors_ci[subdir_name] = []

        for mol_id in mol_ids:
            # Load the analysis.
            if subdir_name == 'fep':
                analysis_file_path = os.path.join('fep', 'zinc_'+mol_id, 'unbiased', 'analysis.json')
            elif subdir_name == 'fep-opes':
                analysis_file_path = os.path.join('fep', 'zinc_'+mol_id, 'opes', 'analysis.json')
            else:
                analysis_file_path = os.path.join('tfep', 'zinc_'+mol_id, subdir_name, 'analysis.json')

            # Check if there's the analysis.
            if not os.path.isfile(analysis_file_path):
                df_errors[subdir_name].append(float('nan'))
                df_errors_ci[subdir_name].append([float('nan'), float('nan')])
                continue

            # Read the final free energy trajectory.
            with open(os.path.join(analysis_file_path), 'r') as f:
                data = json.load(f)

            # Check which number of sample to read.
            if n_samples is None:
                bootstrap_idx = -1
            else:
                bootstrap_idx = data['bootstrap_sample_size'].index(n_samples)

            # df = data['df']['true']
            df = data['df']['bootstrap'][bootstrap_idx]['mean']
            # df = data['df']['bootstrap'][bootstrap_idx]['median']
            df_ci_l = data['df']['bootstrap'][bootstrap_idx]['confidence_interval']['low']
            df_ci_h = data['df']['bootstrap'][bootstrap_idx]['confidence_interval']['high']

            # Save the error for plotting.
            df_errors[subdir_name].append(df - reference_data[mol_id]['DA AMBER'])
            df_errors_ci[subdir_name].append([df - df_ci_l, df_ci_h - df])

        # From shape (n_mols, 2) to (2, n_mols).
        df_errors[subdir_name] = np.array(df_errors[subdir_name])
        df_errors_ci[subdir_name] = np.array(df_errors_ci[subdir_name]).T

        # Get the subdir plot options.
        label, color, _ = get_subdir_plot_options(subdir_name, label_replace, color_rule)

        # Plot.
        ax.errorbar(
            np.arange(n_mols)+shift, df_errors[subdir_name], yerr=df_errors_ci[subdir_name], ls='none',
            marker='x', label=label, color=color,
        )

    if print_stats:
        # Group statistics.
        if groups is None:
            groups = subdir_names
            df_errors_group = df_errors
            df_errors_ci_group = df_errors_ci
        else:
            df_errors_group = {g: [] for g in groups}
            df_errors_ci_group = {g: [] for g in groups}
            for subdir_name, dferr in df_errors.items():
                for g in groups:
                    if g in subdir_name:
                        df_errors_group[g].extend(dferr.tolist())
                        df_errors_ci_group[g].extend(df_errors_ci[subdir_name].tolist())
                        break

        # Compute statistics.
        rmses = {}
        rmses_errors = {}
        for g, dferr in df_errors_group.items():
            rmses[g] = np.sqrt(np.mean(np.array(dferr)**2))

            # Error propagation.
            dferrci = np.array(df_errors_ci_group[g])
            rmses_err = dferrci * 2  # squared
            rmses_err = np.sqrt(np.sum(rmses_err**2)) / np.sqrt(len(dferrci))  # mean
            rmses_err /= 2  # root
            rmses_errors[g] = rmses_err

        # Print statistics.
        stats_df = pd.DataFrame({
            'name': groups,
            'rmse': [rmses[g] for g in groups],
            'rmse_err': [rmses_errors[g] for g in groups],
        }).sort_values('rmse', ascending=True)
        pd.set_option('display.max_rows', 1000)
        print(stats_df)
        return

    # Plot the reference with CI.
    t_statistic = stats.t.ppf(0.975, N_REPEATS_REF-1)
    sigmas = np.array([reference_data[mol_id][ref_calc]['sigma DA'] for mol_id in mol_ids])
    ddf_ref = t_statistic * sigmas / np.sqrt(N_REPEATS_REF)
    ax.hlines(y=0, xmin=0, xmax=n_mols-1, ls='--', color='black', label='benchmark')
    ax.fill_between(x=range(n_mols), y1=-ddf_ref, y2=ddf_ref, alpha=0.25, color='black')

    # Fix x labels.
    mol_nums = [reference_data[mol_id]['num'] for mol_id in mol_ids]
    plt.xticks(range(n_mols), mol_nums)

    ax.legend(
        # loc='upper left', bbox_to_anchor=(1.0, 1.0), ncol=1,
        loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=3,
        fontsize='small', fancybox=True,
        labelspacing=0.6, columnspacing=0.8, handletextpad=0.25, handlelength=1.2,
    )
    ax.set_ylabel(r'$\delta f_{\mathrm{MM \to QM}}$ [kcal/mol]')
    ax.set_xlabel('molecule ID')

    if y_lim is not None:
        ax.set_ylim(y_lim)

    fig.tight_layout()

    if fig_file_suffix is None:
        fig.savefig('figures/df-predictions/df_all.pdf')
    else:
        fig.savefig('figures/df-predictions/df_all-' + fig_file_suffix + '.pdf')
    # plt.show()


def plot_all_rmse_traj(subdir_names, mol_ids=None, plot_ci=True,
                       label_replace=None, ls_rule=None, color_rule=None,
                       fig_file_suffix=None):
    """Plot the trajectory of the RMSE errors across all molecules."""
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set_context('paper', font_scale=1.0)
    sns.set_style('whitegrid')

    # Sample size to read.
    bootstrap_sample_sizes = [100, 250, 500, 1000, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 300000, 500000, 1000000]

    # Read reference data.
    reference_data = read_reference_data()

    # Find all molecules IDS for which we have results and order them.
    if mol_ids is None:
        mol_ids = []
        for subdir_path in sorted(glob.glob(os.path.join('fep', 'zinc_*'))):
            # The second folder of subdir_path is e.g. zinc_00140610.
            mol_ids.append(pathlib.Path(subdir_path).parts[1][5:])

    # boot_stats[subdir_name][sample_size_idx] is a dict as the result of tfep.analysis.bootstrap on rmses.
    boot_stats = {n: [] for n in subdir_names}

    # Compute all statistics for each sample size and molecule.
    for subdir_name in subdir_names:
        for sample_size in bootstrap_sample_sizes:
            # df_errs[mol_idx] is the error of the free energy (in kcal/mol).
            df_errs = []

            for mol_id in mol_ids:
                # Load the analysis.
                if subdir_name == 'fep':
                    analysis_file_path = os.path.join('fep', 'zinc_'+mol_id, 'unbiased', 'analysis.json')
                else:
                    analysis_file_path = os.path.join('tfep', 'zinc_'+mol_id, subdir_name, 'analysis.json')
                with open(os.path.join(analysis_file_path), 'r') as f:
                    data = json.load(f)

                # Compute error.
                bootstrap_idx = data['bootstrap_sample_size'].index(sample_size)
                df = data['df']['bootstrap'][bootstrap_idx]['mean']
                df_errs.append(df - reference_data[mol_id]['DA AMBER'])

            # Compute the RMSE for across molecules for this sample sizes.
            boot_stats[subdir_name].append(tfep.analysis.bootstrap(
                data=torch.tensor(df_errs),
                statistic=rmse,
                n_resamples=1000,
            ))

    # Plot all trajectories.
    fig, ax = plt.subplots(figsize=(3.33, 2.8))

    # Plot reference.
    ax.hlines(y=0, xmin=bootstrap_sample_sizes[0], xmax=bootstrap_sample_sizes[-1],
              ls='--', color='black', label='benchmark')

    for subdir_name, subdir_stats in boot_stats.items():
        # Get the subdir plot options.
        label, color, ls = get_subdir_plot_options(subdir_name, label_replace, color_rule, ls_rule)

        # Plot mean and CI.
        mean_stats = [stat['mean'] for stat in subdir_stats]
        ax.plot(bootstrap_sample_sizes, mean_stats, label=label, ls=ls, color=color)

        if plot_ci is True or (isinstance(plot_ci, set) and subdir_name in plot_ci):
            ci_l = [stat['confidence_interval']['low'] for stat in subdir_stats]
            ci_h = [stat['confidence_interval']['high'] for stat in subdir_stats]

            # If I pass color=None, it's always set to C0 rather than cycling.
            if color is None:
                kwargs = {}
            else:
                kwargs = {'color': color}
            ax.fill_between(bootstrap_sample_sizes, ci_l, ci_h, alpha=0.25, **kwargs)

    # Compact legend.
    ax.legend(loc='lower left', bbox_to_anchor=(-0.05, 1.0), ncol=3,
              # loc='upper right', ncol=2,
              fontsize='small', fancybox=True,
              labelspacing=0.6, columnspacing=0.8,
              handletextpad=0.25, handlelength=1.2)

    # Configure axes.
    ax.set_ylabel(r'$\Delta f_{\mathrm{MM \to QM}}$ RMSE  [kcal/mol]')
    ax.set_xlabel('number of QM calculations')
    ax.set_xscale('log')
    ax.set_xlim((bootstrap_sample_sizes[0], bootstrap_sample_sizes[-1]))
    ax.set_ylim((-0.5, 10.0))

    fig.tight_layout()
    if fig_file_suffix is None:
        fig.savefig('figures/rmse-traj/rmse.pdf')
    else:
        fig.savefig('figures/rmse-traj/rmse-' + fig_file_suffix + '.pdf')
    # plt.show()


def plot_avg_potential_repeats(mol_ids_good, mol_ids_bad, mol_ids_ugly, mol_ids_opes=None,
                               randomized_double_bond=False, file_name_suffix=''):
    """Returns the data in the TFEP cache and the reference potential energies.

    All values are returned in the same units as they were stored.
    """
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set_context('paper', font_scale=1.0)
    sns.set_style('whitegrid')
    if randomized_double_bond:
        figsize = (3.33, 2.8)
    else:
        figsize = (4., 2.8)
    fig, ax = plt.subplots(figsize=figsize)

    ene_dir_path = '../hipen/out/energy'
    random_double_bond_ene_dir_path = '../hipen/out-double-bond-rand/energy'
    ene_col_idx = 1
    n_repeats = 10

    # Handle mutable defaults.
    if mol_ids_opes is None:
        mol_ids_opes = []

    # Append 'o' to the OPES entries to distinguish them from the unbiased ones.
    mol_ids_opes = [n+'o' for n in mol_ids_opes]

    # Merge molecule IDS.
    mol_ids = sorted(mol_ids_good + mol_ids_bad + mol_ids_ugly + mol_ids_opes)

    # Transform as set for the color.
    mol_ids_good = set(mol_ids_good)
    mol_ids_bad = set(mol_ids_bad)
    mol_ids_ugly = set(mol_ids_ugly)

    # Read reference data.
    reference_data = read_reference_data(ref_calc='CRO')

    # Read potentials.
    ref_potentials_table = {
        'molecule ID': [],
        'repeat': [],
        '$\delta$ potential  [kcal/mol]': [],
        'set': []
    }
    for mol_id in mol_ids:
        if mol_id in mol_ids_good:
            hipen_set = 'good'
        elif mol_id in mol_ids_bad:
            hipen_set = 'bad'
        elif mol_id in mol_ids_ugly:
            hipen_set = 'ugly'
        else:
            hipen_set = 'bad OPES'

        # Determine molecule ID and read the reference energies..
        if mol_id in mol_ids_opes:
            # Remove the final 'o' from mol_id.
            mol_num = str(reference_data[mol_id[:-1]]['num']) + 'o'

            # Split the 100ns-long OPES simulation in 10 blocks.
            # We didn't run any simulation with a randomized double bond.
            colvar_file_path = os.path.join('..', 'hipen', 'out-opes', 'colvar', f'colvar.{mol_id[:-1]}.11.data')
            colvar_data = plumedio.read_table(colvar_file_path, col_names=['opes.bias', 'ene'], as_array=True)
            bias, ref_potentials = colvar_data.T
            weights = scipy.special.softmax(bias / KT).reshape((10, len(ref_potentials)//10))
            ref_potentials_means = np.average(ref_potentials.reshape((10, len(ref_potentials)//10)), axis=1, weights=weights)
        else:
            mol_num = str(reference_data[mol_id]['num'])

            # Check if we need to plot the averge potentials for the randomized double bonds.
            if randomized_double_bond and os.path.exists(os.path.join(random_double_bond_ene_dir_path, f'energy.{mol_id}.mm.mm.1.ene')):
                ene_file_paths = [os.path.join(random_double_bond_ene_dir_path, f'energy.{mol_id}.mm.mm.{i}.ene') for i in range(1, n_repeats+1)]
            else:
                ene_file_paths = [os.path.join(ene_dir_path, f'energy.{mol_id}.mm.mm.{i}.ene') for i in range(1, n_repeats+1)]

            # Compute average potentials.
            ref_potentials_means = []
            for ene_file_path in ene_file_paths:
                ref_potentials_means.append(np.loadtxt(ene_file_path, skiprows=1, usecols=ene_col_idx).mean())
            ref_potentials_means = np.array(ref_potentials_means)
        ref_potentials_tot_mean = ref_potentials_means.mean()

        for repeat_idx, mean in enumerate(ref_potentials_means):
            ref_potentials_table['molecule ID'].append(mol_num)
            ref_potentials_table['repeat'].append(repeat_idx)
            ref_potentials_table['$\delta$ potential  [kcal/mol]'].append(mean - ref_potentials_tot_mean)
            ref_potentials_table['set'].append(hipen_set)

    # Plot.
    ref_potentials_table = pd.DataFrame(ref_potentials_table)
    sns.stripplot(data=ref_potentials_table, x='molecule ID', y='$\delta$ potential  [kcal/mol]',
                  hue='set', jitter=False, linewidth=.8, ax=ax,
                  palette={'good': 'C2', 'bad': 'C1', 'ugly': 'C3', 'bad OPES': 'C0'})

    # Configure axes.
    ax.set_ylim((-6, 6))

    # Legend.
    if randomized_double_bond:
        ncol = 1
    else:
        ncol = 2
    ax.legend(loc='upper center', fontsize='small', fancybox=True, ncol=ncol,
              labelspacing=0.6, columnspacing=0.8,
              handletextpad=0.25, handlelength=1.2)

    fig.tight_layout()
    fig.savefig('figures/ref-potentials/potentials' + file_name_suffix + '.pdf')


def compute_tfep_repeats(mol_ids, subdir_name):
    """Compute the free energy difference of the separate repeats."""
    from run_tfep import read_data, generate_tfep_estimator

    n_repeats = 10
    is_opes = 'opes' in subdir_name
    all_dfs = {mol_id: [] for mol_id in mol_ids}

    # Check if this is a particular repeat.
    subdir_name_split = subdir_name.split('-')
    if len(subdir_name_split) > 3 and subdir_name_split[3][0] == 'r':
        repeat_num = int(subdir_name_split[3][1:])
    else:
        repeat_num = None

    for mol_id in mol_ids:
        print('Processing', mol_id, flush=True)

        # Read the potentials and frame indices.
        mol_name = 'zinc_' + mol_id
        save_dir_path = os.path.join('tfep', mol_name, subdir_name)

        # Read data.
        data = read_data(standard_fep=False, mol_id=mol_id, data_dir_path=save_dir_path, is_opes=is_opes, repeat_num=repeat_num)

        # All energies in the tfepcache are stored in units of kT.
        # The reference potentials are stored in kcal/mol.
        target_potentials = data['potential'] * KT
        ref_potentials = data['ref_potential']
        mapped_potentials = target_potentials - data['log_det_J'] * KT

        # Generate a tfep estimator in units of kcal/mol.
        tfep_estimator = generate_tfep_estimator(KT)

        # Compute the generalized work values.
        work = mapped_potentials - ref_potentials

        # The bias is stored in kcal/mol.
        try:
            tfep_data = torch.vstack([work, data['opes.bias']]).T
        except KeyError:  # Unbiased simulation.
            tfep_data = work

        # Divide the work values by repeat.
        for repeat_idx in range(n_repeats):
            repeat_frame_indices = ((repeat_idx*100000 <= data['dataset_sample_index']) &
                                    (data['dataset_sample_index'] < (repeat_idx+1)*100000))
            repeat_tfep_data = tfep_data[repeat_frame_indices]
            all_dfs[mol_id].append(tfep_estimator(repeat_tfep_data).tolist())

    # Save result.
    with open(os.path.join('tfep/df-repeats-' + subdir_name + '.json'), 'w') as f:
        json.dump(all_dfs, f)


def plot_tfep_repeats(mol_ids_good, mol_ids_bad, mol_ids_ugly, subdir_name, mol_ids_opes=None):
    """Returns the data in the TFEP cache and the reference potential energies.

    All values are returned in the same units as they were stored.
    """
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set_context('paper', font_scale=1.0)
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(4., 2.8))

    # Handle mutable defaults.
    if mol_ids_opes is None:
        mol_ids_opes = []

    # Append 'o' to the OPES entries to distinguish them from the unbiased ones.
    mol_ids_opes = [n+'o' for n in mol_ids_opes]

    # Merge molecule IDS.
    mol_ids = sorted(mol_ids_good + mol_ids_bad + mol_ids_ugly + mol_ids_opes)

    # Read reference free energies.
    reference_data = read_reference_data(ref_calc='TFEP')

    # Transform as set for the color.
    mol_ids_good = set(mol_ids_good)
    mol_ids_bad = set(mol_ids_bad)

    # Unbiased dynamics calculations.
    with open(os.path.join('tfep/df-repeats-' + subdir_name + '.json'), 'r') as f:
        dfs = json.load(f)

    # OPES calculations.
    with open(os.path.join('tfep/df-repeats-' + subdir_name + '-r11-opes.json'), 'r') as f:
        opes_dfs = json.load(f)
        for k, v in opes_dfs.items():
            dfs[k+'o'] = v


    # Format as table for seaborn plotting.
    dfs_table = {
        'molecule ID': [],
        'repeat': [],
        'delta_f': [],
        'set': []
    }
    for mol_id in mol_ids:
        # Determine molecule ID and reference free energy.
        if mol_id in mol_ids_opes:
            # Remove the final 'o' from mol_id.
            mol_num = str(reference_data[mol_id[:-1]]['num']) + 'o'
            ref_df = reference_data[mol_id[:-1]]['DA AMBER']

            hipen_set = 'bad OPES'
        else:
            mol_num = str(reference_data[mol_id]['num'])
            ref_df = reference_data[mol_id]['DA AMBER']

            if mol_id in mol_ids_good:
                hipen_set = 'good'
            elif mol_id in mol_ids_bad:
                hipen_set = 'bad'
            else:
                hipen_set = 'ugly'

        # Build table to convert to Pandas Dataframe.
        for repeat_idx, repeat_df in enumerate(dfs[mol_id]):
            dfs_table['molecule ID'].append(mol_num)
            dfs_table['repeat'].append(repeat_idx)
            dfs_table['delta_f'].append(repeat_df - ref_df)
            dfs_table['set'].append(hipen_set)

    # Plot.
    dfs_table = pd.DataFrame(dfs_table)
    sns.stripplot(data=dfs_table, x='molecule ID', y='delta_f',
                  hue='set', jitter=False, linewidth=.8, ax=ax,
                  palette={'good': 'C2', 'bad': 'C1', 'ugly': 'C3', 'bad OPES': 'C0'})

    # Configure axes.
    ax.set_ylabel(r'$\delta f_{\mathrm{MM \to QM}}$  [kcal/mol]')
    ax.set_ylim((-4, 9))
    ax.legend(loc='upper center', fontsize='small', fancybox=True, ncol=2,
              labelspacing=0.6, columnspacing=0.8,
              handletextpad=0.25, handlelength=1.2)
    fig.tight_layout()
    fig.savefig('figures/df-predictions/df-repeats.pdf')


def plot_dihedrals(mol_id, fig_subdir_name, opes=False, step=100, repeat_num=None, from_colvar=False):
    """Plot the dihedral distribution and trajectory."""
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set_context('paper', font_scale=1.1)
    sns.set_style('whitegrid')

    # Last row has number of iteration ***** which is not compatible with loadtxt.
    def fix_iter(val):
        if val == b'*****':
            return 100000.
        else:
            return float(val)

    # Read reference data.
    reference_data = read_reference_data(ref_calc='CRO')

    # The output for unbiased and OPES folders are different.
    if opes:
        out_dir_name = 'out-opes'
        first_frame_idx = 50000
    else:
        out_dir_name = 'out'
        first_frame_idx = 0

    # Determine the repeats.
    if repeat_num is None:
        repeat_num = list(range(1, 11))
    elif not isinstance(repeat_num, list):
        repeat_num = [repeat_num]
    n_repeats = len(repeat_num)

    # Read the dihedrals for each repeat.
    data = None
    for replicate_idx, replicate_num in enumerate(repeat_num):
        # Read data.
        if from_colvar:
            # With the long OPES simulations, there are too many frames for the
            # CHARMM dihedral analysis, but we have saved the dihedrals of the
            # entire simulation directly in the PLUMED output file.
            file_path = os.path.join('..', 'hipen', 'out-opes', 'colvar', f'colvar.{mol_id}.{replicate_num}.data')
            fields = plumedio.read_table_field_names(file_path)
            fields = [k for k in fields if 'chi' in k]
            repeat_data = plumedio.read_table(file_path, col_names=fields, as_array=True)

            # Convert from radian to degrees.
            repeat_data *= 180 / np.pi
        else:
            file_name = 'dihe.' + mol_id + '.mm.' + str(replicate_num) + '.dat'
            repeat_data = np.loadtxt(
                os.path.join('..', 'hipen', out_dir_name, 'dihe', file_name),
                skiprows=14,  # Skip header.
                converters={0: fix_iter},
            )

            # Remove first column which is the row number.
            repeat_data = repeat_data[:, 1:]

        # Subsample.
        repeat_data = repeat_data[first_frame_idx::step]
        n_frames = repeat_data.shape[0]

        # Initialize complete data.
        if data is None:
            data = np.empty((n_frames*n_repeats, repeat_data.shape[1]))

        # Add a column with the repeat number for the hue in the histplot.
        start_idx = replicate_idx * n_frames
        end_idx = start_idx + n_frames
        data[start_idx:end_idx] = repeat_data

    # Transform to pandas dataframe for seaborn plotting.
    # We need to flatten the angle values so that we can use the 'chi' as 'col' in displot.
    n_dihedrals = data.shape[1]
    data_pd = {
        'replicate': np.repeat(np.array(repeat_num), n_frames*n_dihedrals),
        'chi': np.tile(np.arange(n_dihedrals)+1, n_frames*n_repeats),
        'dihedral [degree]': data.flatten(),
    }

    # Prepend/append the angle +- 360 so that we force the displot to be periodic.
    data_pd['replicate'] = np.tile(data_pd['replicate'], 3)
    data_pd['chi'] = np.tile(data_pd['chi'], 3)
    data_pd['dihedral [degree]'] = np.concatenate([data_pd['dihedral [degree]']-360.,
                                                   data_pd['dihedral [degree]'],
                                                   data_pd['dihedral [degree]']+360.])
    data_pd = pd.DataFrame(data_pd)

    # Plot distributions.
    facet_grid = sns.displot(data=data_pd, x='dihedral [degree]', hue='replicate', col='chi',
                             kind='kde', legend=True, palette='tab10',
                             col_wrap=min(4, n_dihedrals), height=2.4, aspect=7.25/2.4/min(4, n_dihedrals),
                             bw_adjust=0.2, clip=(-180., 180.))

    # Fix titles.
    for i, ax in enumerate(facet_grid.axes):
        ax.set_title('$\chi_' + str(i+1) + '$')
        ax.set_xlim([-180, 180])
        ax.set_xticks([-180, -90, 0, 90, 180])

    # These dihedral distributions molecule are shown in the main manuscript.
    mol_num = reference_data[mol_id]['num']
    if mol_num == 14:
        dihedral_values = np.array([47, -71, -77, -73])
        for ls, vals in zip(['-', '--'], [dihedral_values, -dihedral_values]):
            for ax, val in zip(facet_grid.axes, vals):
                ax.vlines(val, 0, 0.00025, color='black', ls=ls)
                ax.set_ylim([-0.000001, 0.00025])
    else:
        # Set title for the supporting information figures.
        facet_grid.fig.suptitle('molecule ' + str(reference_data[mol_id]['num']))

    facet_grid.tight_layout()

    # Create directory.
    dir_path = os.path.join('figures', 'dihedrals', fig_subdir_name)
    os.makedirs(dir_path, exist_ok=True)
    facet_grid.savefig(os.path.join(dir_path, f'dihe-{mol_num}.pdf'))
    # plt.show()


def plot_free_energy_trajectories(mol_ids, subdir_names, ref_calc='CRO', label_replace=None):
    """Plot free energy as a function of the number of samples from the bootstrap analysis.

    If multiple mol_ids and subdir_names are provided,, they are represented with
    differen colors/line styles respectively.

    Parameters
    ----------
    mol_ids : str or List[str]
        The molecules to plot (e.g., '00095858').
    subdir_names : str or List[str]
        The name of the analysis subdirectory (e.g., 'zmatrix-spline6-b48').

    """
    from matplotlib import pyplot as plt
    import seaborn as sns

    # Mutable defaults.
    if label_replace is None:
        label_replace = {}

    # Color/line style palettes.
    color_palette = ['C'+str(i) for i in range(10)]
    ls_palette = ['-', '--', '-.', ':']

    # Make sure inputs are lists.
    if not isinstance(mol_ids, list):
        mol_ids = [mol_ids]
    if not isinstance(subdir_names, list):
        subdir_names = [mol_ids]

    sns.set_context('paper', font_scale=1.0)
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(3.33, 2.8))

    # Read reference data.
    reference_data = read_reference_data(ref_calc='TFEP')

    # Plot the reference line
    ax.hlines(0, 100, 1000000, ls='--', color='black')

    # Plot everything.
    for mol_id, subdir_name in itertools.product(mol_ids, subdir_names):
        # If there are multiple mol IDs we assign different colors to mol IDs
        # and different ls to subdirs. Otherwise, different colors to subdirs.
        if len(mol_ids) > 1:
            color = color_palette[mol_ids.index(mol_id)]
            ls = ls_palette[subdir_names.index(subdir_name)]
            label = 'mol. {} '.format(reference_data[mol_id]['num'])
        else:
            color = color_palette[subdir_names.index(subdir_name)]
            ls = ls_palette[0]
            label = ''
        try:
            label += label_replace[subdir_name]
        except KeyError:
            label += subdir_name

        if subdir_name == 'fep':
            analysis_file_path = os.path.join('fep', 'zinc_'+mol_id, 'unbiased', 'analysis.json')
        elif subdir_name == 'fep-opes':
            analysis_file_path = os.path.join('fep', 'zinc_'+mol_id, 'opes', 'analysis.json')
        else:
            analysis_file_path = os.path.join('tfep', 'zinc_'+mol_id, subdir_name, 'analysis.json')

        # Load the analysis.
        with open(analysis_file_path, 'r') as f:
            data = json.load(f)

        # Compute the free energy trajectory and the CI.
        sample_size = data['bootstrap_sample_size']
        df = np.array([res['mean'] for res in data['df']['bootstrap']])
        df_ci_l = np.array([res['confidence_interval']['low'] for res in data['df']['bootstrap']])
        df_ci_h = np.array([res['confidence_interval']['high'] for res in data['df']['bootstrap']])

        # Plot free energy difference w.r.t. the reference.
        df_ref = reference_data[mol_id]['DA AMBER']
        df -= df_ref
        df_ci_l -= df_ref
        df_ci_h -= df_ref

        # Plot the free energy trajectory
        ax.plot(sample_size, df, label=label, color=color, ls=ls)
        ax.fill_between(sample_size, df_ci_l, df_ci_h, color=color, alpha=0.25)

    ax.legend(loc='upper right', fontsize='small', fancybox=True, ncol=2,
              labelspacing=0.6, columnspacing=0.8,
              handletextpad=0.25, handlelength=1.2)
    ax.set_xlabel('number of QM calculations')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\delta f_{\mathrm{MM \to QM}}$ [kcal/mol]')

    ax.set_xlim((100, 1000000))

    fig.tight_layout()
    fig.savefig(f'figures/df-predictions/df_traj.pdf')
    # plt.show()


def extract_symmetric_mol14():
    """Create a PDB with two symmetric configurations for molecule 14."""
    import MDAnalysis as mda
    from modules import maps

    in_file_base = '../hipen/coors/zinc_00167648'
    out_file_path = 'figures/symmetry/mol14.pdb'

    # Indices of the dihedrals chi1, chi2, and chi3.
    dihedrals_indices = [17, 0, 5, 3, 36]
    # There is a shift of 180 between the angles defined for the Z-matrix and the angles defined for CHARMM.
    dihedrals_values1 = np.array([47., -71., -77., -73., 90.]) + 180
    dihedrals_values2 = -dihedrals_values1

    # Read input.
    universe = mda.Universe(in_file_base + '.psf', in_file_base + '.crd')

    # Create objects performing the transformation from and to Z-matrix.
    tfep_map = maps.zmatrix.TFEPMap(universe=universe)
    cart2ic_flow = tfep_map._build_cart2ic_flow()

    # Print dihedrals.
    for dihedral_idx in dihedrals_indices:
        print(tfep_map.z_matrix[dihedral_idx+3] + 1)

    def _write_config(dihedrals_values):
        if dihedrals_values is not None:
            for dihedral_idx, dihedral_val in zip(dihedrals_indices, dihedrals_values):
                # The first dihedral is at index 5 and there's a new dihedral every 3.
                ic_idx = 5 + dihedral_idx*3
                # Torsion angles are normalized between [0, 1] representing [0, 2pi].
                ic[0, ic_idx] = dihedral_val / 360.
            y, _ = cart2ic_flow.inverse(ic, x0, R)
            ts.positions = y[0].reshape(-1, 3).detach().numpy()
        writer.write(universe.atoms)

    # Write to pdb.
    with mda.Writer(out_file_path, multiframe=True) as writer:
        for ts in universe.trajectory:
            # We need the batch dimension.
            x = torch.from_numpy(ts.positions).flatten().unsqueeze(0)
            ic, _, x0, R = cart2ic_flow(x)

            # Write symmetric configurations.
            _write_config(dihedrals_values1)
            _write_config(dihedrals_values2)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    color_rule = {}
    for i, pattern in enumerate(['fep', '48', '96', '192', '384', '576', '768']):
        color_rule[pattern] = 'C' + str(i)

    mol_ids = ['00077329', '00079729', '00086442', '00107550', '00133435',
               '00138607', '00140610', '00164361', '00167648', '00169358',
               '01867000', '06568023']
    mol_ids_bad = ['00061095', '00087557', '00095858', '01755198', '03127671',
                   '04344392', '33381936', '00107778', '00123162', '04363792']

    # -------------------------------------------- #
    # Final free energy estimates of all molecules #
    # -------------------------------------------- #
    # LaTex table with all results (Table 1 of the manuscript).
    reference_data = read_reference_data(ref_calc='TFEP')
    for mol_id, mol_data in reference_data.items():
        mol_df = '{:.2f} [{:.2f}, {:.2f}]'.format(
            mol_data['TFEP']['df']['bootstrap'][-1]['mean'] + mol_data['AMBER offset'],
            mol_data['TFEP']['df']['bootstrap'][-1]['confidence_interval']['low'] + mol_data['AMBER offset'],
            mol_data['TFEP']['df']['bootstrap'][-1]['confidence_interval']['high'] + mol_data['AMBER offset'],
        )
        print(mol_data['num'], '&', mol_id, '&', mol_df, r'\\')
        print('\hline')


    # Plot final Df prediction for all repeats with different batch sizes (Figure 3 of the manuscript).
    for flow in ['zmatrix-spline6']:
        subdir_names = ['fep']
        for b in ['48', '96', '192', '384', '576', '768']:
            subdir_names.append(flow + '-b' + b)

        plot_all_df(
            subdir_names=subdir_names,
            color_rule=color_rule,
            label_replace={'fep': 'FEP', 'zmatrix': 'Z-matrix', 'global': 'Cartesian', '-spline6-b': ' '},
            # print_stats=True,
            mol_ids=mol_ids,
            y_lim=(-2.5, 8.5),
            fig_file_suffix='good',
        )

    # --------------- #
    # RMSE trajectory #
    # --------------- #
    # Plot RMSE as function of number of QM evaluations (Figure 3 of the manuscript).
    subdir_names = ['fep', 'global-spline6-b48']
    for b in ['48', '96', '192', '384', '576', '768']:
        subdir_names.append('zmatrix-spline6-b' + b)

    plot_all_rmse_traj(
        subdir_names=subdir_names, mol_ids=mol_ids, plot_ci={'zmatrix-spline6-b48', 'zmatrix-spline6-b768', 'fep'},
        label_replace={'fep': 'FEP', 'zmatrix': 'Z-matrix', 'global': 'Cartesian', '-spline6-b': ' '},
        ls_rule={'global': '--'},
        color_rule=color_rule,
    )

    # Comparison Z-matrix and Cartesian RMSES for supporting information.
    for b in ['48', '96', '192', '384', '576', '768']:
        subdir_names = ['zmatrix-spline6-b' + b, 'global-spline6-b' + b]
        plot_all_rmse_traj(
            subdir_names=subdir_names, mol_ids=mol_ids, plot_ci=set(subdir_names),
            label_replace={'fep': 'FEP', 'zmatrix': 'Z-matrix', 'global': 'Cartesian', '-spline6-b': ' '},
            ls_rule={'global': '--'},
            fig_file_suffix=b,
        )

    # --------------------------------- #
    # Average potentials of the repeats #
    # --------------------------------- #
    # Average potential energy (Supporting information)
    plot_avg_potential_repeats(mol_ids_good=mol_ids, mol_ids_bad=mol_ids_bad[:-3],
                               mol_ids_ugly=mol_ids_bad[-3:], mol_ids_opes=['00061095', '00095858'])
    plot_avg_potential_repeats(mol_ids_good=[], mol_ids_bad=mol_ids_bad[:-3],
                               mol_ids_ugly=mol_ids_bad[-3:], randomized_double_bond=True,
                               file_name_suffix='-random-double-bonds')

    # ---------------------------- #
    # Free energies of the repeats #
    # ---------------------------- #
    # Compute the free energy predictions for each individual repeat for the unbiased and OPES dynamics.
    compute_tfep_repeats(mol_ids=mol_ids + mol_ids_bad, subdir_name='zmatrix-spline6-b48')
    compute_tfep_repeats(mol_ids=['00061095', '00095858'], subdir_name='zmatrix-spline6-b48-r11-opes')

    # Plot all Df predictions for each repeat (Figure 4 of the manuscript).
    plot_tfep_repeats(mol_ids_good=mol_ids, mol_ids_bad=mol_ids_bad[:-3], mol_ids_ugly=mol_ids_bad[-3:],
                      subdir_name='zmatrix-spline6-b48', mol_ids_opes=['00061095', '00095858'])

    # -------------------------------- #
    # Dihedral distribution/trajectory #
    # -------------------------------- #
    # Plot all dihedral distributions (Figure 4 and SI of the manuscript).
    for mol_id in mol_ids:
        # Molecules without randomized dihedrals.
        if mol_id in {'00086442', '00140610', '00169358'}:
            continue
        plot_dihedrals(mol_id, fig_subdir_name='dihe-good')
    for mol_id in mol_ids_bad:
        plot_dihedrals(mol_id, fig_subdir_name='dihe-bad')
    for mol_id in ['00061095', '00095858']:
        plot_dihedrals(mol_id=mol_id, fig_subdir_name='dihe-bad-opes', opes=True)
    for mol_id in ['00061095', '00095858']:
        plot_dihedrals(mol_id=mol_id, fig_subdir_name='dihe-bad-opes', opes=True, repeat_num=[11], from_colvar=True, step=1000)

    # ---------------------- #
    # Free energy trajectory #
    # ---------------------- #
    # Free energy trajectory FEP vs OPES+TFEP (Figure 4 of the manuscript).
    plot_free_energy_trajectories(mol_ids=['00061095', '00095858'],
                                  subdir_names=['zmatrix-spline6-b48-r11-opes', 'fep-opes'],
                                  label_replace={'fep-opes': 'FEP', 'zmatrix-spline6-b48-r11-opes': 'TFEP'})

    # -------------------------------- #
    # Extract symmetric configurations #
    # -------------------------------- #
    # Create PDB file to generate Figure 5 of the manuscript.
    extract_symmetric_mol14()
