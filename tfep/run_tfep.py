#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Train the normalizing flow potential.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import json
import logging
import multiprocessing
import os
import time

import ase.calculators.amber
import MDAnalysis
import numpy as np
import pint
import scipy.stats
import torch

import tfep.analysis
import tfep.io
import tfep.potentials.ase
from tfep.utils.parallel import ProcessPoolStrategy
from tfep.utils.plumed.dataset import PLUMEDDataset

from modules import maps
from modules.trainer import TFEPTrainer


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

UNIT = pint.UnitRegistry()

# Number of CHARMM MM simulation repeats.
N_REPEATS = 10

# Paths to AMBER input files.
SCRIPT_DIR_PATH = os.path.dirname(__file__)
HIPEN_DIR_PATH = os.path.realpath(os.path.join('..', 'hipen'))
HIPEN_OUT_DIR_PATH = os.path.join(HIPEN_DIR_PATH, 'out')
HIPEN_OUT_OPES_DIR_PATH = os.path.join(HIPEN_DIR_PATH, 'out-opes')
AMBER_DIR_PATH = os.path.realpath(os.path.join(SCRIPT_DIR_PATH, '..', 'amber'))
PRMTOP_DIR_PATH = os.path.join(AMBER_DIR_PATH, 'prmtop')

# Configure logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thermodynamic variable of the simulation.
TEMPERATURE = 300.0 * UNIT.kelvin
KT = TEMPERATURE * UNIT.molar_gas_constant


# =============================================================================
# ANALYSIS UTILS
# =============================================================================

def generate_tfep_estimator(kT):
    """Generate a TFEP estimator with a fixed kT.

    The function takes as parameter the value of kT for the given unit of energy
    and temperature and returns another function compatible with
    ``tfep.analysis.bootstrap`` taking work values (and optionally the bias)
    as input and outputing the free energy difference, both in the same energy
    units as kT.

    The returned function takes as input a ``torch.Tensor`` called ``data`` of
    shape ``(n_samples,)`` or ``(n_samples, 2)``. In the first case, ``data[i]``
    is the work value for sample ``i``. In the latter, ``data[i][0]`` and ``data[i][1]``
    must be the work and the bias (both in the same units as ``kT``) of the
    ``i``-th sample, respectively.

    The returned function also takes a ``vectorized`` keyword. If ``True``,
    ``data`` must have shape an extra ``batch_size`` dimension prepended and the
    return value will have shape ``(batch_size,)``.
    """
    def _tfep_estimator(data, vectorized=False):
        # Separate work and bias.
        if vectorized:
            if len(data.shape) == 2:
                work, bias = data, None
            else:
                work, bias = data.permute(2, 0, 1)
        else:
            if len(data.shape) == 1:
                work, bias = data, None
            else:
                work, bias = data.T

        # Compute the log_weights.
        if bias is None:
            n_samples = torch.tensor(work.shape[-1])
            log_weights = -torch.log(n_samples)
        else:
            # Normalize the weights.
            log_weights = torch.nn.functional.log_softmax(bias/kT, dim=-1)

        return - kT * torch.logsumexp(-work/kT + log_weights, dim=-1)

    return _tfep_estimator


# =============================================================================
# UTILS
# =============================================================================

def create_working_dir(tmp_dir_path):
    """Create a process specific working directory where to execute AMBER."""
    job_id = os.getenv('SLURM_JOB_ID', 'debug')
    work_dir_path = os.path.join(tmp_dir_path, job_id, 'proc'+str(os.getpid()))
    os.makedirs(work_dir_path, exist_ok=True)
    os.chdir(work_dir_path)


# =============================================================================
# READY TO USE TFEP MAPS
# =============================================================================

TFEP_MAPS = {
    'global-affine': maps.base.TFEPMap,
    'global-spline': maps.globalspline.TFEPMap,
    'zmatrix-spline': maps.zmatrix.TFEPMap,
    'global-volpres': maps.volpresglobal.TFEPMap,
    'zmatrix-volpres': maps.volpreszmatrix.TFEPMap,
}


# =============================================================================
# ANALYSIS
# =============================================================================

def evaluate_potentials(
        potential_energy_func,
        kT,
        tfep_cache,
        dataset,
        batch_size,
        global_step,
        flow=None
):
    """Evaluate and store the potentials in the cache for all samples in the data loader."""
    if flow is not None:
        flow.eval()

    # Check whether to collect timings.
    debug = logger.getEffectiveLevel() <= logging.DEBUG

    # Check if we have already computed some potentials.
    current_data = tfep_cache.read_eval_tensors(step_idx=global_step, as_numpy=True)
    # TODO: THIS DOESN'T WORK IF THE FIRST INDEX != 0
    if 'trajectory_sample_index' in current_data:
        # Identify the indices of the dataset that still needs to be computed.
        computed_traj_indices_set = set(current_data['trajectory_sample_index'])
        to_compute_dataset_indices = [i for i, v in enumerate(dataset.trajectory_sample_indices)
                                      if v not in computed_traj_indices_set]

        # Create a subset of the dataset.
        dataset = tfep.io.TrajectorySubset(dataset, to_compute_dataset_indices)

    # Create data_loader.
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    if debug:
        dt_total = time.time()

    # Compute all potentials.
    for batch_idx, batch_data in enumerate(data_loader):
        if debug:
            dt_batch = time.time()
        logger.info('Starting batch {}/{}'.format(batch_idx+1, len(data_loader)))

        x = batch_data['positions']

        # Apply map.
        if debug:
            dt_forward = time.time()
        if flow is not None:
            x, log_det_J = flow(x)
        if debug:
            dt_forward = time.time() - dt_forward

        # Compute the potential.
        if debug:
            dt_potential = time.time()
        try:
            potentials = potential_energy_func(x, batch_data['dimensions'])
        except KeyError:
            # There are no box vectors.
            potentials = potential_energy_func(x)
        if debug:
            dt_potential = time.time() - dt_potential

        # Store units in units of kT.
        potentials = potentials / kT

        # Store everything in the cache. It doesn't make sense to store
        # dataset_sample_index here since after resuming and taking a subset
        # they don't correspond to the indices in the dataset.
        if debug:
            dt_save = time.time()
        tensors = {
            'trajectory_sample_index': batch_data['trajectory_sample_index'],
            'potential': potentials,
        }
        if flow is not None:
            tensors['log_det_J'] = log_det_J

        tfep_cache.save_eval_tensors(step_idx=global_step, tensors=tensors)

        if debug:
            dt_save = time.time() - dt_save
            dt_batch = time.time() - dt_batch
            logger.debug('Timings (in seconds): forward={}, potential={}, saving={}, batch={}'.format(
                        dt_forward, dt_potential, dt_save, dt_batch))

    if debug:
        dt_total = time.time() - dt_total
        logger.debug('Total time: {}'.format(dt_total))

    logger.info('Evaluation completed!')


def read_data(standard_fep, mol_id, data_dir_path, is_opes=False, repeat_num=None):
    """Returns the data in the TFEP cache and the reference potential energies.

    All values are returned in the same units as they were stored.
    """
    # The column of the reference potential energies in the XVG file.
    energy_col_idx = 1

    # Load the TFEP cache with the target potential energies.
    tfep_cache_dir_path = os.path.join(data_dir_path, 'tfep_cache')
    tfep_cache = tfep.io.TFEPCache(tfep_cache_dir_path)

    # Read the data in the cache.
    if standard_fep:
        data = tfep_cache.read_eval_tensors(step_idx=0, sort_by='trajectory_sample_index', remove_nans='potential')
    else:
        data = tfep_cache.read_train_tensors(epoch_idx=0, remove_nans='potential')

    # Which repeats to load.
    if repeat_num is None:
        repeat_num = list(range(1, N_REPEATS+1))
    else:  # Make sure repeat_num is a list.
        repeat_num = [repeat_num]

    # Used to re-order the samples.
    samples_order = torch.argsort(data['trajectory_sample_index'])

    # Read the reference energies and bias.
    if is_opes:
        plumed_dataset = prepare_biased_dataset(mol_id, repeat_nums=repeat_num, load_energy=True)

        # Re-order samples correctly.
        data['ref_potential'] = plumed_dataset.data['ene'][samples_order]
        data['opes.bias'] = plumed_dataset.data['opes.bias'][samples_order]
    else:
        ene_file_paths = [os.path.join(HIPEN_OUT_DIR_PATH, 'energy', f'energy.{mol_id}.mm.mm.{i}.ene') for i in repeat_num]
        ref_potentials = []
        for ene_file_path in ene_file_paths:
            ref_potentials.append(np.loadtxt(ene_file_path, skiprows=1, usecols=energy_col_idx))
        ref_potentials = np.concatenate(ref_potentials)

        # Select/re-order samples correctly.
        data['ref_potential'] = torch.from_numpy(ref_potentials[samples_order.tolist()])

    # Save batch size.
    data['batch_size'] = tfep_cache.batch_size

    return data


def prepare_biased_dataset(mol_id, repeat_nums, load_energy=True, traj_dataset=None):
    """Load the PLUMED data in dataset and matches samples with the trajectory frames.

    Because PLUMED doesn't save the result of the last step and AMBER doesn't
    save the frame at time 0, the records must be shifted by one to match.
    Further, the first 5 ns of OPES simulation is discarded from both the
    PLUMED and trajectory data.

    Parameters
    ----------
    mol_id : str
        The molecule ID (e.g., '00140610').
    repeat_nums : List[int]
        If given, only these repeats are loaded (1 for first repeat). Otherwise,
        all repeats are loaded.
    load_energy : bool, optional
        If ``True``, the 'ene' column is also read from the PLUMED output file.
    traj_dataset : TrajectoryDataset, optional
        If given, a subset of the trajectory dataset is also created so that its
        data points match those in the plumed dataset with the same index.

    Returns
    -------
    plumed_dataset : PLUMEDDataset
        The dataset with the PLUMED data after discarding equilibration.
    traj_dataset : TrajectorySubset, optional
        If ``traj_dataset`` is given, this is the subset of frames matching those
        in ``plumed_dataset``.

    """
    n_repeats = len(repeat_nums)

    # Which columns to read?
    col_names = ['opes.bias']
    if load_energy:
        col_names.append('ene')

    # Read all PLUMED output files. The bias is already in kcal/mol.
    colvar_file_paths = [os.path.join(HIPEN_OUT_OPES_DIR_PATH, 'colvar', f'colvar.{mol_id}.{i}.data') for i in repeat_nums]
    plumed_datasets = [PLUMEDDataset(fp, col_names=col_names) for fp in colvar_file_paths]

    # Discard the first 5 ns of OPES equilibration. The PLUMED and AMBER
    # files are saved every 100 fs.
    equilibration = 5 * UNIT.nanosecond
    save_interval = 100 * UNIT.femtosecond
    n_discarded = int((equilibration / save_interval.to(UNIT.nanosecond)).magnitude)
    for d in plumed_datasets:
        for k, v in d.data.items():
            d.data[k] = v[n_discarded:]

    # Concatenate the replicate PLUMED output files.
    plumed_dataset = plumed_datasets[0]
    for k, v in plumed_dataset.data.items():
        plumed_dataset.data[k] = torch.cat([v] + [d.data[k] for d in plumed_datasets[1:]])

    if traj_dataset is not None:
        # The first and last samples saved by AMBER are at time save_interval
        # and 15ns while in PLUMED they are at 0 and 15ns - save_interval so,
        # beside discarding the first 5ns of each trajectory, we also need to
        # remove the last frame for the datasets to be the same size.
        n_samples_per_traj = len(traj_dataset) // n_repeats
        indices = torch.arange(n_discarded-1, n_samples_per_traj-1)
        indices = torch.cat([indices + i*n_samples_per_traj for i in range(n_repeats)])
        traj_dataset = tfep.io.TrajectorySubset(traj_dataset, indices=indices)

        return plumed_dataset, traj_dataset
    return plumed_dataset


# =============================================================================
# MAIN
# =============================================================================

def train_and_eval(
        mol_id,
        flow_name,
        batch_size=None,
        standard_fep=False,
        subdir_name='',
        is_opes=False,
        repeat_num=None,
        seed=0,
        **map_kwargs
):
    # Make sure the experiment is reproducible.
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Use double throughout since the NN is not the bottleneck in memory/speed.
    torch.set_default_dtype(torch.float64)

    # Directory where the reference simulations are stored.
    if is_opes:
        hipen_out_dir_path = HIPEN_OUT_OPES_DIR_PATH
    else:
        hipen_out_dir_path = HIPEN_OUT_DIR_PATH

    # Which replicate simulations to merge to build the dataset?
    if repeat_num is None:
        repeat_num = list(range(1, N_REPEATS+1))
    else:  # Make sure repeat_num is a list.
        repeat_num = [repeat_num]

    # Concatenate all trajectories into a single dataset.
    mol_name = 'zinc_' + mol_id
    psf_file_path = os.path.join(HIPEN_DIR_PATH, 'coors', mol_name+'.psf')
    traj_file_paths = [os.path.join(hipen_out_dir_path, 'traj', f'dyna.{mol_id}.mm.{i}.dcd') for i in repeat_num]
    universe = MDAnalysis.Universe(psf_file_path, *traj_file_paths)

    # Create the trajectory dataset.
    dataset = tfep.io.TrajectoryDataset(universe=universe)

    # Add information about bias if we must reweight the loss function.
    if is_opes:
        # Read PLUMED data, discard equilibration, and match PLUMED entries with the trajectory frames.
        plumed_dataset, dataset = prepare_biased_dataset(
            mol_id, repeat_nums=repeat_num, load_energy=False, traj_dataset=dataset)
        # Merge the two sets in a single one.
        dataset = tfep.io.dataset.MergedDataset(dataset, plumed_dataset)

    # Check environment and number of CPUs available.
    n_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 2))
    assert int(os.getenv('SLURM_NTASKS', 1)) == 1
    assert int(os.getenv('SLURM_JOB_NUM_NODES', 1)) == 1

    # Determine default batch size.
    if batch_size is None:
        batch_size = n_cpus

    # Initialize the data loader.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not standard_fep,
    )

    # Initialize TFEPCache where to save potential energies.
    if standard_fep:
        tfep_cache_dir_path = os.path.join('fep', mol_name, subdir_name, 'tfep_cache')
    else:
        tfep_cache_dir_path = os.path.join('tfep', mol_name, subdir_name, 'tfep_cache')
    tfep_cache = tfep.io.TFEPCache(save_dir_path=tfep_cache_dir_path, data_loader=data_loader)

    # Set pytorch number of threads for intraop parallelism.
    torch.set_num_threads(n_cpus)

    # Get the atom symbols and total charge of the molecule.
    symbols = []
    tot_charge = 0.0
    for atom in universe.atoms:
        # atom.name is something like "C12" so we need to remove the numbers.
        name = ''.join([c for c in atom.name if not c.isdigit()])
        symbols.append(name.capitalize())
        tot_charge += atom.charge
    tot_charge = round(tot_charge)
    logger.info(f'Fixing the QM charge of {mol_id} to {tot_charge}')

    # Create a tmp directory with a copy of amber_3ob.in with the correct total charge.
    tmp_dir_path = os.path.realpath(os.path.join('tmp', mol_name))
    amber_in_file_path = os.path.join(tmp_dir_path, 'amber_3ob.in')
    os.makedirs(tmp_dir_path, exist_ok=True)
    if not os.path.exists(amber_in_file_path):
        with open(os.path.join(AMBER_DIR_PATH, 'amber_3ob.in'), 'r') as fin, open(amber_in_file_path, 'w') as fout:
            for line in fin:
                if 'qmcharge=0' in line:
                    line = line.replace('qmcharge=0', 'qmcharge='+str(tot_charge))
                fout.write(line)

    # Run. The initializer will configure the working dir of each subprocess.
    mp_context = multiprocessing.get_context('forkserver')
    with mp_context.Pool(n_cpus, initializer=create_working_dir, initargs=(tmp_dir_path,)) as pool:
        # The target potential.
        potential_energy_func = tfep.potentials.ase.PotentialASE(
            calculator=ase.calculators.amber.Amber(
                amber_exe='sander -O',
                infile=amber_in_file_path,
                outfile='amber.out',
                topologyfile=os.path.join(PRMTOP_DIR_PATH, mol_name+'.prmtop'),
                incoordfile='coord.crd',
            ),
            symbols=symbols,
            position_unit=UNIT.angstrom,
            energy_unit=UNIT.kcal/UNIT.mole,
            parallelization_strategy=ProcessPoolStrategy(pool),
        )

        # Convert kT to the units returned by potential_energy_func.
        kT = KT.to(potential_energy_func.energy_unit).magnitude

        if standard_fep:
            evaluate_potentials(
                potential_energy_func,
                kT,
                tfep_cache,
                data_loader.dataset,
                data_loader.batch_size,
                global_step=0
            )
        else:
            # Create the trainer and fit the network.
            flow = TFEP_MAPS[flow_name](
                universe=universe,
                **map_kwargs,
            )

            # Configure trainer.
            trainer = TFEPTrainer(
                flow=flow,
                potential_energy_func=potential_energy_func,
                kT=kT,
                tfep_cache=tfep_cache,
                save_model_step_interval=500,
            )

            # Train the network.
            trainer.fit(data_loader)


def run_tfep_analysis(mol_id, standard_fep=False, subdir_name='', is_opes=False, repeat_num=None):
    """Compute the free energy difference and work statistics."""
    # Directory where the reference simulations are stored.
    mol_name = 'zinc_' + mol_id
    if standard_fep:
        save_dir_path = os.path.join('fep', mol_name, subdir_name)
    else:
        save_dir_path = os.path.join('tfep', mol_name, subdir_name)

    # Read data.
    data = read_data(standard_fep, mol_id=mol_id, data_dir_path=save_dir_path, is_opes=is_opes, repeat_num=repeat_num)

    # All energies in the tfepcache are stored in units of kT.
    # The reference potentials are stored in kcal/mol.
    kT = KT.to(UNIT.kcal/UNIT.mol).magnitude
    target_potentials = data['potential'] * kT
    ref_potentials = data['ref_potential']

    if standard_fep:
        mapped_potentials = target_potentials
    else:
        mapped_potentials = target_potentials - data['log_det_J'] * kT

    # Log so that we now how many energy calculations have not converged.
    print('Found', len(target_potentials), 'valid samples to compute the DF.')

    # Read the reference results (which are stored in kcal/mol).
    with open(os.path.join(HIPEN_DIR_PATH, 'reference.json'), 'r') as f:
        reference_data = json.load(f)[mol_id]

    # Apply offset between CHARMM and AMBER.
    with open(os.path.join(AMBER_DIR_PATH, 'offsets_3ob.json'), 'r') as f:
        amber_offset = json.load(f)[mol_id]
    reference_data['DA AMBER'] = reference_data['CRO']['DA'] - amber_offset

    # Generate a tfep estimator in units of kcal/mol.
    tfep_estimator = generate_tfep_estimator(kT)

    # Compute the generalized work.
    work = mapped_potentials - ref_potentials

    # The bias is stored in kcal/mol.
    try:
        tfep_data = torch.vstack([work, data['opes.bias']]).T
    except KeyError:  # Unbiased simulation.
        tfep_data = work

    # Compute the free energy and work statistics with the whole dataset.
    df = tfep_estimator(tfep_data)

    # Compute some statistics.
    potential_err = work - reference_data['DA AMBER']
    rmse = torch.sqrt((potential_err**2).mean())
    mae = torch.abs(potential_err).mean()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        ref_potentials.detach().numpy(), mapped_potentials.detach().numpy())

    result = {
        'df': {'true': df},  # There will be also bootstrap statistics here.
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_value**2': r_value**2,
        'rmse': rmse,
        'mae': mae,
        'std target': torch.std(target_potentials),
        'std mapped': torch.std(mapped_potentials),
        'std reference': torch.std(ref_potentials),
        'std work': torch.std(work),
    }

    # Bootstrap.
    result['bootstrap_sample_size'] = [100, 250, 500, 1000, 2500, 5000, 10000, 20000, 50000, 100000]
    if len(work) > result['bootstrap_sample_size'][-1]:
        result['bootstrap_sample_size'].extend([200000, 300000, 500000, len(work)])
    result['df']['bootstrap'] = tfep.analysis.bootstrap(
        data=tfep_data,
        statistic=tfep_estimator,
        n_resamples=2000,
        bootstrap_sample_size=result['bootstrap_sample_size'],
        take_first_only=not standard_fep,
        batch=1000,
    )

    # Convert all tensors to lists/floats using recursion before serializing.
    def tensor_to_python(_x):
        if isinstance(_x, dict):
            for k, v in _x.items():
                _x[k] = tensor_to_python(v)
        elif isinstance(_x, list):
            for i, v in enumerate(_x):
                _x[i] = tensor_to_python(v)
        else:
            try:
                _x = _x.tolist()
            except AttributeError:
                pass
        return _x

    result = tensor_to_python(result)

    from pprint import pprint
    print('\nstandard_fep =', standard_fep)
    print('reference df =', reference_data['DA AMBER'], 'kcal/mol')
    pprint(result)
    print()

    # Save everything on disk.
    os.makedirs(save_dir_path, exist_ok=True)
    with open(os.path.join(save_dir_path, 'analysis.json'), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    # The arguments passed through the command line overwrite the SLURM variables.
    import argparse
    parser = argparse.ArgumentParser()

    # Options for both standard and targeted reweighting.
    parser.add_argument('--molid', dest='mol_id',
                        help='The ID of the molecule (e.g., 00140610).')
    parser.add_argument('--flow', dest='flow_name',
                        help='The name of the flow. Allowed values are global-affine, global-spline, '
                             'zmatrix-spline, global-volpres, and zmatrix-volpres.')
    parser.add_argument('--nmaf', dest='n_maf_layers', default=6, type=int,
                        help='The number of MAF layers of the transformation flow.')
    parser.add_argument('--nmafic', dest='n_maf_layers_ic', default=0, type=int,
                        help='The number of MAF layers used for the change of coordinate flow.')
    parser.add_argument('--fep', dest='standard_fep', action='store_true',
                        help='Runs the standard FEP without training a flow')
    parser.add_argument('--subdir', dest='subdir_name', default='',
                        help='The name of the output subdirectory within fep/ or tfep/.')
    parser.add_argument('--opes', dest='is_opes', action='store_true',
                        help='Flags that the reference simulation is an OPES run and must be reweighted.')
    parser.add_argument('--repeat', dest='repeat_num', type=int,
                        help='Run TFEP only on this replicate simulation (1 for the first simulation).')
    parser.add_argument('--df', dest='compute_df', action='store_true',
                        help='Use the cached potential to compute the free energy difference.')
    parser.add_argument('--batch', dest='batch_size', type=int, help='Batch size used for training.')
    args = parser.parse_args()

    compute_df = args.compute_df
    del args.compute_df

    if compute_df:
        # remove unused arguments if present.
        for arg_name in ['flow_name', 'n_maf_layers', 'n_maf_layers_ic', 'batch_size']:
            if hasattr(args, arg_name):
                delattr(args, arg_name)
        run_tfep_analysis(**vars(args))
    else:
        train_and_eval(**vars(args))
