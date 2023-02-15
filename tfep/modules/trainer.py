#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Helper class to train normalizing flows.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import logging
import os
import time

import torch

import tfep.loss

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Configure logging.
logger = logging.getLogger(__name__)


# =============================================================================
# TFEP TRAINER
# =============================================================================

class TFEPTrainer:
    """
    Utility class to train a normalizing flow map for TFEP.

    The class offers the following facilities:

    - Running the training loop and saving the potential energies and log Jacobians
      evaluated for the loss function in a :class:`~tfep.io.TFEPCache`.
    - Saving checkpoints at each step and flow models at regular intervals.
    - Resuming from a previous checkpoint, even if in the middle of an epoch.

    By default we place the checkpoints in the same directory used for the tfep
    cache.

    Parameters
    ----------
    flow : torch.nn.Module
        The normalizing flow to train.
    potential_energy_func : torch.nn.Module
        A PyTorch module encapsulating the target potential energy function
        (e.g. :class:`tfep.potentials.psi4.PotentialPsi4`).
    kT : float
        The value of the thermal energy in the same units returned by
        ``potential_energy_func``.
    tfep_cache : tfep.io.TFEPCache
        The database used to cache the potential energies and Jacobians computed
        during training.
    checkpoints_dir_path : str, optional
        The directory where checkpoints and flow models must be saved. If not
        passed, the directory is placed in the same folder used for ``tfep_cache``.
    optimizer : torch.optim.Optimizer, optional
        The optimizer used for the parameter optimization. If not passed a default
        ``torch.optim.AdamW`` optimizer is used.
    max_n_epochs : int or None
        The number of epochs to run. If resuming from an existing checkpoint,
        this can be None as it is restored from the checkpoint.
    save_checkpoint_step_interval : int
        The number of steps between two saved models. On resuming, this is ignored
        and its value is loaded from the checkpoint.
    save_model_step_interval : int or None
        The number of steps between two saved models. On resuming, this is ignored
        and its value is loaded from the checkpoint. If smaller than 1, only the
        latest model is saved. Note that this does not control the checkpoint
        frequency.

    """

    LATEST_MAP_FILE_NAME = 'latest_map.pth'
    TRAINER_CHECKPOINT_FILE_NAME = 'trainer_checkpoint.pth'
    DEFAULT_CHECKPOINTS_DIR_NAME = 'checkpoints'

    def __init__(
            self,
            flow,
            potential_energy_func,
            kT,
            tfep_cache,
            checkpoints_dir_path=None,
            optimizer=None,
            scheduler=None,
            max_n_epochs=1,
            save_checkpoint_step_interval=1,
            save_model_step_interval=0,
    ):
        self.flow = flow
        self.potential_energy_func = potential_energy_func
        self.kT = kT
        self.tfep_cache = tfep_cache

        if checkpoints_dir_path is None:
            checkpoints_dir_path = self._default_checkpoint_dir_path
        self.checkpoints_dir_path = os.path.abspath(checkpoints_dir_path)

        if optimizer is None:
            optimizer = self.create_default_optimizer()
        self.optimizer = optimizer

        self.scheduler = scheduler
        self.max_n_epochs = max_n_epochs
        self.save_checkpoint_step_interval = save_checkpoint_step_interval
        self.save_model_step_interval = save_model_step_interval

        self.current_epoch_idx = 0
        self.current_batch_idx = 0
        self._base_seed = None

        # Check if a checkpoint exist.
        checkpoint_file_path = self.trainer_checkpoint_file_path
        if os.path.isfile(checkpoint_file_path):
            logger.info('Resuming from {}'.format(checkpoint_file_path))
            self._load_checkpoint(max_n_epochs, save_model_step_interval, save_checkpoint_step_interval)

    @property
    def flow_checkpoint_file_path(self):
        """The path to the file storing the checkpoint for the flow model state."""
        return os.path.join(self.checkpoints_dir_path, self.LATEST_MAP_FILE_NAME)

    @property
    def trainer_checkpoint_file_path(self):
        """The path to the file storing the checkpoint for the trainer state."""
        return os.path.join(self.checkpoints_dir_path, self.TRAINER_CHECKPOINT_FILE_NAME)

    def fit(self, data_loader):
        """Run the training on the given data."""
        # Check whether to collect timings.
        debug = logger.getEffectiveLevel() <= logging.DEBUG

        # Set training mode.
        self.flow.train()

        # Create checkpoint directory.
        os.makedirs(self.checkpoints_dir_path, exist_ok=True)

        # Initialize loss function.
        loss_func = tfep.loss.BoltzmannKLDivLoss()

        # It is important to be able resume mid-batch. Both because evaluating QM
        # potential can be expensive, and also to collect full epochs of these potentials.
        # To do this, we save the seed used to generate the first epoch, and then
        # at the i-th epoch we reseed using first_seed+i+1. This way, whatever
        # random operation happens before or after the DataLoader generates a
        # permutation does not affect the input.
        n_batches = len(data_loader)
        for epoch_idx in range(self.current_epoch_idx, self.max_n_epochs):
            if debug:
                dt_epoch = time.time()

            # Now reseed for reproducibility of the batch.
            if self._base_seed is None:
                self._base_seed = torch.seed()
            torch.manual_seed(self._base_seed+epoch_idx+1)

            # Go through all batches until we find the current one.
            for batch_idx, batch_data in enumerate(data_loader):
                if batch_idx < self.current_batch_idx:
                    continue
                if debug:
                    dt_batch = time.time()

                # Log progress.
                logger.info(f'Starting epoch {epoch_idx+1}/{self.max_n_epochs}, '
                            f'batch {batch_idx+1}/{n_batches}')

                # Forward.
                x = batch_data['positions']
                if debug:
                    dt_forward = time.time()
                result = self.flow(x)
                if debug:
                    dt_forward = time.time() - dt_forward

                # Continuous flows also return a regularization term.
                try:
                    y, log_det_J = result
                    reg = None
                except ValueError:
                    y, log_det_J, reg = result

                # Compute potentials and loss.
                if debug:
                    dt_potential = time.time()
                try:
                    potential_y = self.potential_energy_func(y, batch_data['dimensions'])
                except KeyError:
                    # There are no box vectors.
                    potential_y = self.potential_energy_func(y)
                if debug:
                    dt_potential = time.time() - dt_potential

                # Convert potentials to units of kT.
                potential_y = potential_y / self.kT

                # Convert bias to units of kT.
                try:
                    log_weights = batch_data['opes.bias'] / self.kT
                except KeyError:  # Unbiased simulation.
                    log_weights = None

                # Compute loss.
                loss = loss_func(target_potentials=potential_y, log_det_J=log_det_J, log_weights=log_weights)

                # Add regularization for continuous flows.
                if reg is not None:
                    loss = loss + reg.mean()

                # Backpropagation.
                self.optimizer.zero_grad()
                if debug:
                    dt_backward = time.time()
                loss.backward()
                if debug:
                    dt_backward = time.time() - dt_backward

                # Optimization step.
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step(loss)

                # Log potentials.
                if debug:
                    dt_save = time.time()
                self.tfep_cache.save_train_tensors(
                    tensors={
                        'dataset_sample_index': batch_data['dataset_sample_index'],
                        'trajectory_sample_index': batch_data['trajectory_sample_index'],
                        'potential': potential_y,
                        'log_det_J': log_det_J,
                    },
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                )

                # Log loss.
                self.tfep_cache.save_train_metrics(
                    tensors={'loss': loss},
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                )

                # Save checkpoint.
                self.current_batch_idx += 1
                if batch_idx == n_batches-1:
                    # Increment epoch idx before saving a checkpoint so that we'll
                    # resume from the next one.
                    self.current_epoch_idx += 1

                # Save the checkpoint.
                step_idx = epoch_idx * n_batches + batch_idx
                if step_idx % self.save_checkpoint_step_interval == 0:
                    self.save_checkpoint()

                # Save network parameters.
                if (self.save_model_step_interval > 0) and (step_idx % self.save_model_step_interval == 0):
                    model_name = f'epoch-{epoch_idx}-batch-{batch_idx}.pth'
                    model_file_path = os.path.join(self.checkpoints_dir_path, model_name)
                    torch.save(self.flow.state_dict(), model_file_path)

                if debug:
                    dt_save = time.time() - dt_save
                    dt_batch = time.time() - dt_batch
                    logger.debug('Timings (in seconds): forward={}, potential={}, backward={}, saving={}, batch={}'.format(
                        dt_forward, dt_potential, dt_backward, dt_save, dt_batch))

            if debug:
                dt_epoch = time.time() - dt_epoch
                logger.debug('Total epoch time (in seconds): {}'.format(dt_epoch))

        logger.info('Training completed!')

    def create_default_optimizer(self):
        """Create the default optimizer (currently AdamW)."""
        return torch.optim.AdamW(self.flow.parameters())

    def save_checkpoint(self):
        """Save the state of the model and the trainer for later resuming."""
        # Save latest flow separately so that we can simply load it for evaluation.
        torch.save(self.flow.state_dict(), self.flow_checkpoint_file_path)

        # Trainer and optimizer parameters.
        trainer_checkpoint = {
            'optimizer': self.optimizer.state_dict(),
            'max_n_epochs': self.max_n_epochs,
            'save_model_step_interval': self.save_model_step_interval,
            'save_checkpoint_step_interval': self.save_checkpoint_step_interval,
            'current_epoch_idx': self.current_epoch_idx,
            'current_batch_idx': self.current_batch_idx,
            'base_seed': self._base_seed,
        }
        if self.scheduler is not None:
            trainer_checkpoint['scheduler'] = self.scheduler.state_dict()
        torch.save(trainer_checkpoint, self.trainer_checkpoint_file_path)

    @property
    def _default_checkpoint_dir_path(self):
        """By default we place the checkpoints in the same directory used for the tfep cache."""
        return os.path.join(os.path.dirname(self.tfep_cache.save_dir_path), self.DEFAULT_CHECKPOINTS_DIR_NAME)

    def _load_checkpoint(self, max_n_epochs, save_model_step_interval, save_checkpoint_step_interval):
        """Load the state of the flow, trainer, and optimizer."""
        # Load the flow parameters.
        self.flow.load_state_dict(torch.load(self.flow_checkpoint_file_path))

        # Load state the trainer state.
        trainer_state = torch.load(self.trainer_checkpoint_file_path)

        # Resume optimizer.
        self.optimizer.load_state_dict(trainer_state['optimizer'])

        # Resume scheduler.
        if 'scheduler' in trainer_state:
            self.scheduler.load_state_dict(trainer_state['scheduler'])

        # max_n_epochs passed in __init__ overwrites the saved one.
        if self.max_n_epochs is None:
            self.max_n_epochs = trainer_state['max_n_epochs']
        elif self.max_n_epochs < trainer_state['max_n_epochs']:
            raise ValueError(f"Passed max_n_epochs={max_n_epochs} but the flow has been "
                             f"trained already for {trainer_state['max_n_epochs']} epochs")

        # Update attributes that cannot currently be changed on resuming.
        self.current_epoch_idx = trainer_state['current_epoch_idx']
        self.current_batch_idx = trainer_state['current_batch_idx']
        self.save_model_step_interval = trainer_state['save_model_step_interval']
        self.save_checkpoint_step_interval = trainer_state['save_checkpoint_step_interval']
        self._base_seed = trainer_state['base_seed']
