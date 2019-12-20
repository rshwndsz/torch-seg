# Python STL
import time
import os
import logging
from typing import Dict, List, Tuple
# PyTorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.backends.cudnn as cudnn

# Local
from .loss import MixedLoss
from .data import provider
from .data import DATA_FOLDER
from .metrics import Meter

_DIRNAME = os.path.dirname(__file__)
_TIME_FMT = "%I:%M:%S %p"


class Holder(object):
    """An object to store values till end of training

    Attributes
    ----------
    phases : tuple(str)
        Phases of learning
    scores : tuple(str)
        Short names (keys as defined in Meter) of scores
    store : dict{str, dict{str, list}}
        Store list of values for each phase
    """
    def __init__(self,
                 phases: Tuple[str, ...] = ('train', 'val'),
                 scores: Tuple[str, ...] = ('loss', 'iou')):
        self.phases: Tuple[str] = phases
        self.scores: Tuple[str] = scores
        self.store: Dict[Dict[str, List[float]]] = {
            score: {
                phase: [] for phase in self.phases
            } for score in self.scores
        }

    def add(self, metrics, phase):
        for score in self.store.keys():
            try:
                self.store[score][phase].append(metrics[score])
            except KeyError:
                logger = logging.getLogger(__name__)
                logger.warning(f"Key '{score}' not found. Skipping...", exc_info=True)
                continue

    def reset(self):
        self.store: Dict[Dict[str, List[float]]] = {
            score: {
                phase: [] for phase in self.phases
            } for score in self.scores
        }


class Trainer(object):
    """An object to encompass all training and validation

    Training loop, validation loop, logging, checkpoints are all
    implemented here.

    Attributes
    ----------
    num_workers : int
        Number of workers
    batch_size : int
        Batch size
    lr : int
        Learning rate
    num_epochs : int
        Number of epochs
    current_epoch : int
        Current epoch
    phases : list[str]
        List of learning phases
    val_freq : int
        Validation frequency
    device : torch.device
        GPU or CPU
    checkpoint_path : str
        Path to checkpoint file
    save_path : str
        Path to file where state will be saved
    net
        Our NN in PyTorch
    criterion
        Loss function
    optimizer
        Optimizer
    scheduler
        Learning rate scheduler
    dataloaders : dict[str, torch.utils.data.DataLoader]
        Dataloaders for each phase
    best_loss : float
        Best validation loss
    holder : Holder
        Object to store loss & scores
    """
    def __init__(self, model, args):
        """Initialize a Trainer object

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model of your NN
        args : :obj:
            CLI arguments
        """
        # Set hyperparameters
        self.num_workers: int = args.num_workers  # Raise this if shared memory is high
        self.batch_size: Dict[str, int] = {"train": args.batch_size,
                                           "val": args.batch_size}
        self.lr: float = args.lr  # See: https://twitter.com/karpathy/status/801621764144971776?lang=en
        self.num_epochs: int = args.num_epochs
        self.current_epoch: int = 0
        self.phases: Tuple[str, ...] = ("train", "val")
        self.val_freq: int = args.val_freq

        # Torch-specific initializations
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")
        else:
            self.device = torch.device("cuda:0")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        if args.checkpoint_name is not None:
            self.checkpoint_path: str = os.path.join(_DIRNAME, "checkpoints",
                                                     args.checkpoint_name)
        else:
            self.checkpoint_path = None

        self.save_path: str = os.path.join(_DIRNAME, "checkpoints",
                                           args.save_fname)

        # Model, loss, optimizer & scheduler
        self.net = model
        self.net = self.net.to(self.device)  # <<<< Catch: https://pytorch.org/docs/stable/optim.html
        self.criterion = MixedLoss(9.0, 4.0)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr)  # "Adam is safe" - http://karpathy.github.io/2019/04/25/recipe/
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                           patience=3, verbose=True,
                                           cooldown=0, min_lr=3e-6)

        # Faster convolutions at the expense of memory
        cudnn.benchmark = True

        # Get loaders for training and validation
        self.dataloaders = {
            phase: provider(
                data_folder=DATA_FOLDER,
                phase=phase,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }

        # Initialize losses & scores
        self.best_loss: float = float("inf")  # Very high best_loss for the first iteration
        self.holder = Holder(self.phases, scores=('loss', 'iou', 'dice',
                                                  'dice_2', 'aji', 'prec',
                                                  'pq', 'sq', 'dq'))

    def forward(self,
                images: torch.Tensor,
                targets: torch.Tensor):
        """Forward pass

        Parameters
        ----------
        images : torch.Tensor
            Input to the NN
        targets : torch.Tensor
            Supervised labels for the NN

        Returns
        -------
        loss: torch.Tensor
            Loss from one forward pass
        logits: torch.Tensor
            Raw output of the NN, without any activation function
            in the last layer
        """
        images = images.to(self.device)
        masks = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, masks)
        return loss, logits

    def iterate(self,
                epoch: int,
                phase: str):
        """1 epoch in the life of a model

        Parameters
        ----------
        epoch : int
            Current epoch
        phase : str
            Phase of learning
            In ['train', 'val']
        Returns
        -------
        epoch_loss: float
            Average loss for the epoch
        """
        # Initialize meter
        meter = Meter(phase, epoch)
        # Log epoch, phase and start time
        start_time = time.strftime(_TIME_FMT, time.localtime())
        logger = logging.getLogger(__name__)
        logger.info(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start_time}")

        # Set up model, loader and initialize losses
        self.net.train(phase == "train")
        batch_size = self.batch_size[phase]
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0

        # Learning!
        self.optimizer.zero_grad()
        # TODO: Add progress bar
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            # Forward pass
            loss, logits = self.forward(images, targets)
            if phase == "train":
                # Backprop for training only
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Get losses
            with torch.no_grad():
                running_loss += loss.item()
                logits = logits.detach().cpu()
                meter.update(targets, logits)

        # Collect loss & scores
        epoch_loss = running_loss / total_batches
        metrics = meter.epoch_log(epoch_loss, start_time, _TIME_FMT)
        # Store epoch-wise loss and scores
        # TODO: Move this spaghetti into Meter with hooks (ノಠ益ಠ)ノ彡┻━┻
        self.holder.add(metrics, phase)

        # Empty GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Return average loss from the criterion for this epoch
        return epoch_loss

    def start(self):
        """Start the loops!"""
        for epoch in range(1, self.num_epochs + 1):    # <<< Change: Hardcoded starting epoch
            # Update current_epoch
            self.current_epoch = epoch
            # Train model for 1 epoch
            self.iterate(epoch, "train")
            # Construct the state for a possible save later
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            # Validate model for 1 epoch
            if epoch % self.val_freq == 0:
                val_loss = self.iterate(epoch, "val")
                # Step the scheduler based on validation loss
                self.scheduler.step(val_loss)
                # TODO: Add EarlyStopping

                # Save model if validation loss is lesser than anything seen before
                if val_loss < self.best_loss:
                    logger = logging.getLogger(__name__)
                    logger.info("******** New optimal found, saving state ********")
                    state["best_loss"] = self.best_loss = val_loss
                    try:
                        torch.save(state, self.save_path)
                    except FileNotFoundError as e:
                        logger.exception(f"Error while saving checkpoint", exc_info=True)
            print()
