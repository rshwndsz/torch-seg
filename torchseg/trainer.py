# Python STL
import time
import os
# PyTorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.backends.cudnn as cudnn

# Local
from .loss import MixedLoss
from .data import provider
from .data import DATA_FOLDER
from .metrics import Meter, epoch_log

_DIRNAME = os.path.dirname(__file__)
_CHECKPOINT_PATH = os.path.join(_DIRNAME, "checkpoints", "model-2.pth")


class Trainer(object):
    """This class takes care of training and validation of our model"""

    def __init__(self, model):
        # Set hyperparameters
        self.num_workers = 8  # Raise this if shared memory is high
        self.batch_size = {"train": 32, "val": 32}
        self.lr = 3e-4  # See: https://twitter.com/karpathy/status/801621764144971776?lang=en
        self.num_epochs = 100
        self.phases = ["train", "val"]

        # Torch-specific initializations
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

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
        self.best_loss = float("inf")  # Very high best_loss for the first iteration
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.acc_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        """Forward pass"""
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        """1 epoch in the life of a model"""
        # TODO: Use relative time instead of absolute
        # Log epoch, phase and time
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")

        # Set up model, loader and initialize losses
        self.net.train(phase == "train")
        batch_size = self.batch_size[phase]
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0

        # Learning!
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            # Forward pass
            loss, outputs = self.forward(images, targets)
            if phase == "train":
                # Backprop for training only
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Get losses
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)

        # Calculate losses
        epoch_loss = running_loss / total_batches
        dice, iou, acc = epoch_log(phase, epoch, epoch_loss, meter, start)
        # Collect losses
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        self.acc_scores[phase].append(acc)

        # Empty GPU cache
        torch.cuda.empty_cache()
        # Return average loss from the criterion for this epoch
        return epoch_loss

    def start(self):
        """Start the loops!"""
        #         from IPython.core.debugger import set_trace
        #         set_trace()
        # TODO: Add start and end epochs
        for epoch in range(1, self.num_epochs + 1):
            # Train model for 1 epoch
            self.iterate(epoch, "train")
            # Construct the state for a possible save later
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            # TODO: Add validation frequency
            # Validate model for 1 epoch
            if epoch % 5 == 0:
                val_loss = self.iterate(epoch, "val")
                # Step the scheduler based on validation loss
                self.scheduler.step(val_loss)
                # TODO: Add EarlyStopping
                # TODO: Add model saving on KeyboardInterrupt (^C)

                # Save model if validation loss is lesser than anything seen before
                if val_loss < self.best_loss:
                    print("******** New optimal found, saving state ********")
                    state["best_loss"] = self.best_loss = val_loss
                    # TODO: Add error handling here
                    # TODO: Use a different file for each save
                    # TODO: Sample file name: ./checkpoints/model-e-020-v-0.1234.pth
                    torch.save(state, _CHECKPOINT_PATH)
            print()
