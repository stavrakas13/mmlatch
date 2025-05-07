import os
from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast

import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, State
from ignite.handlers import EarlyStopping
from ignite.metrics import Loss, RunningAverage
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from mmlatch.handlers import CheckpointHandler, EvaluationHandler
from mmlatch.util import from_checkpoint, to_device, GenericDict

TrainerType = TypeVar("TrainerType", bound="Trainer")


class Trainer(object):
    def __init__(
        self: TrainerType, # refers to instance of Trainer
        model: nn.Module, 
        optimizer: Optimizer,
        lr_scheduler=None, #learning rate scheduler
        newbob_metric="loss", #metric for newbob scheduler
        checkpoint_dir: str = "../checkpoints",
        experiment_name: str = "experiment", #used for organizing experiments in /checkpoints
        score_fn: Optional[Callable] = None, #score to evaluate performance
        model_checkpoint: Optional[str] = None,
        optimizer_checkpoint: Optional[str] = None,
        metrics: GenericDict = None, # a dictionary for additional metrics
        patience: int = 10, #epochs before early stopping
        validate_every: int = 1, #frequency of performing validation
        accumulation_steps: int = 1, #number of steps before backprop
        loss_fn: _Loss = None, #loss function
        non_blocking: bool = True, #asynchronous data transfer to gpu
        retain_graph: bool = False, #retain computation graph after backprop
        dtype: torch.dtype = torch.float,
        device: str = "cpu",
        enable_plot_embeddings: bool = False,  # New plot_embeddings mode flag
    ) -> None:
        self.dtype = dtype
        self.retain_graph = retain_graph
        self.non_blocking = non_blocking
        self.device = device
        self.loss_fn = loss_fn
        self.validate_every = validate_every
        self.patience = patience
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.enable_plot_embeddings = enable_plot_embeddings  # Store plot_embeddings mode flag
        
        #validates the checkpoint paths
        model_checkpoint = self._check_checkpoint(model_checkpoint)
        optimizer_checkpoint = self._check_checkpoint(optimizer_checkpoint)
        #loads  model from checkpoint
        self.model = cast(
            nn.Module,
            from_checkpoint(model_checkpoint, model, map_location=torch.device("cpu")),
        )
        #cast the model parameter to specified data types
        self.model = self.model.type(dtype).to(device)
        #load optimizer from checkpoint
        self.optimizer = from_checkpoint(optimizer_checkpoint, optimizer)
        self.lr_scheduler = lr_scheduler

        if metrics is None:
            metrics = {}
        #Ensures that loss is always calculated
        if "loss" not in metrics:
            metrics["loss"] = Loss(self.loss_fn)
        #initialize the trainer, train evaluator and validation evaluator
        self.trainer = Engine(self.train_step)
        self.train_evaluator = Engine(self.eval_step)
        self.valid_evaluator = Engine(self.eval_step)
        #attaches the metrics to the train and validation evaluators
        for name, metric in metrics.items():
            metric.attach(self.train_evaluator, name)
            metric.attach(self.valid_evaluator, name)

        self.pbar = ProgressBar()
        self.val_pbar = ProgressBar(desc="Validation")

        self.score_fn = score_fn if score_fn is not None else self._score_fn
        #initialize the checkpoint handler
        if checkpoint_dir is not None:
            self.checkpoint = CheckpointHandler(
                checkpoint_dir,
                experiment_name,
                score_name="validation_loss",
                score_function=self.score_fn,
                n_saved=2,
                require_empty=False,
                save_as_state_dict=True,
            )
        #initialize the early stopping handler
        self.early_stop = EarlyStopping(patience, self.score_fn, self.trainer)
        #initialize the evaluation handler
        self.val_handler = EvaluationHandler(
            pbar=self.pbar,
            validate_every=1,
            early_stopping=self.early_stop,
            newbob_scheduler=self.lr_scheduler,
            newbob_metric=newbob_metric,
        )
        # attach methods sets up event handlers, metrics, and other integrations with Ignite's engines.
        self.attach()
        print(
            f"Trainer configured to run {experiment_name}\n"
            f"\tpretrained model: {model_checkpoint} {optimizer_checkpoint}\n"
            f"\tcheckpoint directory: {checkpoint_dir}\n"
            f"\tpatience: {patience}\n"
            f"\taccumulation steps: {accumulation_steps}\n"
            f"\tnon blocking: {non_blocking}\n"
            f"\tretain graph: {retain_graph}\n"
            f"\tdevice: {device}\n"
            f"\tmodel dtype: {dtype}\n"
        )
    def _check_checkpoint(self: TrainerType, ckpt: Optional[str]) -> Optional[str]:
        """checks if checkpoint is valid"""
        if ckpt is None:
            return ckpt

        ckpt = os.path.join(self.checkpoint_dir, ckpt)

        return ckpt

    @staticmethod
    def _score_fn(engine: Engine) -> float:
        """Returns the scoring metric for checkpointing and early stopping

        Args:
            engine (ignite.engine.Engine): The engine that calculates
            the val loss

        Returns:
            (float): The validation loss
        """
        negloss: float = -engine.state.metrics["loss"]

        return negloss

    #parse_batch method is used to extract inputs and targets from the batch
    def parse_batch(
        self: TrainerType, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0], device=self.device, non_blocking=self.non_blocking) #moves the input (batch[0]to the device
        targets = to_device(
            batch[1], device=self.device, non_blocking=self.non_blocking
        ) #moves the target (batch[1]) to the device

        return inputs, targets

    def get_predictions_and_targets(
        self: TrainerType, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        inputs, targets = self.parse_batch(batch) #extracts inputs and targets from the batch
        y_pred,mask_txt,mask_au,mask_vi = self.model(inputs, self.enable_plot_embeddings) #predicts the output

        return y_pred, targets,mask_txt,mask_au,mask_vi #returns the predicted output and the target

    def train_step(
        self: TrainerType, engine: Engine, batch: List[torch.Tensor]
    ) -> float:
        self.model.train()
        y_pred, targets,_,_,_ = self.get_predictions_and_targets(batch)
        loss = self.loss_fn(y_pred, targets)  # type: ignore
        loss = loss / self.accumulation_steps #scales the loss by the accumulation steps
        loss.backward(retain_graph=self.retain_graph) #backpropagates the loss, retain_graph flag determines whether the computational graph is retained

        if self.lr_scheduler is not None: #prints the learning rate every 128 epochs if we have a scheduler
            if (engine.state.iteration - 1) % 128 == 0:
                print("LR = {}".format(self.optimizer.param_groups[0]["lr"]))
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5) #clips the gradient norm to prevent exploding gradient

        # if we have accumulated enough gradients, we step the optimizer and zero the gradients
        if (self.trainer.state.iteration + 1) % self.accumulation_steps == 0:
            self.optimizer.step()  # type: ignore
            self.optimizer.zero_grad()
        loss_value: float = loss.item()

        return loss_value #returns the loss value for the current batch
    
    #defines the evaluation step to be executed for each batch during evaluation.
    def eval_step(
        self: TrainerType, engine: Engine, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        self.model.eval() #sets the model to evaluation mode
        with torch.no_grad():
            y_pred, targets,_,_,_ = self.get_predictions_and_targets(batch)

            return y_pred, targets

    #predict method is used to predict the output of the model on the given dataloader
    def predict(self: TrainerType, dataloader: DataLoader) -> State:
        predictions, targets,masks_txt,masks_au,masks_vi = [],[],[],[],[]

        for batch in dataloader:
            self.model.eval()
            with torch.no_grad():
                pred, targ,mask_txt,mask_au,mask_vi = self.get_predictions_and_targets(batch)
                predictions.append(pred)
                targets.append(targ)
                masks_txt.append(mask_txt)
                masks_au.append(mask_au)
                masks_vi.append(mask_vi)

        return predictions, targets,masks_txt,masks_au,masks_vi
    def set_mask_index(self, new_mask_index):
        """
        Updates the mask_index for all FeedbackUnit instances.
        
        Args:
            new_mask_index (int): New mask index value (1 to 5).
        """
        self.model.set_mask_index(new_mask_index)
    def set_mask_dropout(self, new_mask_dropout):
        """Updates mask_dropout for all Feedback ."""
        self.model.set_mask_dropout(new_mask_dropout)
    #Defines the training loop
    def fit(
        self: TrainerType,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
    ) -> State:
        print(
            "Trainer will run for\n"
            f"model: {self.model}\n"
            f"optimizer: {self.optimizer}\n"
            f"loss: {self.loss_fn}"
        )
        #attaches the evaluation handler to the trainer, train_evaluator and valid_evaluator
        self.val_handler.attach(
            self.trainer, self.train_evaluator, train_loader, validation=False
        )
        self.val_handler.attach(
            self.trainer, self.valid_evaluator, val_loader, validation=True
        )
        self.model.zero_grad() #zeros the gradients
        # self.valid_evaluator.run(val_loader) #runs the validation evaluator on the validation loader
        self.trainer.run(train_loader, max_epochs=epochs) #runs the trainer on the training loader for the specified number of epochs

    #overfit_single_batch method is used to overfit the model on a single batch, to ensure that the model is learning
    def overfit_single_batch(self: TrainerType, train_loader: DataLoader) -> State:
        single_batch = [next(iter(train_loader))] #choose a single batch from the training loader

        if self.trainer.has_event_handler(self.val_handler, Events.EPOCH_COMPLETED): #removes the validation handler if it is already attached (not needed in overfit single batch)
            self.trainer.remove_event_handler(self.val_handler, Events.EPOCH_COMPLETED)

        self.val_handler.attach(  # type: ignore
            self.trainer,
            self.train_evaluator,
            single_batch,
            validation=False,
        )
        out = self.trainer.run(single_batch, max_epochs=100)

        return out
    
    # perform a debug training run using a subset of the data
    def fit_debug(
        self: TrainerType, train_loader: DataLoader, val_loader: DataLoader
    ) -> State:
        train_loader = iter(train_loader)  # type: ignore
        train_subset = [next(train_loader), next(train_loader)]  # type: ignore
        val_loader = iter(val_loader)  # type: ignore
        val_subset = [next(val_loader), next(val_loader)]  # type: ignore
        out = self.fit(train_subset, val_subset, epochs=6)  # type: ignore

        return out
    #Defines a private method to attach the checkpoint handler to the validation evaluator.
    def _attach_checkpoint(self: TrainerType) -> TrainerType:
        ckpt = {"model": self.model, "optimizer": self.optimizer} #dictionary containing the model and optimizer

        if self.checkpoint_dir is not None:
            self.valid_evaluator.add_event_handler( #t after every completion of the validation evaluator, a checkpoint will be saved based on the current state.
                Events.COMPLETED, self.checkpoint, ckpt
            )

        return self
    #sets up running averages, progress bars, attaches early stopping to validation evaluator, attaches checkpointing, 
    # and adds a handler for graceful exit in case of exceptions like KeyboardInterrupt.
    def attach(self: TrainerType) -> TrainerType:
        ra = RunningAverage(output_transform=lambda x: x) #running average of the loss
        ra.attach(self.trainer, "Train Loss") #attaches the running average to the trainer
        self.pbar.attach(self.trainer, ["Train Loss"]) #attaches the progress bar to the trainer
        self.val_pbar.attach(self.train_evaluator)
        self.val_pbar.attach(self.valid_evaluator)
        self.valid_evaluator.add_event_handler(Events.COMPLETED, self.early_stop)#attaches the early stopping handler to the validation evaluator
        self = self._attach_checkpoint()

        def graceful_exit(engine, e):#graceful exit in case of exceptions like KeyboardInterrupt
            if isinstance(e, KeyboardInterrupt):
                engine.terminate()
                print("CTRL-C caught. Exiting gracefully...")
            else:
                raise (e)

        self.trainer.add_event_handler(Events.EXCEPTION_RAISED, graceful_exit)
        self.train_evaluator.add_event_handler(Events.EXCEPTION_RAISED, graceful_exit)
        self.valid_evaluator.add_event_handler(Events.EXCEPTION_RAISED, graceful_exit)

        return self


class MOSEITrainer(Trainer):#inherits from the Trainer class and specialises 2 functions for MOSEI dataset
    def parse_batch(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = {
            k: to_device(v, device=self.device, non_blocking=self.non_blocking)
            for k, v in batch[0].items()
        }
        targets = to_device(
            batch[1], device=self.device, non_blocking=self.non_blocking
        )

        return inputs, targets

    def get_predictions_and_targets(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        inputs, targets = self.parse_batch(batch)
        #y_pred = self.model(inputs)
        y_pred, mask_txt, mask_au, mask_vi, *rest = self.model(inputs, self.enable_plot_embeddings)

        y_pred = y_pred.squeeze()
        targets = targets.squeeze()


        return y_pred, targets, mask_txt,mask_au,mask_vi
