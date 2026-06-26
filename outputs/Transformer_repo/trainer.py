"""trainer.py

This module implements the Trainer class which manages the training loop for the Transformer model as described in
"Attention Is All You Need". It handles forward and backward passes, label-smoothed loss computation, learning rate 
scheduling using a custom inverse square-root scheduler with warmup, checkpoint management, and periodic validation.

The Trainer requires:
  - A TransformerModel instance (imported from model.py)
  - A dictionary of DataLoader objects (with at least "train" and "val" keys) from dataset_loader.py
  - A configuration object loaded via config.py

It also defines a CheckpointManager class for saving and loading checkpoints.

All hyperparameters (e.g., total_steps, warmup_steps, d_model, dropout, etc.) are sourced directly from the configuration.
Default values are set and strong type annotations are used throughout.
"""

import math
import os
import time
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from config import get_config
# Note: model.py defines the TransformerModel.
from model import TransformerModel

# Set up basic logging.
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
)


class CheckpointManager:
    """
    CheckpointManager handles saving and loading model checkpoints.
    
    Methods:
      - save_checkpoint(model, optimizer, scheduler, step): Saves checkpoint to disk.
      - load_checkpoint(checkpoint_path, model, optimizer, scheduler): Loads checkpoint from disk.
    """
    def __init__(self, checkpoint_dir: str = "./checkpoints") -> None:
        self.checkpoint_dir: str = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self, 
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        scheduler: LambdaLR, 
        step: int
    ) -> None:
        """
        Saves a checkpoint with model state, optimizer state, scheduler state, and current step.
        """
        checkpoint_path: str = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.pt")
        checkpoint_data: Dict[str, Any] = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Checkpoint saved at step {step} to {checkpoint_path}")
    
    def load_checkpoint(
        self, 
        checkpoint_path: str, 
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        scheduler: LambdaLR
    ) -> int:
        """
        Loads checkpoint data from file into model, optimizer, and scheduler.
        Returns the loaded step number.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        loaded_step = checkpoint_data.get("step", 0)
        logging.info(f"Loaded checkpoint from {checkpoint_path} at step {loaded_step}")
        return loaded_step


class Trainer:
    """
    Trainer class orchestrates the training loop for the Transformer model.

    Attributes:
      - model (TransformerModel): The Transformer model to be trained.
      - data_loaders (dict): Dictionary containing training and validation DataLoaders.
      - config: Configuration object loaded via get_config().
      - optimizer (optim.Optimizer): Adam optimizer with hyperparameters from config.
      - scheduler (LambdaLR): Custom learning rate scheduler implementing:
                lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
      - checkpoint_manager (CheckpointManager): Manages checkpoint saving.
      - device (torch.device): Training device (GPU if available, else CPU).
      - total_steps (int): Total training steps from configuration.
      - warmup_steps (int): Warmup steps for learning rate schedule.
      - last_checkpoint_time (float): Wall-clock time of the last checkpoint save.
      - checkpoint_interval (int): Interval in seconds to trigger checkpointing.
    """

    def __init__(self, model: TransformerModel, data_loaders: Dict[str, torch.utils.data.DataLoader], config: Any) -> None:
        """
        Initializes the Trainer with the model, data loaders, and configuration.
        
        Args:
            model (TransformerModel): The Transformer model.
            data_loaders (dict): Dictionary containing 'train' and 'val' DataLoader objects.
            config (Any): The configuration object loaded from config.py.
        """
        self.config = config
        self.model = model
        self.data_loaders = data_loaders

        # Set device based on availability.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logging.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")

        # Retrieve hyperparameters from config.
        self.total_steps: int = self.config.training.total_steps
        self.warmup_steps: int = self.config.training.warmup_steps
        self.d_model: int = self.config.model.d_model
        self.label_smoothing: float = self.config.model.label_smoothing

        # Initialize the Adam optimizer with parameters from config.
        self.optimizer: optim.Optimizer = optim.Adam(
            self.model.parameters(),
            lr=1.0,  # Initial lr will be scaled by scheduler.
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
            eps=self.config.optimizer.epsilon
        )

        # Define the Lambda function for learning rate scheduling.
        self.scheduler: LambdaLR = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: (self.d_model ** (-0.5)) * 
                                     min((max(step, 1)) ** (-0.5), max(step, 1) * (self.warmup_steps ** (-1.5)))
        )

        # Initialize a CheckpointManager.
        self.checkpoint_manager = CheckpointManager()
        # Set checkpoint interval: "Every 10 minutes" -> 600 seconds.
        self.checkpoint_interval: int = 600
        self.last_checkpoint_time: float = time.time()

        # Global training step count.
        self.global_step: int = 0

        logging.info(f"Trainer initialized with total_steps={self.total_steps}, warmup_steps={self.warmup_steps}, d_model={self.d_model}")

    @staticmethod
    def _compute_loss(logits: torch.Tensor, target: torch.Tensor, smoothing: float, pad_index: int = 0) -> torch.Tensor:
        """
        Computes label-smoothed cross-entropy loss.
        
        Args:
            logits (Tensor): Logits of shape (batch_size, seq_len, vocab_size).
            target (Tensor): Target tokens of shape (batch_size, seq_len).
            smoothing (float): Label smoothing constant (e.g., 0.1).
            pad_index (int): Index for padding tokens to be ignored.
        
        Returns:
            Tensor: Scalar loss value.
        """
        # Flatten logits and target for computation.
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)  # shape: (batch_size * seq_len, vocab_size)
        target_flat = target.view(-1)  # shape: (batch_size * seq_len)
        
        # Compute log probabilities.
        log_probs = torch.log_softmax(logits_flat, dim=-1)  # shape: (N, vocab_size)
        
        # Create a one-hot representation of targets.
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(smoothing / (vocab_size - 1))
            # For non-pad tokens, set the target probability.
            non_pad_mask = target_flat.ne(pad_index)
            target_indices = target_flat[non_pad_mask].unsqueeze(1)
            true_dist.index_fill_(1, target_indices, 0.0)
            true_dist[non_pad_mask, target_flat[non_pad_mask]] = 1.0 - smoothing
        
        # Compute KL-divergence loss.
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1)[target_flat.ne(pad_index)])
        return loss

    def train(self) -> None:
        """
        Runs the training loop until the total number of training steps is reached.
        This method handles batching, forward and backward passes, learning rate scheduling,
        logging, and checkpointing.
        """
        self.model.train()
        train_loader = self.data_loaders.get("train")
        if train_loader is None:
            raise ValueError("Training DataLoader not provided in data_loaders with key 'train'.")

        logging.info("Starting training...")
        start_time = time.time()

        # Create an iterator for the training DataLoader.
        data_iter = iter(train_loader)
        while self.global_step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart the DataLoader iterator if the dataset is exhausted.
                data_iter = iter(train_loader)
                batch = next(data_iter)

            src, tgt = batch  # Each batch is a tuple (src, tgt)
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Forward pass.
            self.optimizer.zero_grad()
            logits = self.model(src, tgt)  # logits shape: (batch_size, seq_len, vocab_size)
            loss = self._compute_loss(logits, tgt, self.label_smoothing, pad_index=0)
            
            # Backward pass.
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1

            # Logging training progress every 100 steps.
            if self.global_step % 100 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0.0
                remaining_steps = self.total_steps - self.global_step
                eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else float("inf")
                logging.info(
                    f"Step {self.global_step}/{self.total_steps} | Loss: {loss.item():.4f} | LR: {current_lr:.8f} "
                    f"| {steps_per_sec:.2f} steps/sec | ETA: {eta/60:.2f} min"
                )

            # Checkpointing based on time interval.
            current_time = time.time()
            if current_time - self.last_checkpoint_time >= self.checkpoint_interval:
                self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.scheduler, self.global_step)
                self.last_checkpoint_time = current_time

        logging.info("Training complete.")
        # Final checkpoint save after training.
        self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.scheduler, self.global_step)

    def validate(self) -> Dict[str, float]:
        """
        Evaluates the model on the validation set.
        Returns a dictionary containing validation loss and perplexity.
        """
        self.model.eval()
        val_loader = self.data_loaders.get("val")
        if val_loader is None:
            raise ValueError("Validation DataLoader not provided in data_loaders with key 'val'.")
        
        total_loss: float = 0.0
        token_count: int = 0

        with torch.no_grad():
            for batch in val_loader:
                src, tgt = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                logits = self.model(src, tgt)
                loss = self._compute_loss(logits, tgt, self.label_smoothing, pad_index=0)
                # Count non-padding tokens.
                non_pad = (tgt != 0).sum().item()
                total_loss += loss.item() * non_pad
                token_count += non_pad

        avg_loss = total_loss / token_count if token_count > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss < 300 else float("inf")
        logging.info(f"Validation Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
        self.model.train()  # Return to training mode
        return {"validation_loss": avg_loss, "perplexity": perplexity}
