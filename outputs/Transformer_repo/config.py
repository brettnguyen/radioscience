"""config.py

This module contains the central configuration for the Transformer experiments.
All modules (dataset_loader, model, trainer, evaluation, etc.) import this file
to ensure consistency and reproducibility across hyperparameters, training settings,
model architecture parameters, evaluation metrics, and hardware specifications.

The configuration parameters below are strictly derived from the provided config.yaml,
the reproduction plan, and the "Attention Is All You Need" paper.

Configuration Structure:
  - TrainingConfig: Settings for training such as total_steps, batch token sizes,
    warmup_steps, and checkpoint intervals.
  - OptimizerConfig: Hyperparameters for the Adam optimizer and its learning rate schedule.
  - ModelConfig: Architecture details (encoder/decoder layers, model dimensions, attention,
    dropout, label smoothing, embedding sharing, and positional encoding).
  - EvaluationConfig: Inference and evaluation settings, including beam search parameters
    and metrics (BLEU for translation and F1 for parsing).
  - HardwareConfig: GPU-related settings used during experiments.
  - Config: Master configuration that aggregates all individual configuration settings.
  
All default values are set explicitly based on the specifications from config.yaml.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class BatchTokenSizeConfig:
    source: int = 25000  # Approximately 25,000 source tokens per batch.
    target: int = 25000  # Approximately 25,000 target tokens per batch.

@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int = 100000  # Base model training steps (~12 hours on 8 P100 GPUs).
    batch_token_size: BatchTokenSizeConfig = BatchTokenSizeConfig()
    warmup_steps: int = 4000  # Linear warmup period for learning rate.
    checkpoint_interval: str = "Every 10 minutes"  # Checkpoint saving frequency.

@dataclass(frozen=True)
class OptimizerConfig:
    type: str = "adam"  # Optimizer type.
    beta1: float = 0.9  # Adam beta1 parameter.
    beta2: float = 0.98  # Adam beta2 parameter.
    epsilon: float = 1e-9  # Adam epsilon parameter.
    learning_rate_schedule: str = (
        "lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))"
    )
    # Learning rate is governed by d_model (512 for base model) and warmup_steps.

@dataclass(frozen=True)
class ModelConfig:
    type: str = "transformer_base"  # Model type identifier.
    encoder_layers: int = 6  # Number of identical encoder layers.
    decoder_layers: int = 6  # Number of identical decoder layers.
    d_model: int = 512  # Model dimension for embeddings and sub-layers.
    d_ff: int = 2048  # Inner dimension for the feed-forward network.
    num_heads: int = 8  # Number of parallel attention heads.
    d_k: int = 64  # Dimension per head for keys and queries (512/8).
    d_v: int = 64  # Dimension per head for values (512/8).
    dropout: float = 0.1  # Dropout rate applied to sub-layer outputs and embeddings.
    label_smoothing: float = 0.1  # Label smoothing parameter.
    share_embeddings: bool = True  # Share weights between encoder/decoder embeddings and output projection.
    positional_encoding: str = "sinusoidal"  # Type of positional encoding to use.

@dataclass(frozen=True)
class MetricsConfig:
    translation: str = "bleu"  # Metric for translation evaluation.
    parsing: str = "f1"       # Metric for constituency parsing evaluation.

@dataclass(frozen=True)
class EvaluationConfig:
    beam_size: int = 4  # Beam search size for translation inference.
    length_penalty: float = 0.6  # Length penalty factor used during beam search.
    max_output_length: str = "input_length + 50"  # Maximum output length during inference.
    metrics: MetricsConfig = MetricsConfig()  # Evaluation metrics configuration.

@dataclass(frozen=True)
class HardwareConfig:
    gpus: int = 8  # Number of GPUs available for training.
    gpu_type: str = "P100"  # GPU type as reported in the experiments.

@dataclass(frozen=True)
class Config:
    training: TrainingConfig = TrainingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    model: ModelConfig = ModelConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    hardware: HardwareConfig = HardwareConfig()
    notes: List[str] = field(default_factory=lambda: [
        "For the big model configuration, update total_steps to 300000 and adjust d_model and d_ff accordingly.",
        "Batching is performed based on token counts rather than fixed sample numbers to account for variable sequence lengths.",
        "Tokenizer configuration (BPE vs. word-piece, vocabulary sizes) should be set in dataset_loader.py as per task requirements.",
        "Training checkpoint averaging (last 5 for base models; last 20 for big models) should be managed via the trainer's checkpoint manager."
    ])

# Global configuration instance accessible across modules.
global_config: Config = Config()

def get_config() -> Config:
    """Returns the global configuration object for the entire project."""
    return global_config
