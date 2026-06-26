"""main.py

This is the entry point for running the Transformer experiments as described in
"Attention Is All You Need".

Workflow:
1. Load the configuration from config.yaml via config.py.
2. Initialize the DatasetLoader to load translation and parsing data.
3. Instantiate the TransformerModel with configuration-defined parameters.
4. Create a Trainer instance to run the training loop using the translation DataLoaders.
5. After training is complete, run evaluation on both translation and parsing tasks
   using the Evaluation module.
6. Log and output final evaluation metrics.

All configuration parameters (hyperparameters, token batch sizes, checkpoints, learning
rate schedule, evaluation settings, etc.) are derived from config.yaml.
"""

import logging
import os
from dataclasses import asdict
from typing import Dict

import torch

# Import configuration, dataset loader, model, trainer, and evaluation modules.
from config import get_config
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluation import Evaluation

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main() -> None:
    """
    Main function that orchestrates the data loading, model training, and evaluation for the Transformer
    experiments.
    """
    # -----------------------------------------------------------------------------
    # 1. Configuration: Load the global configuration.
    # -----------------------------------------------------------------------------
    config_object = get_config()  # This is a Config dataclass instance.
    # For modules that expect dictionary access (e.g., DatasetLoader), convert config to dict.
    config_dict: Dict = asdict(config_object)
    logging.info("Configuration loaded from config.yaml.")

    # -----------------------------------------------------------------------------
    # 2. Data Loading: Initialize DatasetLoader and load datasets.
    # -----------------------------------------------------------------------------
    dataset_loader = DatasetLoader(config_dict)
    # Load translation training and validation DataLoaders.
    translation_train_loader, translation_val_loader = dataset_loader.load_translation_data()
    # Load parsing training and validation DataLoaders.
    parsing_train_loader, parsing_val_loader = dataset_loader.load_parsing_data()
    logging.info("Datasets loaded successfully for translation and parsing tasks.")

    # -----------------------------------------------------------------------------
    # 3. Model Instantiation: Create TransformerModel.
    # -----------------------------------------------------------------------------
    # The model requires a parameter dictionary. Use default vocabulary size of 37000.
    model_params: Dict = {"vocab_size": 37000}
    transformer_model = TransformerModel(model_params)
    logging.info("Transformer model instantiated with the following parameters:")
    logging.info(f"  Encoder layers: {config_object.model.encoder_layers}, "
                 f"Decoder layers: {config_object.model.decoder_layers}, "
                 f"d_model: {config_object.model.d_model}, "
                 f"num_heads: {config_object.model.num_heads}")
    
    # -----------------------------------------------------------------------------
    # 4. Trainer Setup: Initialize Trainer with the translation DataLoaders.
    # -----------------------------------------------------------------------------
    # Use translation data for training.
    data_loaders = {"train": translation_train_loader, "val": translation_val_loader}
    trainer = Trainer(transformer_model, data_loaders, config_object)
    logging.info("Trainer initialized. Starting training process...")

    # -----------------------------------------------------------------------------
    # 5. Training Process: Run the training loop.
    # -----------------------------------------------------------------------------
    trainer.train()
    logging.info("Training completed successfully.")

    # -----------------------------------------------------------------------------
    # 6. Evaluation: Run evaluation on both translation and parsing test sets.
    # -----------------------------------------------------------------------------
    evaluator = Evaluation(transformer_model, config_object)
    logging.info("Starting translation evaluation...")
    translation_results = evaluator.evaluate_translation(translation_val_loader)
    logging.info(f"Translation Evaluation Results: {translation_results}")

    logging.info("Starting parsing evaluation...")
    parsing_results = evaluator.evaluate_parsing(parsing_val_loader)
    logging.info(f"Parsing Evaluation Results: {parsing_results}")

    # -----------------------------------------------------------------------------
    # 7. Final Output
    # -----------------------------------------------------------------------------
    logging.info("Final Metrics:")
    logging.info(f"  Translation BLEU Score: {translation_results.get('BLEU', 'N/A'):.2f}")
    logging.info(f"  Parsing F1 Score: {parsing_results.get('F1', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
