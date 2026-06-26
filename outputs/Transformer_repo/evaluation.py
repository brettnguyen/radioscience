"""evaluation.py

This module defines the Evaluation class responsible for running inference on
machine translation and English constituency parsing tasks using beam search decoding.
It computes BLEU scores for translation tasks using SacreBLEU and F1 scores for parsing tasks
using an unlabeled bracket F1 scorer. All evaluation parameters (beam search settings,
maximum output lengths, etc.) are driven by the configuration in config.yaml.
"""

import math
import logging
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from sacrebleu import corpus_bleu

# Import the TransformerModel and configuration.
from model import TransformerModel
from config import get_config

# Import Hugging Face Tokenizer for decoding token ids.
from tokenizers import Tokenizer

# Set up basic logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class Evaluation:
    """
    Evaluation class for the Transformer model.

    Provides methods to evaluate the model on machine translation tasks (computing BLEU)
    and on English constituency parsing tasks (computing bracket-based F1).
    """
    def __init__(self, model: TransformerModel, config: object) -> None:
        """
        Initializes the Evaluation object.

        Args:
            model (TransformerModel): The trained Transformer model.
            config (object): The configuration object (from config.py).
        """
        self.model: TransformerModel = model
        self.config = config
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizers for translation and parsing from disk.
        try:
            self.translation_tokenizer: Tokenizer = Tokenizer.from_file("./tokenizers/translation_tokenizer.json")
        except Exception as e:
            logging.warning(f"Failed to load translation tokenizer: {e}")
            self.translation_tokenizer = None

        try:
            self.parsing_tokenizer: Tokenizer = Tokenizer.from_file("./tokenizers/parsing_tokenizer.json")
        except Exception as e:
            logging.warning(f"Failed to load parsing tokenizer: {e}")
            self.parsing_tokenizer = None

        # Set special token IDs using the translation tokenizer if available.
        self.start_token: int = self._get_token_id(self.translation_tokenizer, "[CLS]", default=1)
        self.end_token: int = self._get_token_id(self.translation_tokenizer, "[SEP]", default=2)

        # Get evaluation beam parameters for translation from configuration.
        self.beam_size_translation: int = (
            self.config.evaluation.beam_size if hasattr(self.config.evaluation, "beam_size") else 4
        )
        self.length_penalty_translation: float = (
            self.config.evaluation.length_penalty if hasattr(self.config.evaluation, "length_penalty") else 0.6
        )

    def _get_token_id(self, tokenizer: Tokenizer, token: str, default: int) -> int:
        """
        Utility function to get the token ID from a tokenizer.

        Args:
            tokenizer (Tokenizer): The Hugging Face tokenizer.
            token (str): The token string.
            default (int): Default token id if tokenizer is None or token is not found.

        Returns:
            int: The token id.
        """
        if tokenizer is not None:
            try:
                token_id = tokenizer.token_to_id(token)
                return token_id if token_id is not None else default
            except Exception:
                return default
        return default

    def _beam_search(
        self,
        src: torch.Tensor,
        beam_size: int,
        max_length: int,
        length_penalty: float,
        start_token: int,
        end_token: int
    ) -> List[int]:
        """
        Performs beam search decoding on a single source sequence.

        Args:
            src (torch.Tensor): Source tensor of shape (1, src_seq_len).
            beam_size (int): Beam search width.
            max_length (int): Maximum allowed length for the decoded sequence.
            length_penalty (float): Length penalty factor.
            start_token (int): Start-of-sequence token id.
            end_token (int): End-of-sequence token id.

        Returns:
            List[int]: The best decoded sequence (list of token ids).
        """
        # Encode source only once.
        src = src.to(self.device)
        encoder_output = self.model.encode(src)

        # Each beam candidate is a tuple: (sequence, cumulative_score)
        BeamEntry = Tuple[List[int], float]
        beam: List[BeamEntry] = [([start_token], 0.0)]

        for _ in range(max_length - 1):
            new_beam: List[BeamEntry] = []
            for seq, score in beam:
                # If already completed, carry over.
                if seq[-1] == end_token:
                    new_beam.append((seq, score))
                    continue
                tgt_seq = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    # Model outputs logits of shape (1, seq_len, vocab_size)
                    logits = self.model(src, tgt_seq)
                # Take logits for the last generated token.
                last_logits = logits[0, -1, :]
                log_probs = torch.log_softmax(last_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                for log_prob, token_id in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                    new_seq = seq + [token_id]
                    new_score = score + log_prob
                    new_beam.append((new_seq, new_score))
            if not new_beam:
                break
            # Sort candidates by raw score (descending) and prune to beam_size.
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_size]
            # If all candidates have ended, stop early.
            if all(seq[-1] == end_token for seq, _ in beam):
                break

        # Select the best candidate with adjusted score accounting for length penalty.
        def adjusted_score(entry: BeamEntry) -> float:
            seq, score = entry
            length = len(seq)
            penalty = ((5 + length) / 6) ** length_penalty
            return score / penalty

        best_seq, _ = max(beam, key=adjusted_score)
        return best_seq

    def _post_process_prediction(self, token_ids: List[int], start_token: int, end_token: int) -> List[int]:
        """
        Cleans up the predicted token sequence by removing the start token and truncating at the end token.

        Args:
            token_ids (List[int]): Raw predicted token sequence.
            start_token (int): Start token id.
            end_token (int): End token id.

        Returns:
            List[int]: Processed token sequence.
        """
        if token_ids and token_ids[0] == start_token:
            token_ids = token_ids[1:]
        if end_token in token_ids:
            index = token_ids.index(end_token)
            token_ids = token_ids[:index]
        return token_ids

    def _decode_tokens(self, token_ids: List[int], tokenizer: Tokenizer) -> str:
        """
        Converts a list of token ids into a human-readable string using the provided tokenizer.

        Args:
            token_ids (List[int]): List of token ids.
            tokenizer (Tokenizer): The Hugging Face tokenizer instance.

        Returns:
            str: Decoded text.
        """
        if tokenizer is not None:
            try:
                return tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception as e:
                logging.warning(f"Tokenizer decode error: {e}")
        # Fallback: join token ids as strings.
        return " ".join(map(str, token_ids))

    def _extract_brackets(self, tree_str: str) -> set:
        """
        Extracts bracket spans from a linearized parse tree string.

        Args:
            tree_str (str): Parse tree in bracketed notation.

        Returns:
            set: A set of tuples representing bracket spans.
        """
        tokens = tree_str.replace("(", " ( ").replace(")", " ) ").split()
        spans = set()
        stack = []
        for idx, token in enumerate(tokens):
            if token == "(":
                stack.append(idx)
            elif token == ")" and stack:
                start = stack.pop()
                spans.add((start, idx))
        return spans

    def _compute_bracket_f1(self, pred_tree: str, gold_tree: str) -> float:
        """
        Computes the unlabeled bracket F1 score between predicted and gold parse trees.

        Args:
            pred_tree (str): Predicted parse tree string.
            gold_tree (str): Gold parse tree string.

        Returns:
            float: F1 score.
        """
        pred_brackets = self._extract_brackets(pred_tree)
        gold_brackets = self._extract_brackets(gold_tree)
        if not pred_brackets or not gold_brackets:
            return 0.0
        correct = len(pred_brackets.intersection(gold_brackets))
        precision = correct / len(pred_brackets) if pred_brackets else 0.0
        recall = correct / len(gold_brackets) if gold_brackets else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def evaluate_translation(self, test_data: DataLoader) -> Dict[str, float]:
        """
        Evaluates the Transformer model on a machine translation test set.
        Decoding is performed using beam search, and the BLEU score is computed via SacreBLEU.

        Args:
            test_data (DataLoader): DataLoader for the translation test data.

        Returns:
            dict: Dictionary containing the BLEU score.
        """
        self.model.eval()
        predictions: List[str] = []
        references: List[str] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                # Each batch is a tuple of (src, tgt)
                src_batch, tgt_batch = batch
                batch_size = src_batch.size(0)
                for i in range(batch_size):
                    src_sample = src_batch[i:i+1]  # (1, seq_len)
                    tgt_sample = tgt_batch[i]      # gold target tokens
                    # Maximum length is computed as input length plus 50.
                    src_length = src_sample.size(1)
                    max_length = src_length + 50

                    predicted_ids = self._beam_search(
                        src=src_sample,
                        beam_size=self.beam_size_translation,
                        max_length=max_length,
                        length_penalty=self.length_penalty_translation,
                        start_token=self.start_token,
                        end_token=self.end_token
                    )
                    processed_ids = self._post_process_prediction(predicted_ids, self.start_token, self.end_token)
                    predicted_text = self._decode_tokens(processed_ids, self.translation_tokenizer)

                    # Process the reference target tokens.
                    tgt_list = tgt_sample.tolist()
                    processed_ref_ids = self._post_process_prediction(tgt_list, self.start_token, self.end_token)
                    reference_text = self._decode_tokens(processed_ref_ids, self.translation_tokenizer)

                    predictions.append(predicted_text)
                    references.append(reference_text)

                    logging.info(
                        f"Translation sample {batch_idx}-{i}: Predicted: '{predicted_text}' | Reference: '{reference_text}'"
                    )

        bleu_score = corpus_bleu(predictions, [references])
        logging.info(f"Translation BLEU score: {bleu_score.score:.2f}")
        return {"BLEU": bleu_score.score}

    def evaluate_parsing(self, test_data: DataLoader) -> Dict[str, float]:
        """
        Evaluates the Transformer model on an English constituency parsing test set.
        Decoding is performed using beam search with parsing-specific parameters 
        (beam_size=21, length_penalty=0.3, max length = input length + 300), and F1 is computed using bracket overlap.

        Args:
            test_data (DataLoader): DataLoader for the parsing test data.

        Returns:
            dict: Dictionary containing the average F1 score.
        """
        self.model.eval()
        parsing_beam_size: int = 21
        parsing_length_penalty: float = 0.3

        f1_scores: List[float] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                src_batch, tgt_batch = batch
                batch_size = src_batch.size(0)
                for i in range(batch_size):
                    src_sample = src_batch[i:i+1]
                    tgt_sample = tgt_batch[i]  # gold parse tokens
                    src_length = src_sample.size(1)
                    max_length = src_length + 300

                    predicted_ids = self._beam_search(
                        src=src_sample,
                        beam_size=parsing_beam_size,
                        max_length=max_length,
                        length_penalty=parsing_length_penalty,
                        start_token=self.start_token,
                        end_token=self.end_token
                    )
                    processed_ids = self._post_process_prediction(predicted_ids, self.start_token, self.end_token)
                    predicted_parse = self._decode_tokens(processed_ids, self.parsing_tokenizer)

                    gold_ids = tgt_sample.tolist()
                    processed_gold_ids = self._post_process_prediction(gold_ids, self.start_token, self.end_token)
                    gold_parse = self._decode_tokens(processed_gold_ids, self.parsing_tokenizer)

                    f1 = self._compute_bracket_f1(predicted_parse, gold_parse)
                    f1_scores.append(f1)

                    logging.info(
                        f"Parsing sample {batch_idx}-{i}: F1: {f1:.4f} | Predicted: '{predicted_parse}' | Gold: '{gold_parse}'"
                    )

        average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        logging.info(f"Parsing average F1 score: {average_f1:.4f}")
        return {"F1": average_f1}
