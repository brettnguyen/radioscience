"""model.py

This module implements the TransformerModel as described in "Attention Is All You Need".
It aggregates embeddings, sinusoidal positional encodings, encoder and decoder stacks,
multi-head attention, position-wise feed-forward networks, and the final output projection.
All hyperparameters and settings are read from the configuration (config.yaml) via config.py.

Classes:
  - PositionalEncoding: Implements sinusoidal positional encodings.
  - MultiHeadAttention: Implements scaled dot-product multi-head attention.
  - PositionwiseFeedForward: Implements the two-layer feed-forward network.
  - EncoderLayer: A single encoder block with self-attention and feed-forward sub-layers.
  - DecoderLayer: A single decoder block with masked self-attention, encoder-decoder attention, and feed-forward sub-layers.
  - Encoder: Stacks multiple EncoderLayer modules.
  - Decoder: Stacks multiple DecoderLayer modules.
  - TransformerModel: The complete Transformer model with shared embedding,
      encoder, decoder, and final projection.
      
All default values are set explicitly based on config.yaml and the original paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config

# ---------------------------
# Positional Encoding Module
# ---------------------------
class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for non-recurrent neural networks.
    The positional encodings are computed once in log space.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """
        Args:
            d_model (int): Model dimensionality.
            dropout (float): Dropout probability after adding positional encodings.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sinusoidal function to even indices.
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine function to odd indices.
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Output tensor with positional encodings applied.
        """
        seq_len = x.size(1)
        # Add positional encoding and apply dropout.
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# ---------------------------
# Multi-Head Attention Module
# ---------------------------
class MultiHeadAttention(nn.Module):
    """
    Implements the multi-head attention mechanism as described in the paper.
    Projects queries, keys, and values, computes scaled dot-product attention for each head,
    and concatenates the results.
    """
    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            d_k (int): Dimensionality of each head for keys and queries.
            d_v (int): Dimensionality of each head for values.
            dropout (float): Dropout probability on attention weights.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.linear_q = nn.Linear(d_model, num_heads * d_k)
        self.linear_k = nn.Linear(d_model, num_heads * d_k)
        self.linear_v = nn.Linear(d_model, num_heads * d_v)
        self.linear_out = nn.Linear(num_heads * d_v, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query (Tensor): Query tensor of shape (batch_size, query_len, d_model).
            key (Tensor): Key tensor of shape (batch_size, key_len, d_model).
            value (Tensor): Value tensor of shape (batch_size, value_len, d_model).
            mask (Tensor, optional): Boolean mask tensor broadcastable to (batch_size, num_heads, query_len, key_len).
                                      Positions with True are masked out.
        Returns:
            Tensor: Attention output of shape (batch_size, query_len, d_model).
        """
        batch_size = query.size(0)

        # Linear projections and reshape to (batch_size, num_heads, seq_len, d_k/d_v)
        q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # shape: (batch, num_heads, query_len, key_len)
        if mask is not None:
            # Mask out positions where mask is True (i.e., illegal connections)
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # shape: (batch, num_heads, query_len, d_v)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)
        output = self.linear_out(attn_output)
        return output

# ---------------------------
# Position-wise Feed-Forward Module
# ---------------------------
class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network.
    Consists of two linear transformations with a ReLU activation in between.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model (int): Dimensionality of the model.
            d_ff (int): Dimensionality of the feed-forward inner layer.
            dropout (float): Dropout probability.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Output tensor of same shape after feed-forward processing.
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ---------------------------
# Encoder Layer Module
# ---------------------------
class EncoderLayer(nn.Module):
    """
    Implements one layer of the Transformer encoder.
    Each layer has a self-attention sub-layer and a feed-forward sub-layer,
    with residual connections and layer normalization.
    """
    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_k (int): Key and query dimension per head.
            d_v (int): Value dimension per head.
            d_ff (int): Inner layer dimension for the feed-forward network.
            dropout (float): Dropout probability.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Output tensor of same shape after processing.
        """
        # Self-attention sub-layer with residual connection and layer normalization.
        attn_out = self.self_attn(x, x, x, mask=None)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward sub-layer with residual connection and layer normalization.
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# ---------------------------
# Decoder Layer Module
# ---------------------------
class DecoderLayer(nn.Module):
    """
    Implements one layer of the Transformer decoder.
    Each layer has masked self-attention, encoder-decoder attention,
    and a feed-forward sub-layer with residual connections and layer normalization.
    """
    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_k (int): Key and query dimension per head.
            d_v (int): Value dimension per head.
            d_ff (int): Feed-forward inner layer dimension.
            dropout (float): Dropout probability.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): Decoder input of shape (batch_size, tgt_seq_len, d_model)
            encoder_output (Tensor): Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask (Tensor, optional): Mask for target self-attention of shape broadcastable to
                                         (batch_size, num_heads, tgt_seq_len, tgt_seq_len)
        Returns:
            Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention sub-layer.
        self_attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        # Encoder-decoder attention sub-layer.
        enc_dec_attn_out = self.enc_dec_attn(x, encoder_output, encoder_output, mask=None)
        x = self.norm2(x + self.dropout(enc_dec_attn_out))
        # Feed-forward sub-layer.
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

# ---------------------------
# Encoder Stack Module
# ---------------------------
class Encoder(nn.Module):
    """
    Stacks multiple EncoderLayer modules to form the complete encoder.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        """
        Args:
            num_layers (int): Number of encoder layers.
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_k (int): Key/query dimension per head.
            d_v (int): Value dimension per head.
            d_ff (int): Feed-forward inner dimension.
            dropout (float): Dropout probability.
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_k, d_v, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Output tensor of same shape after all encoder layers.
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# ---------------------------
# Decoder Stack Module
# ---------------------------
class Decoder(nn.Module):
    """
    Stacks multiple DecoderLayer modules to form the complete decoder.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        """
        Args:
            num_layers (int): Number of decoder layers.
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_k (int): Key/query dimension per head.
            d_v (int): Value dimension per head.
            d_ff (int): Feed-forward inner dimension.
            dropout (float): Dropout probability.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_k, d_v, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): Decoder input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output (Tensor): Encoder output tensor of shape (batch_size, src_seq_len, d_model)
            tgt_mask (Tensor, optional): Target mask tensor for self-attention.
        Returns:
            Tensor: Decoder output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask)
        return self.norm(x)

# ---------------------------
# Helper Function for Masking
# ---------------------------
def generate_subsequent_mask(size: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf (masked) with zeros in the lower triangle.
    This is used in the decoder to mask future positions.
    
    Args:
        size (int): The size of the sequence.
    Returns:
        Tensor: A boolean mask tensor of shape (size, size) where True indicates positions to be masked.
    """
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    return mask

# ---------------------------
# Transformer Model
# ---------------------------
class TransformerModel(nn.Module):
    """
    TransformerModel aggregates the components of the Transformer:
      - Shared token embeddings.
      - Sinusoidal positional encoding.
      - Encoder stack.
      - Decoder stack.
      - Final output projection tied to embeddings if configured.
    
    Public Methods:
      - __init__(params: dict): Initializes the model with configuration parameters.
      - encode(src: Tensor) -> Tensor: Processes source sequences.
      - decode(encoder_output: Tensor, tgt: Tensor) -> Tensor: Processes target sequences.
      - forward(src: Tensor, tgt: Tensor) -> Tensor: Computes logits over the vocabulary.
    """
    def __init__(self, params: dict) -> None:
        """
        Initializes the Transformer model.
        
        Args:
            params (dict): Dictionary of parameters. Must include 'vocab_size'; if absent, defaults to 37000.
        """
        super(TransformerModel, self).__init__()
        # Get model configuration from global config.
        global_config = get_config()
        model_config = global_config.model

        self.encoder_layers: int = model_config.encoder_layers
        self.decoder_layers: int = model_config.decoder_layers
        self.d_model: int = model_config.d_model
        self.d_ff: int = model_config.d_ff
        self.num_heads: int = model_config.num_heads
        self.d_k: int = model_config.d_k
        self.d_v: int = model_config.d_v
        self.dropout_rate: float = model_config.dropout
        self.label_smoothing: float = model_config.label_smoothing
        self.share_embeddings: bool = model_config.share_embeddings
        self.positional_encoding_type: str = model_config.positional_encoding

        # Set vocabulary size (default to 37000 if not provided).
        self.vocab_size: int = params.get("vocab_size", 37000)

        # Shared embedding layer for source, target, and output projection.
        self.embedding: nn.Embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Positional encoding module (sinusoidal).
        if self.positional_encoding_type.lower() == "sinusoidal":
            self.positional_encoding: PositionalEncoding = PositionalEncoding(self.d_model, self.dropout_rate)
        else:
            raise ValueError("Only 'sinusoidal' positional encoding is implemented.")

        # Encoder and Decoder stacks.
        self.encoder: Encoder = Encoder(
            num_layers=self.encoder_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_ff=self.d_ff,
            dropout=self.dropout_rate
        )
        self.decoder: Decoder = Decoder(
            num_layers=self.decoder_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_ff=self.d_ff,
            dropout=self.dropout_rate
        )

        # Final linear projection to vocabulary logits.
        self.output_projection: nn.Linear = nn.Linear(self.d_model, self.vocab_size)
        if self.share_embeddings:
            self.output_projection.weight = self.embedding.weight

        # Scaling factor for embeddings.
        self.scale: float = math.sqrt(self.d_model)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encodes the source sequence.
        
        Args:
            src (Tensor): Source tokens of shape (batch_size, src_seq_len)
        Returns:
            Tensor: Encoder output of shape (batch_size, src_seq_len, d_model)
        """
        # Embed and scale.
        src_emb = self.embedding(src) * self.scale
        # Add positional encodings.
        src_emb = self.positional_encoding(src_emb)
        # Pass through the encoder stack.
        encoder_output = self.encoder(src_emb)
        return encoder_output

    def decode(self, encoder_output: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Decodes the target sequence using encoder outputs.
        
        Args:
            encoder_output (Tensor): Encoder outputs of shape (batch_size, src_seq_len, d_model)
            tgt (Tensor): Target tokens of shape (batch_size, tgt_seq_len)
        Returns:
            Tensor: Decoder output of shape (batch_size, tgt_seq_len, d_model)
        """
        tgt_emb = self.embedding(tgt) * self.scale
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_seq_len = tgt.size(1)
        # Generate subsequent mask to prevent attending to future tokens.
        subsequent_mask = generate_subsequent_mask(tgt_seq_len).to(tgt.device)
        # Expand mask dimensions to be broadcastable to (batch_size, num_heads, tgt_seq_len, tgt_seq_len).
        tgt_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask)
        return decoder_output

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        
        Args:
            src (Tensor): Source tokens of shape (batch_size, src_seq_len)
            tgt (Tensor): Target tokens of shape (batch_size, tgt_seq_len)
        Returns:
            Tensor: Logits over the vocabulary of shape (batch_size, tgt_seq_len, vocab_size)
        """
        encoder_output = self.encode(src)
        decoder_output = self.decode(encoder_output, tgt)
        logits = self.output_projection(decoder_output)
        return logits
