"""Text Encoder module using BERT for text embedding."""

import torch
import torch.nn as nn
from typing import Union, List
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """
    BERT-based text encoder for converting text descriptions to embeddings.

    Args:
        model_name: Name of pretrained BERT model
        embedding_dim: Output embedding dimension
        max_length: Maximum sequence length
        freeze: Whether to freeze BERT weights
        pooling: Pooling strategy ("cls", "mean", "max", or "concat")
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        max_length: int = 128,
        freeze: bool = True,
        pooling: str = "cls"
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.freeze = freeze
        self.pooling = pooling.lower()

        # Validate pooling mode
        valid_pooling = ["cls", "mean", "max", "concat"]
        if self.pooling not in valid_pooling:
            raise ValueError(f"pooling must be one of {valid_pooling}, got '{self.pooling}'")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            self.bert_model.eval()

        hidden_size = self.bert_model.config.hidden_size

        # Determine output dimension based on pooling mode
        if self.pooling == "concat":
            # Concat: [CLS, mean, max] -> 3x hidden_size
            pooled_dim = hidden_size * 3
        else:
            # cls, mean, max -> hidden_size
            pooled_dim = hidden_size

        # Project to embedding_dim if needed
        self.projection = None if embedding_dim == pooled_dim else nn.Linear(pooled_dim, embedding_dim)

    def _pool_tokens(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        """Pool token embeddings using mean and max pooling."""
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        masked = token_embeddings * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)
        mean_embedding = masked.sum(dim=1) / denom
        masked_for_max = token_embeddings.masked_fill(mask == 0, float("-inf"))
        max_embedding = masked_for_max.max(dim=1).values
        return mean_embedding, max_embedding

    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text string or list of text strings

        Returns:
            Text embeddings tensor [batch_size, embedding_dim]
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Enable gradient even though BERT is frozen (for mapper training)
        if self.freeze:
            with torch.enable_grad():
                outputs = self.bert_model(**inputs)

                # Select pooling strategy
                if self.pooling == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooling == "mean":
                    mean_embedding, _ = self._pool_tokens(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                    embeddings = mean_embedding
                elif self.pooling == "max":
                    _, max_embedding = self._pool_tokens(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                    embeddings = max_embedding
                else:  # concat
                    cls_embedding = outputs.pooler_output
                    mean_embedding, max_embedding = self._pool_tokens(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                    embeddings = torch.cat([cls_embedding, mean_embedding, max_embedding], dim=1)

                # Project if needed
                if self.projection:
                    embeddings = self.projection(embeddings)

            return embeddings
        else:
            outputs = self.bert_model(**inputs)

            # Select pooling strategy
            if self.pooling == "cls":
                embeddings = outputs.pooler_output
            elif self.pooling == "mean":
                mean_embedding, _ = self._pool_tokens(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = mean_embedding
            elif self.pooling == "max":
                _, max_embedding = self._pool_tokens(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = max_embedding
            else:  # concat
                cls_embedding = outputs.pooler_output
                mean_embedding, max_embedding = self._pool_tokens(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = torch.cat([cls_embedding, mean_embedding, max_embedding], dim=1)

            # Project if needed
            if self.projection:
                embeddings = self.projection(embeddings)

            return embeddings
