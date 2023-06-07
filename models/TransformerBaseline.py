import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from AddBiomechanicsDataset import InputDataKeys, OutputDataKeys


class TransformerLayer(nn.Module):
    def __init__(self, timestep_vector_dim: int, num_heads: int, dim_feedforward: int, dropout: float, dtype=torch.float64):
        super(TransformerLayer, self).__init__()

        self.multihead_attention = nn.MultiheadAttention(
            timestep_vector_dim, num_heads, dropout=dropout, batch_first=True, dtype=dtype)
        self.feedforward = nn.Sequential(
            nn.Linear(timestep_vector_dim, dim_feedforward, dtype=dtype),
            nn.ReLU(),
            nn.Linear(dim_feedforward, timestep_vector_dim, dtype=dtype)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(timestep_vector_dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(timestep_vector_dim, dtype=dtype)

    def forward(self, x: torch.Tensor):
        # Multihead self-attention
        # x.shape = (batch_size, seq_len, d_model)
        # attn_output.shape = (batch_size, seq_len, d_model)
        # The inputs to multihead_attention are the (query, key, value) tensors, all of which are the same for self-attention
        attn_output, _ = self.multihead_attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)

        # Feedforward neural network
        ff_output = self.feedforward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)

        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, window_size: int, embedding_dim: int, dtype=torch.float64):
        super(TemporalEmbedding, self).__init__()
        self.embedding = nn.Embedding(window_size, embedding_dim, dtype=dtype)

    def forward(self, x):
        embedded = self.embedding(x)
        return embedded


class SimpleAttention(nn.Module):
    def __init__(self, key_query_dim: int, dtype=torch.float64):
        super(SimpleAttention, self).__init__()

        self.query_linear = nn.Linear(
            key_query_dim, key_query_dim, dtype=dtype)
        self.key_linear = nn.Linear(key_query_dim, key_query_dim, dtype=dtype)

    def forward(self, query, key, value):
        q = self.query_linear(query)
        k = self.key_linear(key)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(weights, value)

        return output


class TransformerBaseline(nn.Module):
    timestep_vector_dim: int
    window_size: int
    temporal_embedding_dim: int
    output_vector_dim: int

    def __init__(self, dofs: int, window_size: int, temporal_embedding_dim: int = 30, num_layers: int = 3, num_heads: int = 3, dim_feedforward: int = 60, dropout: float = 0.0, dtype=torch.float64):
        super(TransformerBaseline, self).__init__()

        # Compute the size of the input vector to the transformer, which is the concatenation of (q, dq, ddq, com_pos, com_vel, com_acc)
        self.timestep_vector_dim = (
            dofs * 3) + (3 * 3) + temporal_embedding_dim
        # Output vector is 2 foot-ground contact predictions, 3 COM acceleration predictions, and 6 contact force predictions
        self.output_vector_dim = (2 + 3 + 6)
        self.window_size = window_size
        self.temporal_embedding_dim = temporal_embedding_dim

        self.temporal_embedding = TemporalEmbedding(
            window_size, temporal_embedding_dim, dtype=dtype)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.timestep_vector_dim,
                             num_heads, dim_feedforward, dropout, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(self.timestep_vector_dim,
                            self.output_vector_dim, dtype=dtype)
        self.contact_sigmoid = nn.Sigmoid()

        # This is single-headed attention, cause we only use one head
        self.com_attention = SimpleAttention(self.timestep_vector_dim)

    def forward(self, x: Dict[str, torch.Tensor]):
        batch_size = x[InputDataKeys.POS].size(0)

        # This concatenates (q, dq, ddq, com_pos, com_vel, com_acc) into a single vector per timestep
        input_vecs = torch.cat([
            x[InputDataKeys.POS],
            x[InputDataKeys.VEL],
            x[InputDataKeys.ACC],
            x[InputDataKeys.COM_POS],
            x[InputDataKeys.COM_VEL],
            x[InputDataKeys.COM_ACC]],
            dim=1)
        input_vecs = input_vecs.transpose(1, 2)

        # Now we need to generate the timestep embeddings
        timestep_indices = torch.arange(input_vecs.size(1))
        embedded = self.temporal_embedding(timestep_indices)
        embedded = embedded.expand(batch_size,
                                   self.window_size,
                                   self.temporal_embedding_dim)

        # Full input
        vecs = torch.cat([input_vecs, embedded], dim=2)

        # Actually run the transformer
        for transformer_layer in self.transformer_layers:
            vecs = transformer_layer(vecs)
        output = self.fc(vecs)

        # Reconstruct the COM acc as an attention layer over the input COM accs
        # Keys are the output vecs, queries are the output vecs, values are the input COM accs
        raw_com_accs = x[InputDataKeys.COM_ACC].transpose(1, 2)
        blending_coms = self.com_attention(
            vecs, vecs, raw_com_accs)

        # Now we need to split the output into the different components
        output_dict: Dict[str, torch.Tensor] = {}
        # Now decode the output into the different components
        output_dict[OutputDataKeys.CONTACT] = self.contact_sigmoid(
            output[:, :, :2]).transpose(1, 2)
        output_dict[OutputDataKeys.COM_ACC] = blending_coms.transpose(1, 2)
        output_dict[OutputDataKeys.CONTACT_FORCES] = output[:,
                                                            :, 5:].transpose(1, 2)

        return output_dict
