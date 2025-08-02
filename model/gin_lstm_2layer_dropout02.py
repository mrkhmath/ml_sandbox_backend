
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool

class GINLSTM_2LayerDropout(nn.Module):
    def __init__(self, node_feat_dim=397, dok_embed_dim=8, hidden_dim=128, dropout=0.2):
        super().__init__()

        self.gin1 = GINConv(nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))
        self.gin2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))

        self._init_weights()

        self.dok_embed = nn.Embedding(5, dok_embed_dim)
        self.lstm = nn.LSTM(hidden_dim + dok_embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def _init_weights(self):
        for conv in [self.gin1, self.gin2]:
            for layer in conv.nn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

    def forward(self, sequences):
        if isinstance(sequences, list) and len(sequences) == 1 and isinstance(sequences[0], list):
            sequences = sequences[0]

        logits = []
        device = next(self.parameters()).device

        for step in sequences:
            data = step["graph"].to(device)
            dok = step["dok"].to(device)
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

            x = self.gin1(data.x, data.edge_index)
            x = self.gin2(x, data.edge_index)
            pooled = global_mean_pool(x, batch)

            dok_emb = self.dok_embed(dok)
            step_emb = torch.cat([pooled, dok_emb], dim=-1).squeeze(0)
            logits.append(step_emb)

        lstm_input = torch.stack(logits).unsqueeze(0)
        lstm_out, _ = self.lstm(lstm_input)
        output = self.fc(lstm_out.squeeze(0)).squeeze()

        return output
