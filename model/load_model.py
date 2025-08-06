import torch
from model.gin_lstm_2layer_dropout02 import GINLSTM_2LayerDropout

def load_trained_model(path):
    model = GINLSTM_2LayerDropout(
        node_feat_dim=397,
        dok_embed_dim=8,
        hidden_dim=128,
        dropout=0.2
    )
    state = torch.load(path, map_location=torch.device("cpu"))

    # Load state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    return model
