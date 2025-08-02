import torch
from model.gin_lstm_2layer_dropout02 import GINLSTM_2LayerDropout

def load_model(checkpoint_path="epoch14_acc0.7379_f10.7350.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GINLSTM_2LayerDropout().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device