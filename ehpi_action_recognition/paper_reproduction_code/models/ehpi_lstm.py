import torch

from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EhpiLSTM(nn.Module):
    def __init__(self, num_joints: int, num_classes: int):
        super().__init__()
        self.hidden_dim = 64
        self.num_layers = 2
        self.num_input_parameters = 36
        self.lstm = nn.LSTM(num_joints*2, self.hidden_dim, self.num_layers, bias=True, batch_first=True)
        self.hidden2label = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        y = self.hidden2label(lstm_out)
        # log_probs = F.log_softmax(y)
        return y[:, -1, :]