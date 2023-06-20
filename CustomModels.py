from torch import nn
import torch

class SimpleLinear(nn.Module):
  def __init__(self, start_shape, hidden_units, end_shape):
    super().__init__()
    self.linear_sequence = nn.Sequential(
      nn.Linear(start_shape, hidden_units),
      nn.Linear(hidden_units, end_shape)
    )
  def forward(self,x):
    return self.linear_sequence(x)


class SimpleLSTM(nn.Module):
  def __init__(self, start_shape, hidden_units, end_shape, num_stack_layers):
    super().__init__()
    self.num_stacked_layers = num_stack_layers
    self.hidden_units = hidden_units
    self.lstm = nn.LSTM(start_shape, hidden_units, num_stack_layers, batch_first=True)
    self.finish = nn.Linear(hidden_units, end_shape)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  def forward(self, x):
    h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_units).to(self.device)
    c0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_units).to(self.device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.finish(out[:, -1, :])
    return out