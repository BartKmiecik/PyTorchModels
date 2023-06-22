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

class MultiClassification(nn.Module):
  def __init__(self,
               input_features,
               output_features,
               hidden_units):
    super().__init__()
    self.combo_layer1 = nn.Sequential(
        nn.Conv2d(in_channels=input_features,
                  out_channels=hidden_units,
                  stride=1,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  stride=1,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.combo_layer2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=1,
                  stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=1,
                  stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7,
                  out_features=output_features)
    )

  def forward(self, x):
    x = self.combo_layer1(x)
    #print(x.shape)
    x = self.combo_layer2(x)
    #print(x.shape)
    x = self.classifier(x)
    return x