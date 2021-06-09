import torch.nn as nn

# 19 8 6
class MLP1(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        # nn.Flatten(),
        nn.Linear(768, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 19)
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


class MLP2(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        # nn.Flatten(),
        nn.Linear(768, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 8)
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


class MLP3(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      # nn.Flatten(),
        nn.Linear(768, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 6)
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


