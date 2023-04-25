import torch

class MLP(torch.nn.Module):
    """
    An MLP with ReLU activations.
    """
    
    def __init__(self, input_dim: int=1,
                 hidden_dim: int=64,
                 output_dim: int=1,
                 num_hidden_layers: int=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim, bias=False)) # No bias for last layer.
        self.activations = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.activations = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.nn.functional.relu(x)
                self.activations.append(x > 0) # Record the activation pattern.
        return x