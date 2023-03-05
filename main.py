import torch
from functorch import make_functional, jacrev

class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_hidden_layers: int=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.nn.functional.relu(x)
        return x

    
def jacobian(model: torch.nn.Module, x: torch.tensor) -> torch.Tensor:
    """Compute the Jacobian of a model w.r.t. its weights.
    
    Args:
        model (torch.nn.Module): The model.
        x (torch.Tensor): The input to the model.
    
    Returns:
        J (torch.Tensor): The vectorized Jacobian of the model w.r.t. its weights.
    """

    # Write model as a function of its parameters and input.
    func_model, params = make_functional(model)

    # Compute the derivative with respect to the parameters.
    grads = jacrev(func_model)(params, x)

    # Reshape into a vector.
    reshaped_grads = []
    batch_size = x.shape[0]
    for g in grads:
        reshaped_grads.append(g.reshape(batch_size, -1))
    J = torch.cat(reshaped_grads, dim=1)
    return J

if __name__ == '__main__':
    model = MLP(input_dim=10, hidden_dim=40, output_dim=1, num_hidden_layers=2)
    x = torch.randn(1000, 10)
    J = jacobian(model, x)
    print(J.shape)
    print(torch.linalg.matrix_rank(J))