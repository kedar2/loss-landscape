import torch
from models import MLP
from metrics import jacobian

if __name__ == '__main__':
    model = MLP(input_dim=10, hidden_dim=40, output_dim=1, num_hidden_layers=2)
    x = torch.randn(1000, 10)
    J = jacobian(model, x)
    print("Shape of Jacobian: ", J.shape)
    print("Rank of Jacobian:", torch.linalg.matrix_rank(J))