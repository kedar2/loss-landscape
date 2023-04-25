import torch
from tqdm import tqdm
from typing import Tuple
from metrics import jacobian

def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                X: torch.Tensor,
                y: torch.Tensor,
                num_epochs: int) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    
    # Metrics to track.
    act_changes = []
    train_loss = []
    jacobian_ranks = []
    A = []
    J = jacobian(model, X)
    jacobian_ranks.append(torch.linalg.matrix_rank(J).item())

    print("Starting training.")

    for _ in tqdm(range(num_epochs)):
        # Backward pass.
        optimizer.zero_grad()
        preds = model(X)
        A_new = model.activations
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        # Record metrics.
        train_loss.append(loss.item())
        J = jacobian(model, X)
        jacobian_ranks.append(torch.linalg.matrix_rank(J).item())
        if len(A) != len(A_new) or not all(torch.equal(a, b) for a, b in zip(A, A_new)):
            # Record a change in the activation pattern.
            A = A_new
            act_changes.append(1)
        else:
            act_changes.append(0)


    logged_metrics = {'act_changes': act_changes, 'train_loss': train_loss, 'jacobian_ranks': jacobian_ranks}
    return model, logged_metrics