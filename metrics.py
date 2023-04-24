import torch
from torch.func import jacrev, functional_call


def jacobian(model: torch.nn.Module, x: torch.tensor) -> torch.Tensor:
    """
    Compute the Jacobian of a model w.r.t. its (vectorized) weights.
    
    Args:
        model (torch.nn.Module): The model.
        x (torch.Tensor): The input to the model.
    
    Returns:
        J (torch.Tensor): The Jacobian of the model w.r.t. its weights.
    """

    # Compute the derivative with respect to the parameters.
    f = lambda params, inputs: functional_call(model, params, (inputs,))
    params = dict(model.named_parameters())
    grads = jacrev(f)(params, x)

    # Reshape into a vector.
    reshaped_grads = []
    batch_size = x.shape[0]
    for _, g in grads.items():
        reshaped_grads.append(g.reshape(batch_size, -1))
    J = torch.cat(reshaped_grads, dim=1)
    return J