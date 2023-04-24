import torch
from tqdm import tqdm

def train_model(model, optimizer, loss_fn, X, y, num_epochs):
    act_changes = torch.zeros(num_epochs)
    train_loss = -torch.ones(num_epochs)
    A = torch.zeros(X.shape)
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        preds, A_new = model(X)
        if not torch.equal(A, A_new):
          A = A_new
          act_changes[epoch] = 1
        loss = loss_fn(preds, y)
        loss.backward()
        train_loss[epoch] = loss
        optimizer.step()
    preds,_ = model(X)
    loss = loss_fn(preds, y)
    print("Finished training, final training loss: " + str(loss.item()))
    return model, act_changes, train_loss