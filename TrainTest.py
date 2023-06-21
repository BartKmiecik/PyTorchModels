import tqdm
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error
class train_regression_model():
    def __init__(self, model: nn.Module, data_loader, optimizer: torch.optim, loss_fn, epochs: int, ):
        for epoch in tqdm(range(epochs)):
            # print(f"Epoch: {epoch}")
            train_loss = 0
            train_acc = 0
            for batch, (X, y) in enumerate(data_loader):
                model.train()
                y_preds = model(X)

                loss = loss_fn(y_preds, y)
                train_loss += loss
                train_acc += mean_absolute_error(y.cpu().detach().numpy(),
                                                 y_preds.cpu().detach().numpy())
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                # if batch % 400 == 0:
                # print(f"Looked at {batch * len(X)}/{len(train_loader.dataset)} samples")
            train_loss /= len(data_loader)
            train_acc /= len(data_loader)
            test_loss, test_acc = 0, 0

            print(f"\nTrain loss: {train_loss} | Train acc: {train_acc}")

class evaluate_regression_model():
    def __init__(self, model: nn.Module, data_loader,loss_fn):
        model.eval()
        test_loss = 0
        test_acc = 0
        with torch.inference_mode():
            for X, y in data_loader:
                test_pred = model(X)
                loss = loss_fn(test_pred, y)
                test_loss += loss

                test_acc += mean_absolute_error(y.cpu().detach().numpy(),
                                                test_pred.cpu().detach().numpy())

            test_loss /= len(data_loader)
            test_acc /= len(data_loader)

        print(f"Test loss: {test_loss} | Test acc: {test_acc}\n")


