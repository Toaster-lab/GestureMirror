import torch
from torch import nn
import os

HERE = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(HERE, 'data.pt')
model_file = os.path.join(HERE, 'model_mc.pt')

#Hyperparamters:
batch_size = 32
test_size = 0.2
learning_rate : float = 0.01
amout_of_epochs : int = 500
num_of_classes : int = 8

class HandNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_of_classes)
        )
    def forward(self, x):
        return self.network(x)

# Training state is created in setup() so that importing HandNN
# (e.g. from savingTensors.py) has no side effects.
model = entropy_loss = optimizer = train_loader = test_loader = None

def setup():
    global model, entropy_loss, optimizer, train_loader, test_loader
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader

    if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
        raise FileNotFoundError(f'No training data in {data_file}')
    data = torch.load(data_file, weights_only=True)
    X = data[:, :-1]
    y = data[:, -1].long() #CrossEntropyLoss uses the label as index

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    model = HandNN()
    entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def saving_model():
    torch.save(model.state_dict(), model_file)
    print('saved!')

def accuracy(loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            predicted = torch.argmax(model(X_batch), dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    model.train()
    return correct / total

from torchmetrics.classification import MulticlassCalibrationError

def ece(loader, amount_of_bins):
    metric = MulticlassCalibrationError(num_classes=num_of_classes, n_bins=amount_of_bins, norm='l1')
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probability_ece = torch.softmax(model(X_batch), dim=1)
            metric.update(probability_ece, y_batch)
    model.train()
    return metric.compute().item()

def single_iteration():
    # CrossEntropy does Softmax and stuff for me
    for X_batch, y_batch in train_loader:
        loss = entropy_loss(model(X_batch), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def trainingRun():
    test_acc = 0.0
    for i in range(amout_of_epochs):
        loss = single_iteration()
        if i % 100 == 0:
            ece_calibration = ece(train_loader, 15)
            train_acc = accuracy(train_loader)
            test_acc = accuracy(test_loader)
            print(f'iteration: {i}')
            print(f'loss: {loss:.3f}')
            print(f'train accuracy: {train_acc:.3f}')
            print(f'test accuracy: {test_acc:.3f}')
            print(f'ece: {ece_calibration:.3f}')
            print("\n")
    return test_acc

def main():
    setup()
    test_acc = trainingRun()
    if test_acc >= 0.5:
        saving_model()


if __name__ == "__main__":
    main()
