import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# loading data:
data_path = r'main\tensor.pt'
batch_size = 32
data = torch.load(data_path, weights_only=True)
X = data[:, :-1]
y = data[:, -1]

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
dropout_chance = 0.2
class SAMClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(p=dropout_chance),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.network(x)

model = SAMClassification()
log_loss = nn.BCEWithLogitsLoss()

learning_rate = 1e-1

# terminal: pip3 install sam-pytorch -- I am using sam.py
from sam import SAM
optimizer = SAM(model.parameters(), torch.optim.SGD, lr=learning_rate)



def single_iteration():
    for X_train, y_train in train_loader:
        prediction = model(X_train)
        loss = log_loss(prediction.squeeze(1), y_train)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        log_loss(model(X_train).squeeze(1), y_train).backward()
        optimizer.second_step(zero_grad=True)
    return loss.item()
   

def accuracy(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch).squeeze(1)
            predicted = (torch.sigmoid(logits) >= 0.35).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total
from torchmetrics.classification import BinaryCalibrationError

def ece(loader, amount_of_bins):
    metric = BinaryCalibrationError(n_bins=amount_of_bins, norm='l1')
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probabilty_ece = torch.sigmoid(model(X_batch).squeeze(1))
            metric.update(probabilty_ece, y_batch.long())
    model.train()
    return metric.compute().item()

def save_model():
    torch.save(model.state_dict(), r"main\modelSAM.pth")
    print('saved!')

amount_epochs = 751
def main():
    for i in range(amount_epochs):
        loss = single_iteration()
        if i % 50 == 0:
            ece_calibration = ece(train_loader, 15)
            train_acc = accuracy(train_loader)
            print(f'iteration: {i}')
            print(f'validation loss: {loss:.3f}')
            print(f'train accuracy: {train_acc:.3f}')
            test_acc = accuracy(test_loader)
            print(f'test accuracy: {test_acc:.3f}')
            print(f'ece: {ece_calibration:.3f}')
            print("\n")
    if test_acc >= 0.96:
        pass
        #save_model()
        
if __name__ == "__main__":
    main()