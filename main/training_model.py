import torch
from torch import nn

#loading data:
data_file = r'main\tensor.pt'
data = torch.load(data_file, weights_only=True)
X = data[:, :-1]
y = data[:, -1:]

#Hyperparamters:
learning_rate = 0.001
amout_of_epochs = 500

class HandNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.network(x)

model = HandNN()
log_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

def single_iteration():
    prediction = model(X)
    loss = log_loss(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def trainingRun():
    for i in range(amout_of_epochs):
        loss = single_iteration()
        if i % 100 == 0:
            print(f'iteration: {i} loss: {loss}')
    torch.save(model.state_dict(), r"main\model.pth")
    print('saved!')

if __name__ == "__main__":
    trainingRun()

# Problems: -it needs contrast
#           -hand rotation -> closed hand always
#           -distance of hand to cam / cam quality