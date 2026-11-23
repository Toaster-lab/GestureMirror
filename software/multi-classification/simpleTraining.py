import torch
from torch import nn
import os

#loading data:
data_file = r'software\multi-classification\data.pt'

if not os.path.exists(data_file) or os.path.getsize(data_file) == 0: 
    example_tensor = torch.tensor([])
    print('Empty file')
    loaded_tensor = torch.reshape(example_tensor, (0,64))

data = torch.load(data_file, weights_only=True)
X = data[:, :-1]
y = data[:, -1].long() #CrossEntropyLoss uses the label as index
#.long() is just tensor.to(torch.int64)

#Hyperparamters:
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

model = HandNN()
entropy_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

def single_iteration():
    #probability = model(X)
    #predictions = torch.softmax(probability, dim=1)
    #prediction = torch.argmax(predictions, dim=1)
    # CrossEntropy does Softmax and stuff for me
    loss = entropy_loss(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def trainingRun():
    for i in range(amout_of_epochs):
        loss = single_iteration()
        if i % 100 == 0:
            print(f'iteration: {i} loss: {loss}')
def main():
    trainingRun()

if __name__ == "__main__":
    main()
