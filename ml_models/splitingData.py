import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
 
data = torch.load("data.pt", weights_only=True)
 
X = data[:, :-1]
y = data[:, -1]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
 
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)
