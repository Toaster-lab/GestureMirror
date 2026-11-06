import torch
from model_versions.training_model import HandNN
model_path = r"model_versions\model.pth"
model = HandNN()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
#print(model.eval())
print('loaded succesfully')

file = r"testing\testing_tensor.pt"
loaded_tensor = torch.load(file)
print(loaded_tensor.shape)
tensor = loaded_tensor[:,:-1]
with torch.no_grad():
    predicition = model(tensor)
    print(predicition.sigmoid())
