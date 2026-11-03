import torch

file_name = r'ml_models\tensor.pt'
tensor = torch.tensor([[1,2],[3,4]])
def reading_tensor(file_name):
    tensor = torch.load(file_name, weights_only=True)
    print(tensor)
    return tensor
def extending_loaded_tensor(loaded_tensor_file, added_tensor):
    loaded_tensor = reading_tensor(loaded_tensor_file)
    new_tensor = torch.cat((loaded_tensor.flatten(), added_tensor.flatten()))
    print(added_tensor, new_tensor)


reading_tensor(file_name=file_name)