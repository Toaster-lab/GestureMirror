import torch

file_name = r'testing\testing_tensor.pt'
tensor = torch.tensor([])
def reading_tensor(file_name):
    torch.save(tensor, file_name)
    print('saved')
    return tensor
def extending_loaded_tensor(loaded_tensor_file, added_tensor):
    loaded_tensor = reading_tensor(loaded_tensor_file)
    new_tensor = torch.cat((loaded_tensor.flatten(), added_tensor.flatten()))
    print(added_tensor, new_tensor)

reading_tensor(file_name=file_name)