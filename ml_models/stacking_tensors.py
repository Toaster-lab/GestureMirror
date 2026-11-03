import torch

tensors = []

tensor1 = torch.tensor([[1,2],[3,4]])
tensor2 = torch.tensor([[5,6],[7,8]])

tensors.append(tensor1)
tensors.append(tensor2)
tensors_list = torch.stack(tensors)
print(tensors_list)
# how this would look in my code:
loaded_list = tensors_list #it would in my program use torch.load!
new_list = []
tensor3 = torch.tensor([[9,10],[11,12]])
tensor4 = torch.tensor([[13,14],[15,16]])
new_list.append(tensor3)
new_list.append(tensor4)
stacked_list = torch.stack(new_list)
loaded_list = torch.cat((loaded_list, stacked_list), dim=0)
print(loaded_list)