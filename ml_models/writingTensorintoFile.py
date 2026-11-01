import cv2 as cv
import torch
example_tensor1 = torch.tensor([[1,2,3],[4,5,6]])

#print(example_tensor1.shape)

file_name_example = 'ml_models/tensor.pt'

def saving_string(file_name, example_tensor):
    with open(file_name, 'w') as file:
        file.write(str(example_tensor.tolist()))

def saving_tensor(file_name, example_tensor):
    torch.save(example_tensor, file_name)

def waiting_input(input_needed):
    k = cv.waitKey(0)
    if k == ord(input_needed):
        return True
    return False

def main():
    cv.namedWindow('s')
    writing_tensor = waiting_input('s')
    if writing_tensor == True:
        saving_tensor(file_name_example, example_tensor1)
        print('saved')

#main()
cv.destroyAllWindows()

def reading_tensor(file_name):
    print(torch.load(file_name, weights_only=True))
reading_tensor(file_name_example)


# This all works now!!! 
# What I learned: That when torch saves tensors, it isnt readable, I have to convert it back