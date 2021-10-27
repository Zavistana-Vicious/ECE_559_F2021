#%%
import torchvision
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#### Stage 1: Load image and save it as torch.Tensor
# load the image
image = Image.open('data/geometry_dataset_raw/Octagon_8ae1fcce-2a89-11ea-8123-8363a7ec19e6.png')
#print(image)
# show the image
image = image.convert("L")
image.show()

# transform Image into the numpy array
foo = np.asarray(image)
#print(np.shape(image_2_npArray))
#print('the shape of loaded image transformed into numpy array: {}'.format(np.shape(image_2_npArray)))
#print('transformed image: {}'.format(image_2_npArray))

def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))

foo = normalize(foo)

plt.imshow(foo, cmap='Greys')
print(foo)

# %% https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7
# transform the numpy array into the tensor
#image_2_npArray_2_tensor = torchvision.transforms.ToTensor()(foo)
image_2_npArray_2_tensor = torch.Tensor(foo)
print('the shape of numpy array transformed into tensor: {}'.format(np.shape(image_2_npArray_2_tensor)))
print('transformed numpy array: {}'.format(image_2_npArray_2_tensor))
# %%
