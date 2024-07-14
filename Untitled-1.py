# %%
import torch
from torchvision.transforms import v2
from PIL import Image
img=Image.open('/Users/nshravanreddy/pytorch/face/flower.jpg')


# %%
from torchvision import transforms
to_tensor=transforms.ToTensor()
img_t=to_tensor(img)
img_t.shape, img_t.ndim,len(img_t[0]),img_t.size()[1:]

# %%
import matplotlib.pyplot as plt
def plot(images):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for i, (img_t, ax) in enumerate(zip(images, axes)):
        ax.imshow(img_t.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        ax.axis('off')
        ax.set_title(f'Image {i+1}')
    plt.show()

# %%
cropper = transforms.RandomCrop(size=(127, 127))
crops = [cropper(img_t) for _ in range(1000)]


# %%
to_image = transforms.ToPILImage()
def ri(c):
    comb_tensor=torch.zeros_like(c[1])
    for i in c:
        comb_tensor+=i
    comb_tensor/=len(c)
    return comb_tensor
img_r=ri(crops)
img_r_p=to_image(img_r)
plt.imshow(img_r_p)
plt.axis('off')
plt.title('Reconstructed Image')
plt.show()

# %%
img_r.shape,img_t.shape

# %%



