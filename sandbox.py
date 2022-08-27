import torch
import lpips
from PIL import Image
import torchvision.transforms as torch_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

l1_loss = torch.nn.L1Loss()
lpips_loss = lpips.LPIPS("net=alex", verbose=False).to(device)

img_1 = "/usr/stud/dave/storage/user/dave/timelens_vimeo/00001/0001/upsampled/imgs/00000000.png"
img_2 = "/usr/stud/dave/storage/user/dave/timelens_vimeo/00001/0001/upsampled/imgs/00000001.png"

mytransforms = torch_transforms.ToTensor()

img1_tensor = mytransforms(Image.open(img_1)).to(device)
img2_tensor = mytransforms(Image.open(img_2)).to(device)

raw_l1 = l1_loss(img1_tensor, img2_tensor)
raw_lpips = lpips_loss(img1_tensor, img2_tensor)

print(raw_l1, raw_lpips)