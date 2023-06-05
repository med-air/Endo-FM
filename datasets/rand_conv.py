import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from einops import rearrange, repeat


class RandConv(nn.Module):

    def __init__(self, kernel_size=3, alpha=0.7, temporal_input=False):
        super(RandConv, self).__init__()
        self.m = nn.Conv2d(3, 3, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.std_normal = 1 / (np.sqrt(3) * kernel_size)
        self.alpha = alpha
        self.temporal_input = temporal_input

    def forward(self, image):
        with torch.no_grad():
            self.m.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros_like(self.m.weight),
                                                            std=torch.ones_like(self.m.weight) * self.std_normal))
            if self.temporal_input:
                batch_dim = image.shape[0]
                filtered_im = rearrange(image, "b c t h w -> (b t) c h w")
                filtered_im = self.m(filtered_im)
                filtered_im = rearrange(filtered_im, "(b t) c h w -> b c t h w", b=batch_dim)
            else:
                filtered_im = self.m(image)
            return self.alpha * image + (1 - self.alpha) * filtered_im


if __name__ == '__main__':

    filter_im = RandConv(temporal_input=True)
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])


    def whiten(input):
        return (input - mean.reshape(1, -1, 1, 1, 1)) / std.reshape(1, -1, 1, 1, 1)


    def dewhiten(input):
        return torch.clip(input * std.reshape(1, -1, 1, 1, 1) + mean.reshape(1, -1, 1, 1, 1), 0, 1)


    img = Image.open("/Users/kanchana/Documents/current/video_research/repo/dino/data/rand_conv/raw.jpg")
    # img_arr = torch.Tensor(np.array(img).transpose(2, 0, 1)).unsqueeze(0)
    img_arr = repeat(torch.Tensor(np.array(img)).unsqueeze(0), "b h w c -> b c t h w", t=8)
    img_arr = img_arr.float() / 255.

    for idx in range(10):
        with torch.no_grad():
            out = dewhiten(filter_im(whiten(img_arr)))
        # out = alpha * img_arr + (1 - alpha) * out
        im_vis = out[0, :, 0].permute(1, 2, 0).detach().numpy() * 255.
        im_vis = Image.fromarray(im_vis.astype(np.uint8))
        im_vis.save(f"/Users/kanchana/Documents/current/video_research/repo/dino/data/rand_conv/{idx + 1:05d}.jpg")
