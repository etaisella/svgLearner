import os
import torch
from torch.utils.data import Dataset
from utils.svgutils import loadClipassoSVG, renderCLipassoSVG, tensor2SVG

class ClipassoDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 im_height: int=224, 
                 im_width: int=224, 
                 add_noise: bool=True, 
                 noise_std: float=10.0):
        # iterate over all files in data_path and make a list of all svg files
        self.svg_paths = []
        self.im_height = im_height
        self.im_width = im_width
        self.add_noise = add_noise
        self.noise_std = noise_std
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".svg"):
                    self.svg_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.svg_paths)

    def __getitem__(self, idx: int):
        svg_path = self.svg_paths[idx]
        svgdict = loadClipassoSVG(svg_path)
        paths_tensor = svgdict["paths_tensor"].flatten()
        if self.add_noise:
            noise = torch.randn_like(paths_tensor) * self.noise_std
            paths_tensor = paths_tensor + noise
            svgdict = tensor2SVG(paths_tensor)
        img = renderCLipassoSVG(svgdict["shapes"], svgdict["shape_groups"])
        img = img.mean(dim=-1)
        img = img / img.max()
        return img, paths_tensor
    
    def path2trainImg(path_tensor: torch.Tensor) -> torch.Tensor:
        svgdict = tensor2SVG(path_tensor)
        img = renderCLipassoSVG(svgdict["shapes"], svgdict["shape_groups"])
        img = img.mean(dim=-1)
        img = img / img.max()
        return img


class pathBatch(Dataset):
    def __init__(self, 
                 path_batch: torch.Tensor):
        # iterate over all files in data_path and make a list of all svg files
        self.path_batch = path_batch

    def __len__(self):
        return self.path_batch.shape[0]

    def __getitem__(self, idx: int):
        path_tensor = self.path_batch[idx]
        svgdict = tensor2SVG(path_tensor)
        img = renderCLipassoSVG(svgdict["shapes"], svgdict["shape_groups"])
        img = img.mean(dim=-1)
        img = img / img.max()
        return img