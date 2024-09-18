import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .ray_utils import get_ray_directions, get_inverse_pose, get_rays
from torchvision.transforms import ToTensor
from torch import Tensor
from typing import Tuple, Union, List

from typing import TypeVar, Union, Tuple, Optional
from torch import Tensor
from torch.nn.modules.utils import _pair
import torchvision.transforms as transforms

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class MVImgNetNeRF(Dataset):
    def __init__(self, datadir: str, split: str='train', 
                 img_wh: List[int]=[90,160], patch_hw: List[int]=[16,10], number_of_imgs_from_the_same_scene: int=4, transform = default_transform):

        '''
        Load cached data, return image patches and rays of its pixel
        '''

        self.root_dir = datadir 
        self.white_bg = False
        self.number_of_imgs_from_the_same_scene = number_of_imgs_from_the_same_scene
        self.ndc_ray = False
        self.img_wh = img_wh
        self.patch_hw = patch_hw
        self.split = split

        cache_path = os.path.join(self.root_dir, 'cache_portrait_{}.th'.format(split))
        cache = torch.load(cache_path)

        self.img_files = [img_file for img_file in cache['img_files']]
        self.number_of_imgs_per_scene = cache['number_of_imgs_per_scene']
        self.c2ws = cache['c2ws']
        self.focals = cache['focals'] # has to be scene-specific
        self.subdirs = cache['subdirs']
        self.near_far = cache['near_far']
        self.original_img_hws = cache['img_sizes_hw']
                
        self.starting_idxs = [0] + list(np.cumsum(self.number_of_imgs_per_scene))[:-1] #start idx of each scene
        self.ending_idxs = list(np.cumsum(self.number_of_imgs_per_scene)) #end idx of each scene
        self.scene_bbox = torch.tensor([[-0.8,-0.8,-0.8],[0.8,0.8,0.8]])
        self.num_objs = self.starting_idxs + self.ending_idxs[-1:]
        self.n_scenes = len(self.number_of_imgs_per_scene)

        self.transform = transform


    def __len__(self): 
        return len(self.img_files)

    def load_image(self, index: Union[int, np.int_], scene_index: int) -> np.ndarray:
        origin_H, origin_W = self.original_img_hws[scene_index]
        imgs = np.array(Image.open(self.img_files[index]).resize([origin_W, origin_H], Image.LANCZOS).convert('RGB'))
        if self.transform is not None:
            imgs = self.transform(imgs)
        else:
            imgs = imgs / 255

        return imgs

    def get_data(self, scene_idx: int, return_scene_name = False) -> Union[Tuple[List[Tensor], List[Tensor], List[Tensor], List[float], Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor], List[float], Tensor, str]]:
        idxs = torch.arange(self.starting_idxs[scene_idx], self.ending_idxs[scene_idx])
        imgs = [ToTensor()(Image.open(self.img_files[idx])).unsqueeze(0) for idx in idxs]
        c2ws = [self.c2ws[idx] for idx in idxs]
        test_idx = len(c2ws) // 2
        campos = np.linalg.norm(c2ws[test_idx][:3,-1])
        zero_inverse_pose =  get_inverse_pose(c2ws[test_idx], offset=np.array([0,0,-campos]))
        c2ws = [zero_inverse_pose@c2w for c2w in c2ws]

        directions = get_ray_directions(self.img_wh[1], self.img_wh[0], Tensor([self.focals[scene_idx], self.focals[scene_idx]]))
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        ray_origins_directions = [get_rays(directions, torch.from_numpy(c2w).float()) for c2w in c2ws]
        ros = [ro_rd[0].unsqueeze(0) for ro_rd in ray_origins_directions]
        rds = [ro_rd[1].unsqueeze(0) for ro_rd in ray_origins_directions]

        normalized_focal = self.focals[scene_idx] / self.img_wh[1]
        normalized_focals = [normalized_focal for _ in range(len(imgs))]

        if return_scene_name:
            scene_name = self.img_files[idxs[0]].split('/')[-4] + '_' + self.img_files[idxs[0]].split('/')[-3]
            return imgs, ros, rds, normalized_focals, imgs[test_idx], scene_name

        return imgs, ros, rds, normalized_focals, imgs[test_idx]

    #Get the start, end and scene idx of a given index
    def sample_from_same_scene(self, index: int) -> Tuple[int, int, int]:
        scene_idx = [it for it, _ in enumerate(self.starting_idxs) if index >= self.starting_idxs[it] and index < self.ending_idxs[it]]
        return self.starting_idxs[scene_idx[0]], self.ending_idxs[scene_idx[0]], scene_idx[0]

    #return the indexed iamge and (self.number_of_imgs_from_the_same_scene - 1) random images
    def __getitem__(self, index: Tensor) -> dict:
        output = dict()
        start_idx, end_idx, scene_idx = self.sample_from_same_scene(index)

        assert self.number_of_imgs_from_the_same_scene > 1
        new_idxs = np.random.randint(low=start_idx, high=end_idx, size=(self.number_of_imgs_from_the_same_scene-1))
        return_imgs = [self.load_image(it, scene_idx) for it in [index]  + [new_idx for new_idx in new_idxs]]
        
        c2ws = [self.c2ws[it] for it in [index]+ [new_idx for new_idx in new_idxs]]

        output['indexs'] = np.array([index] + new_idxs.tolist())

        # zero image : identity
        campos = np.linalg.norm(c2ws[0][:3,-1])
        zero_inverse_pose =  get_inverse_pose(c2ws[0], offset=np.array([0,0,-campos]))
        c2ws = [zero_inverse_pose@c2w for c2w in c2ws]

        directions = get_ray_directions(self.img_wh[1], self.img_wh[0], Tensor([self.focals[scene_idx], self.focals[scene_idx]]))
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        ray_origins_directions = [get_rays(directions, torch.from_numpy(c2w).float()) for c2w in c2ws]

        ros = np.array([ro_rd[0] for ro_rd in ray_origins_directions])
        rds = np.array([ro_rd[1] for ro_rd in ray_origins_directions])

        output['ndc_ray'] = self.ndc_ray
        output['white_bg'] = self.white_bg
        output['scene_idx'] = scene_idx
        output['original_img_hws'] = np.array(self.original_img_hws[scene_idx])

        output['images'] = torch.Tensor(np.array(return_imgs)).reshape(-1, 3, self.img_wh[1], self.img_wh[0])
        
        output['normalized_focals'] = torch.Tensor(np.array([self.focals[scene_idx] / self.img_wh[1] for _ in range(len(output['images']))]))
        output['ray_origins'] = torch.Tensor(np.array(ros)).reshape(-1, self.img_wh[1], self.img_wh[0], 3).permute(0, 3, 1, 2)
        output['ray_directions'] = torch.Tensor(np.array(rds)).reshape(-1, self.img_wh[1], self.img_wh[0], 3).permute(0, 3, 1, 2)
            
        return output
