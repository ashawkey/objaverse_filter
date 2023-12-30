import sys
sys.path.append('.')

import tyro
import random

import torch
from lrm.options import AllConfigs
from lrm.provider_cap3d import Cap3DDataset

import kiui

def main():    
    opt = tyro.cli(AllConfigs)

    # data
    dataset = Cap3DDataset(opt, training=True)

    # get data for verification
    while True:
        idx = random.randint(0, len(dataset) - 1)
        data = dataset[idx]

        cams = data['cam_poses']
        images = data['images_output']

        kiui.lo(cams)
        kiui.vis.plot_poses(cams.detach().cpu().numpy())
        kiui.vis.plot_image(images.detach().cpu().numpy())
    

if __name__ == "__main__":
    main()