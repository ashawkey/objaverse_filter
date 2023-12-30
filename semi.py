import os
import kiui
import pandas as pd
import numpy as np
import sqlitedict
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import torch
from safetensors.torch import load_file

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

db = sqlitedict.SqliteDict('objaverse_semi.sqlite')

# empirical bad prompt
bad_word = ['flying', 'mountain', 'trash', 'featuring', 'a set of', 'a small', 'numerous', 'square', 'collection', 'broken', 'group', 'ceiling', 'wall', 'various', 'elements', 'splatter', 'resembling', 'landscape', 'stair', 'silhouette', 'garbage', 'debris', 'room', 'preview', 'floor', 'grass', 'house', 'beam', 'white', 'background', 'building', 'cube', 'box', 'frame', 'roof', 'structure']

path = 'data_cap3d'
df = pd.read_csv(os.path.join(path, 'Cap3D_automated_Objaverse_no3Dword_highquality.csv'), header=None)


# load model
from classifier.models import Classifier
model = Classifier()
workspace = 'workspace'
# resume
last_ckpt = os.path.join(workspace, 'model.safetensors')
ckpt = load_file(last_ckpt, device='cpu')
model.load_state_dict(ckpt, strict=False)
model.eval().cuda()
print('[INFO] model loaded')

cv2.startWindowThread()
cv2.namedWindow("Press Space to Accept, Others to Reject")
for i in range(len(df)):
    uid, prompt = df.iloc[i]
    prompt = prompt.lower()

    if uid in db:
        continue
    
    # auto-reject by prompt
    # if any([word in prompt for word in bad_word]) or prompt.count(',') >= 4:
    #     print(f'[INFO] {uid}: bad word, skip')
    #     db[uid] = 0
    #     continue

    # show 8 views
    images = []
    for vid in range(8):
        image_path = os.path.join(path, 'imgs', f'Cap3D_imgs_view{vid}', f'{uid}')
        image = kiui.read_image(image_path, mode='float') # [512, 512, 3] in [0, 1]
        images.append(image)

    images = np.stack(images, axis=0) # [8, 512, 512, 3]

    # predict score
    images_torch = torch.from_numpy(images).permute(0, 3, 1, 2)
    images_torch = F.interpolate(images_torch, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0) # [C, H, W]
    images_torch = TF.normalize(images_torch, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    preds = torch.sigmoid(model(images_torch.cuda())).detach().cpu().numpy() # [8]
    pred_score = (preds.mean() > 0.5).astype(np.int32)

    # show semi-score
    for j, pred in enumerate(preds):
        if pred > 0.5:
            images[j, 0:20, 0:20, :] = [0, 1, 0]
        else:
            images[j, 0:20, 0:20, :] = [1, 0, 0]

    # make 2x2 grid
    N, H, W = images.shape[:3]
    images = images.reshape(2, 2, H, W, 3)
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape(2 * H, 2 * W, 3)
    images = (images * 255).astype(np.uint8)
    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    
    # user action
    cv2.imshow('Press Space to Accept, Others to Reject', images)
    key = cv2.waitKey(0)
    if key == 32: # space, adopt current image
        score = 1
    else: # else, reject
        score = 0

    # record and save
    db[uid] = score
    db.commit()
    print(prompt)
    print(f'[INFO] {uid}: {score} (progress: {i} / {len(df)} = {100 * i/len(df):.2f} %)')

db.close()