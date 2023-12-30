import os
import kiui
import pandas as pd
import numpy as np
import sqlitedict
import cv2

db = sqlitedict.SqliteDict('objaverse_kiui.sqlite')

# empirical bad prompt
bad_word = ['flying', 'mountain', 'trash', 'featuring', 'a set of', 'a small', 'numerous', 'square', 'collection', 'broken', 'group', 'ceiling', 'wall', 'various', 'elements', 'splatter', 'resembling', 'landscape', 'stair', 'silhouette', 'garbage', 'debris', 'room', 'preview', 'floor', 'grass', 'house', 'beam', 'white', 'background', 'building', 'cube', 'box', 'frame', 'roof', 'structure']

path = 'data_cap3d'
df = pd.read_csv(os.path.join(path, 'Cap3D_automated_Objaverse_no3Dword_highquality.csv'), header=None)

cv2.startWindowThread()
cv2.namedWindow("Press Space to Accept, Others to Reject")

for i in range(len(df)):
    uid, prompt = df.iloc[i]
    prompt = prompt.lower()

    if uid in db:
        continue
    
    # auto-reject by prompt
    if any([word in prompt for word in bad_word]) or prompt.count(',') >= 4:
        print(f'[INFO] {uid}: bad word, skip')
        db[uid] = 0
        continue

    # show 4 views
    images = []
    for vid in range(4):
        image_path = os.path.join(path, 'imgs', f'Cap3D_imgs_view{vid}', f'{uid}')
        image = kiui.read_image(image_path, mode='float') # [512, 512, 3] in [0, 1]
        images.append(image)
    # make 2x2 grid
    image = np.stack(images, axis=0) # [4, 512, 512, 3]
    N, H, W = image.shape[:3]
    image = image.reshape(2, 2, H, W, 3)
    image = image.transpose(0, 2, 1, 3, 4)
    image = image.reshape(2 * H, 2 * W, 3)
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # user action
    cv2.imshow('Press Space to Accept, Others to Reject', image)
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