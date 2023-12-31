# v1 is very naive and not ready... but I need a subset now for training
# it just combines the manual selected ones with naive bad-word reject rule.
# this leads to 156330 / 509365 = 30.69% pick up rate

import os
import kiui
import pandas as pd
import numpy as np
import sqlitedict
import tqdm

db = sqlitedict.SqliteDict('objaverse_kiui.sqlite')

# empirical bad prompt
bad_word = ['flying', 'mountain', 'trash', 'featuring', 'a set of', 'a small', 'numerous', 'square', 'collection', 'broken', 'group', 'ceiling', 'wall', 'various', 'elements', 'splatter', 'resembling', 'landscape', 'stair', 'silhouette', 'garbage', 'debris', 'room', 'preview', 'floor', 'grass', 'house', 'beam', 'white', 'background', 'building', 'cube', 'box', 'frame', 'roof', 'structure']

path = 'data_cap3d'
df = pd.read_csv(os.path.join(path, 'Cap3D_automated_Objaverse_no3Dword_highquality.csv'), header=None)


uid_list = []

for i in tqdm.trange(len(df)):
    uid, prompt = df.iloc[i]
    prompt = prompt.lower()

    # manually selected ones
    if uid in db:
        if db[uid] == 1:
            uid_list.append(uid)
        continue
    
    # auto-reject by prompt
    if any([word in prompt for word in bad_word]) or prompt.count(',') >= 4:
        continue
    else:
        uid_list.append(uid)

# basic stats (pick up rate)
print(f'[INFO] uid_list: {len(uid_list)} / {len(df)} = {100 * len(uid_list) / len(df):.2f}%')

# export to a uid txt list
with open('kiuisobj_v1.txt', 'w') as f:
    for uid in uid_list:
        f.write(f'{uid}\n')

db.close()
