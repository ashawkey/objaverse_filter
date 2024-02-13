# Objaverse Subset of Kiui

This repo contains the subset of [Objaverse](https://objaverse.allenai.org/objaverse-1.0) used in [LGM](https://github.com/3DTopia/LGM).

* [kiuisobj_v1.txt](./kiuisobj_v1.txt): ~150K, filter bad prompts.
* [kiuisobj_v1_merged_80K.csv](./kiuisobj_v1_merged_80K.csv): ~82K, v1 + filter missing textures. This is the subset used to train LGM.
* [gobj_merged.json](./gobj_merged.json): ~83K, union of v1 and [G-objaverse](https://github.com/modelscope/richdreamer/blob/main/dataset/gobjaverse/README.md)'s 280K list.
