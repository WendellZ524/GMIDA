# GMI DA

use GMI to help SFDA estimate the source domain distribution

## Steps

1. Train a GAN use public datas
2. Attack the source model: GTA5
3. Adapt the source model to target: Cityscapes

# Dataset
GTA5: orignial (1052,1914,3) in [0, 255] -> 128*256 [0, 255]

now matter the datatype, the value is in [0, 255]

CityScapes: (1024,2048,3) in [0, 255] -> 256x512 [0, 255]

# TODO list

## Dataloader

* [X] GTA5 Dataloader
* [ ] Cityscapes Dataloader

## Networks

* [X] GMI Generator
* [X] GMI Discriminator
* [X] Segmentation network
* [ ] DAM module
* [ ] IPSM module

## Losses

### GMI

### SFDA

* [X] BNS loss
* [ ] DAD loss
* [X] MAE loss
* [ ] ADV loss
* [X] TAR loss

## Evaluation

* [X] Cityscapes Evaluation

| classes | class id | color |
| ------- | ----- | ----- |
|road     | 0 ||
|sidewalk | 1 ||
|building | 2 ||
|wall     | 3 ||
|fence    | 4 ||
|pole     | 5 ||
|light    | 6 ||
|sign     | 7 ||
|veg      | 8 ||
|terrian  | 9 ||
|sky      |10 ||
|person   |11 ||
|rider    |12 ||
|car      |13 ||
|truck    |14 ||
|bus      |15 ||
|train    |16 ||
|motor    |17 ||
|bike     |18 ||
|dont care|255||