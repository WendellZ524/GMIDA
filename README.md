# GMI DA

use GMI to help SFDA estimate the source domain distribution

## Steps

1. Train a GAN use public datas
2. Attack the source model: GTA5
3. Adapt the source model to target: Cityscapes

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
