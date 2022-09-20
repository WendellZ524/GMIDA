# GMI DA

use GMI to help SFDA estimate the source domain distribution


## Steps

1. Train a GAN use public datas
2. Attack the source model: GTA5
3. Adapt the source model to target: Cityscapes


# TODO list


## Dataloader

* [x] GTA5 Dataloader
* [ ] Cityscapes Dataloader

## Networks

* [ ] GMI Generator
* [ ] GMI Discriminator
* [x] Segmentation network
* [ ] DAM module
* [ ] IPSM module

## Losses

### GMI

### SFDA

* [ ] BNS loss
* [ ] DAD loss
* [ ] MAE loss
* [ ] ADV loss
* [ ] TAR


## Evaluation

* [ ] Cityscapes Evaluation
