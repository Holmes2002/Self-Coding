import copy
import math
import os
import os.path
import random
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.distributed import DistributedSampler

import augmentation as psp_trsform


class customs_dset(Dataset):
    def __init__(self, cfg, trs_form, split="val"):
        super(customs_dset, self)
        if split=="val":
            self.list_sample=open("val.txt","r").read().splitlines()

        self.data_custom = open(cfg['train']['data_list'],'r').read().splitlines()

        remainder = (len(self.data_custom) % 8)
        self.data_custom = self.data_custom+self.data_custom[:remainder]
        self.data_LIP = open(cfg['train']['data_LIP'],'r').read().splitlines()
        self.transform = trs_form
        self.data_root=''
        self.list_sample=self.data_custom+self.data_LIP
    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(mode)
    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample[index])
        label_path = os.path.join(self.data_root, self.list_sample[index].replace("jpg","png"))
        
        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "L")
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample)


class customs_dset_unsup(Dataset):
    def __init__(self, cfg, trs_form, split="val"):
        super(customs_dset_unsup, self)
        self.root=''
        self.transform = trs_form
        self.data_root=''
        self.list_sample_new=open('unlabeled.txt','r').read().splitlines()
    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(mode)
    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index])
        label_path = os.path.join(self.data_root, self.list_sample_new[index])
        
        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "L")
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)
def build_transfrom(cfg):
    trs_form = []
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
    if cfg.get("cutout", False):
        n_holes, length = cfg["cutout"]["n_holes"], cfg["cutout"]["length"]
        trs_form.append(psp_trsform.Cutout(n_holes=n_holes, length=length))
    if cfg.get("cutmix", False):
        n_holes, prop_range = cfg["cutmix"]["n_holes"], cfg["cutmix"]["prop_range"]
        trs_form.append(psp_trsform.Cutmix(prop_range=prop_range, n_holes=n_holes))

    return psp_trsform.Compose(trs_form)


def build_customsloader(split, all_cfg,arg):
    cfg_dset = all_cfg["dataset"]
    cfg_trainer = all_cfg["trainer"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    batch_size = cfg.get("batch_size", 1)

    # build transform
    trs_form = build_transfrom(cfg)
    dset = customs_dset( arg.data_custom,arg.data_LIP, trs_form, split="val")

    # build sampler
    sample = DistributedSampler(dset)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_customs_semi_loader(split, all_cfg,arg):
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    dset = customs_dset(cfg, trs_form, split="train")
    if split == "val":
        # build sampler
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        dset_unsup = customs_dset_unsup(
            cfg, trs_form_unsup, split
        )

        
        loader_sup = DataLoader(
            dset,
            batch_size=2,
            num_workers=workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=2,
            num_workers=workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup
