import os
import random
import torch
import numpy as np
import binvox as bv

from torch.utils.data import Dataset
from settings import *

class PRSDataset(Dataset):
    def __init__(self, data_dir, split_dir, bad_list, transform = torch.Tensor, mode = "train"):
        self.data_dir = os.path.join(data_dir, mode)
        self.transform = transform
        self.split_dir = split_dir
        self.closest_file = []
        self.sample_file = []
        self.voxel_file = []
        self.label = []
        self.num_obj = 0

        # Read bad list
        self.bad_model = []
        with open(bad_list) as bad_file:
            lines = bad_file.readlines()
            for line in lines:
                bad_label = line.split("|")[1]
                self.bad_model.append(bad_label)

        for _class in os.listdir(self.data_dir):
            # Skip unprocessed models
            if (UNPROCESSED != None) and (_class not in UNPROCESSED):
                continue

            mode_label = []
            # Get all train/test labels (with no suffix or prefix)
            with open(os.path.join(split_dir, _class[1: ] + "_" + mode + ".txt")) as class_split:
                for obj in class_split:
                    mode_label.append(obj.strip("\n"))

            # Check all obj in this class
            valid_label = []
            class_path = os.path.join(self.data_dir, _class)
            for obj in os.listdir(class_path):
                # Skip bad models
                if (_class + "_" + obj) in self.bad_model:
                    continue

                # Skip test if train, or vice versa
                if obj.split("_")[0] not in mode_label:
                    continue

                # valid obj
                valid_label.append(obj)

            # Drop extra obj, especially applicable when original obj > 4000
            random.shuffle(valid_label)
            valid_label = valid_label[0 : NUM_AUG]

            # Now add these obj to list for later use when running
            for obj in valid_label:
                obj_dir = os.path.join(class_path, obj)
                self.label.append(_class + "_" + obj)
                self.closest_file.append(os.path.join(obj_dir, "closest.npy"))
                self.sample_file.append(os.path.join(obj_dir, "sample.npy"))
                self.voxel_file.append(os.path.join(obj_dir, "voxel.binvox"))

                self.num_obj += 1

    def __len__(self):
        return self.num_obj

    def __getitem__(self, index):
        closest = np.load(self.closest_file[index]).astype(np.float32)
        sample = np.load(self.sample_file[index]).astype(np.float32)
        voxel = bv.Binvox.read(self.voxel_file[index], mode = "dense").numpy()
        # Note: just ignore translate and scale
        # (1, 32, 32, 32), add an extra dim to be multiplied by `in_channel`
        voxel = self.transform(voxel).unsqueeze(0)
        label = self.label[index]

        return closest, sample, voxel, label
