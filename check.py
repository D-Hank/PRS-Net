import os
import numpy as np

from tqdm import tqdm
from settings import *

bad_model = []
bad_record = open(BAD_MODEL_RECORD, "w")

def check_data(data_dir):
    for _class in os.listdir(data_dir):
        class_path = os.path.join(data_dir, _class)
        total = os.listdir(class_path)
        loop = tqdm(enumerate(total), total = len(total))

        for index, obj in loop:
            # New data
            new_dir = os.path.join(class_path, obj)
            sample_path = os.path.join(new_dir, "sample.npy")
            closest_path = os.path.join(new_dir, "closest.npy")
            voxel_path = os.path.join(new_dir, "voxel.binvox")

            # Check existence of processed data
            if not os.path.isfile(closest_path):
                bad_record.write("CLOSEST: |" + _class + "_" + obj + "|\n")

            if not os.path.isfile(voxel_path):
                bad_record.write("VOXEL: |" + _class + "_" + obj + "|\n")

            if not os.path.isfile(sample_path):
                bad_record.write("SAMPLE: |" + _class + "_" + obj + "|\n")
                continue

            # Check sampling points
            sample = np.load(sample_path)
            if sample.shape != (N_SAMPLE, 3):
                bad_record.write("FEW: (%s, %s) |"%sample.shape)
                bad_record.write(_class + "_" + obj + "|\n")

check_data(os.path.join(DATA_DIR, "train"))
check_data(os.path.join(DATA_DIR, "test"))

bad_record.close()
