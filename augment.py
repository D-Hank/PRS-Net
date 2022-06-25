import os
import random
import numpy as np
import point_cloud_utils as pcu

from typing import Dict, List
from settings import *

# Do a random transformation on a model
def random_trans(origin_nodes: np.ndarray):
    # origin_nodes: (N_nodes, 3)
    direction = np.random.rand(3)
    x, y, z = direction / (np.linalg.norm(direction) + 1e-12)
    theta = 2 * np.pi * np.random.rand()
    cos = np.cos(theta)
    sin = np.sin(theta)
    _1_cos_x = (1 - cos) * x
    _1_cos_y = (1 - cos) * y
    # Rodrigues formula
    matrix = np.array([
        [cos + _1_cos_x * x    , _1_cos_x * y - sin * z, _1_cos_x * z + sin * y ],
        [_1_cos_y * x + sin * z, cos + _1_cos_y * y    , _1_cos_y * z - sin * x ],
        [_1_cos_x * z - sin * y, _1_cos_y * z + sin * x, cos + (1 - cos) * z * z]
        ]).astype(np.float32)

    # (N, 3) * (3, 3) -> (N, 3)
    trans_nodes = np.matmul(origin_nodes, matrix.transpose(0, 1))

    return trans_nodes

# Augment a category
def aug_category(class_path: str, _class: str, obj_list: List, count: Dict[str, int], mode: str):
    for obj in obj_list:
        model_dir = os.path.join(class_path, obj, "models")
        model_path = os.path.join(model_dir, "model_normalized.obj")

        # Skip bad models
        if not os.path.isfile(model_path):
            continue

        # Read original obj model
        old_v, f = pcu.load_mesh_vf(model_path)
        # For each sample one, generate new model
        for i in range(0, count[obj]):
            # new data
            new_dir = os.path.join(AUG_DIR, mode, "a" + _class, obj + "_" + str(i))
            new_path = os.path.join(new_dir, "model_normalized.obj")

            # Skip already processed
            if os.path.isfile(new_path):
                continue

            # Make new dir
            if not os.path.isdir(new_dir):
                os.makedirs(new_dir)

            # Save new
            new_v = random_trans(old_v)
            pcu.save_mesh_vf(new_path, new_v, f)


for _class in os.listdir(ORIGINAL_DATA_DIR):
    class_path = os.path.join(ORIGINAL_DATA_DIR, _class)

    # Skip taxonomy
    if not os.path.isdir(class_path):
        continue

    # Skip processed classes
    if (UNPROCESSED != None) and (not _class in UNPROCESSED):
        continue

    print("Entering class: ", _class)

    # Read split file
    train_obj = []
    test_obj = []
    with open(os.path.join(SPLIT_DIR, _class + "_train.txt"), mode = "r") as train_file:
        for obj in train_file:
            obj = obj.strip("\n")
            train_obj.append(obj)

    with open(os.path.join(SPLIT_DIR, _class + "_test.txt"), mode = "r") as test_file:
        for obj in test_file:
            obj = obj.strip("\n")
            test_obj.append(obj)

    # Rest of train objs
    sample_objs = train_obj
    random.shuffle(sample_objs)
    sample_objs = sample_objs[0 : NUM_AUG % len(train_obj)]

    # Count sampling times for each train obj
    # Avoid opening a file multiple times
    num_per_obj = NUM_AUG // len(train_obj)
    train_count = {}
    for sample in set(train_obj):
        train_count[sample] = sample_objs.count(sample) + num_per_obj

    aug_category(class_path, _class, train_obj, train_count, "train")

    # Count for test set
    test_count = {}
    for sample in set(test_obj):
        test_count[sample] = 1

    aug_category(class_path, _class, test_obj, test_count, "test")
