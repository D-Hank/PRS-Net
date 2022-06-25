import os
import shutil
import numpy as np
import open3d as o3d
import igl

from multiprocessing import Process
from typing import List
from tqdm import tqdm
from settings import *

bad_model = []
bad_record = open(BAD_MODEL_RECORD, "w")

# Numpy version of push out mapping: voxel grid index -> grid center in mesh space
def push_out(voxel_grid: np.ndarray):
    # grid: (VS ^ 3 , 3)
    # center for grid: [0, 1) -> 0.5
    cube_coord = (voxel_grid + GRID_SIZE / 2) / VOXEL_SIZE
    # Mesh coordinates for each grid center
    mesh_coord = cube_coord + LOWER_BOUND

    return mesh_coord

# Numpy version of pull back mapping: points in mesh space -> grid center in voxel space
def pull_back(mesh_coord: np.ndarray):
    # mesh_coord: (N_points, 3)
    cube_coord = mesh_coord - LOWER_BOUND
    grid_center = cube_coord * VOXEL_SIZE
    grid_bound = grid_center - GRID_SIZE / 2

    return grid_bound

def gen_voxel(voxel_path: str, model_path: str):
    # Read vertices and faces
    result = os.system(CUDA_VOX_CMD + " -f " + model_path + " -s " + str(VOXEL_SIZE) + " -o binvox -forceb -silent")
    # Skip bad
    if result == 0:
        old_voxel_path = model_path + "_" + str(VOXEL_SIZE) + ".binvox";
        shutil.move(old_voxel_path, voxel_path)

    return result

def gen_point_cloud(cloud_path: str, sample_path: str, model_path: str):
    result = os.system("\"" + PCL_PATH + "\" " + model_path + " " + cloud_path + " -no_vis_result -n_samples " + str(LARGE_SAMPLE))
    # Skip bad mesh
    if result == 0:
        # Resample 1000 points
        pcd = np.asarray(o3d.io.read_point_cloud(cloud_path).points)
        np.random.shuffle(pcd)
        sample = pcd[0 : N_SAMPLE]
        # remove artifact
        os.remove(cloud_path)

        # Save new
        print("Saving sample: ", sample_path, "...")
        np.save(sample_path, sample)

    return result

def gen_closest_points(closest_path: str, model_path: str):
    x, y, z = np.mgrid[0 : VOXEL_SIZE, 0 : VOXEL_SIZE, 0 : VOXEL_SIZE]
    # [(0,0,0), (0,0,1), (0,0,2), ..., (VS-1,VS-1,VS-1)]
    # For Descates coordinates, z fastest, then y, then x
    grid = np.concatenate((
        np.expand_dims(x.flatten(), axis = -1),
        np.expand_dims(y.flatten(), axis = -1),
        np.expand_dims(z.flatten(), axis = -1)), axis = -1).astype(np.float32)

    # Push out: get mesh coordinates for points/center in each voxel
    mesh_coord = push_out(grid)

    # Get the barycentric coordinates of closest points on the surface
    v, f = igl.read_triangle_mesh(model_path)
    dist, idx, closest_points = igl.point_mesh_squared_distance(mesh_coord, v, f)

    # Use float16 to save space
    np.save(closest_path, closest_points.astype(np.float16))


# Process augmented data
def preprocess(mode: str, unprocessed: List[str]):
    aug_dir = os.path.join(AUG_DIR, mode)
    data_dir = os.path.join(DATA_DIR, mode)
    for _class in os.listdir(aug_dir):
        class_path = os.path.join(aug_dir, _class)

        # May be taxonomy?
        if not os.path.isdir(class_path):
            continue

        # Skip processed classes (only for interrupt)
        if (unprocessed != None) and (not _class in unprocessed):
            continue

        print("Entering class: ", _class)
        loop = tqdm(os.listdir(class_path))
        #print(loop)
        for obj in loop:
            #print(obj)
            model_dir = os.path.join(class_path, obj)
            model_path = os.path.join(model_dir, "model_normalized.obj")

            # new data
            new_dir = os.path.join(data_dir, _class, obj)
            cloud_path = os.path.join(new_dir, "cloud.pcd")
            sample_path = os.path.join(new_dir, "sample.npy") # .npy
            closest_path = os.path.join(new_dir, "closest.npy") # .npy

            if not os.path.isdir(new_dir):
                os.makedirs(new_dir)

            # Skip bad models
            if not os.path.isfile(model_path):
                bad_record.write("EMPTY: " + model_path + "\n")
                continue

            # Generate 32*32*32 voxel
            new_voxel_path = os.path.join(new_dir, "voxel.binvox")
            if (not os.path.isfile(new_voxel_path)) and NEED_VOXEL:
                status = gen_voxel(new_voxel_path, model_path)
                # Skip bad voxel
                if status != 0:
                    bad_record.write("VOX: " + model_path + "\n")
                    continue

            # Sample for point cloud
            if (not os.path.isfile(sample_path)) and NEED_SAMPLE:
                status = gen_point_cloud(cloud_path, sample_path, model_path)
                # Skip bad mesh
                if status != 0:
                    bad_record.write("PCL: " + model_path + "\n")
                    continue

            # Generate a regular grid
            if (not os.path.isfile(closest_path)) and NEED_CLOSEST:
                gen_closest_points(closest_path, model_path)

        print("Exit class: ", _class)


# ------------------------START OF PREPROCESS-------------------

# Check for multiprocessing
if __name__ == '__main__':
    if MULTI_PREPROCESS:
        NUM_SUB = 4
        INTERVAL = len(UNPROCESSED) // NUM_SUB
        sub = []
        # Fork child process
        for i in range(NUM_SUB):
            start = i * INTERVAL
            end = (i + 1) * INTERVAL
            print("Child process: ", start, "to", end)
            unprocessed = UNPROCESSED[start : end]
            child = Process(target = preprocess, args = ("train", unprocessed))
            child.start()
            sub.append(child)

        # Join child process
        for i in range(NUM_SUB):
            sub[i].join()

        print("All done!")

    # Use simple plan
    else:
        print("Main process")
        preprocess("train", UNPROCESSED)

    preprocess("test", UNPROCESSED)

bad_record.close()
