# ---------------------------------DEFAULT SETTINGS------------------------

from numpy import pi

# Train and test options
GPU = 0
MODEL_PATH = "./checkpoint/"
DATA_DIR = "../data"
SPLIT_DIR = "data_split"
RESULT_DIR = "results"
EVAL_DIR = "evaluate"

TOTAL_ITER = 10000
LOG_STEP = 1000
CONTINUE = True
NEED_TRAIN = True
NEED_TEST = True

# Data preprocess options
MULTI_PREPROCESS = True
ORIGINAL_DATA_DIR = "../shapenet"

# Sample
NEED_SAMPLE = True
PCL_PATH = "D:\\Softwares\\PCL 1.12.1\\bin\\pcl_mesh_sampling.exe"

LARGE_SAMPLE = 2000
N_SAMPLE = 1000

# Voxel
NEED_VOXEL = True
VOXEL_SIZE = 32
# Lower and upper bound of mesh vertices
LOWER_BOUND = -0.5
UPPER_BOUND = 0.5
GRID_SIZE = 1
BINVOX_CMD = "binvox.exe"
CUDA_VOX_CMD = "./cuda_voxelizer"

# Closest
NEED_CLOSEST = True
# Record of bad model
BAD_MODEL_RECORD = "bad.txt"

# If interrupted, below is used
UNPROCESSED = None

# 'a02691156', 'a02747177', 'a02773838', 'a02801938',
# 'a02808440', 'a02818832', 'a02828884', 'a02843684',
# 'a02871439', 'a02876657', 'a02880940', 'a02924116',
# 'a02933112', 'a02942699', 'a02946921', 'a02954340',
# 'a02958343', 'a02992529', 'a03001627', 'a03046257',
# 'a03085013', 'a03207941', 'a03211117', 'a03261776',
# 'a03325088', 'a03337140', 'a03467517', 'a03513137',
# 'a03593526', 'a03624134', 'a03636649', 'a03642806',
# 'a03691459', 'a03710193', 'a03759954', 'a03761084',
# 'a03790512', 'a03797390', 'a03928116', 'a03938244',
# 'a03948459', 'a03991062', 'a04004475', 'a04074963',
# 'a04090263', 'a04099429', 'a04225987', 'a04256520',
# 'a04330267', 'a04379243', 'a04401088', 'a04460130',
# 'a04468005', 'a04530566', 'a04554684'

# Data augmentation
AUG_DIR = "../augment"
NUM_AUG = 4000

# Visualiation
USE_MAYA = True

# Model parameters
BATCH_SIZE = 32
LEARNING = 1e-2
BETA1 = 0.9
BETA2 = 0.999

IN_CHANNEL_0 = 1
OUT_CHANNEL_0 = 4
NUM_CONV_LAYERS = 5
NUM_PLANE = 3
NUM_ROT = 3
# >=25 is recommended
REGULAR_WEIGHT = 50
LOSS_THRESHOLD = 6e-4 * N_SAMPLE
ANGLE_THRESHOLD = pi / 6
