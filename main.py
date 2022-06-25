import os
import random
import numpy as np

import torch

from prs import PRSRunner
from settings import *

SEED = 1

# ------------------------------------START OF EVERYTHING------------------------------

if __name__ == "__main__":
    # parse arg

    # parse config file

    # Seed everything
    #os.environ[]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    run_prs = PRSRunner()

    if NEED_TRAIN:
        run_prs.train_stage()

    if NEED_TEST:
        run_prs.test_stage()
