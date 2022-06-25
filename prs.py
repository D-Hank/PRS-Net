import time
import os
import glob
import torch
import numpy as np

from settings import *
from model import PRSModel
from dataset import PRSDataset
from visualize import MatPlotVisualization, MayaVisualization

from typing import List, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------NOTICE--------------------------------
# 1. Translate and scale are ignored after preprocessing .The model is already normalized, so we don't need them



# ---------------------------------DEFAULT SETTINGS------------------------

writer = None
device = None


# ----------------------------------START OF THE ALGORITHM----------------------------

class PRSRunner():
    def __init__(
        self,
        gpu = GPU,
        data_dir = DATA_DIR,
        aug_dir = AUG_DIR,
        split_dir = SPLIT_DIR,
        bad_list = BAD_MODEL_RECORD,
        ckpt_path = MODEL_PATH,
        result_dir = RESULT_DIR,
        eval_dir = EVAL_DIR,
        total_iter = TOTAL_ITER,
        batch_size = BATCH_SIZE,
        num_plane = NUM_PLANE,
        num_rot = NUM_ROT,
        learning = LEARNING,
        log_step = LOG_STEP,
        continue_ = CONTINUE,
        loss_threshold = LOSS_THRESHOLD,
        angle_threshold = ANGLE_THRESHOLD,
        use_maya = USE_MAYA
    ):

        # ---------------------------GLOBAL--------------------------
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        global device, writer

        writer = SummaryWriter()
        device = torch.device("cuda:" + str(gpu)) if torch.cuda.is_available() else torch.device("cpu")
        print("Using device", device)

        self.visualizer = None
        self.vis_strategy = MayaVisualization if use_maya else MatPlotVisualization

        self.start_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        print("Start at time: ", self.start_time)

        self.model = PRSModel(
            device = device,
            in_channel_0 = IN_CHANNEL_0,
            out_channel_0 = OUT_CHANNEL_0,
            num_conv_layers = NUM_CONV_LAYERS,
            num_plane = num_plane,
            num_rot = num_rot
            ).to(device)

        self.result_dir = result_dir
        self.ckpt_path = ckpt_path
        self.eval_dir = eval_dir
        self.aug_dir = aug_dir

        self.total_iter = total_iter
        self.log_step = log_step

        # Check existing checkpoint
        ck_list = glob.glob(ckpt_path + "*.pkl")
        last_iter = -1
        if continue_ and ck_list:
            for file in ck_list:
                ck = file.split("_")[-1]
                it = int(ck[ : -4])
                if it >= last_iter:
                    last_iter = it
                    last_ckpt = file

            print("Last iter: ", last_iter)
            self.model = torch.load(last_ckpt).to(device)

        else:
            print("New running created.")

        self.last_iter = last_iter

        self.batch_size = batch_size
        self.num_plane = num_plane
        self.num_rot = num_rot
        self.loss_threshold = loss_threshold
        self.angle_threshold = np.cos(angle_threshold)

        # --------------------------------TRAIN-----------------------------
        self.train_dataset = PRSDataset(data_dir = data_dir, split_dir = split_dir, bad_list = bad_list, transform = torch.Tensor, mode = "train")
        self.train_dataloader = DataLoader(dataset = self.train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)
        self.optimizer = torch.optim.Adam(
            [{"params": self.model.parameters(), "initial_lr": learning}],
            lr = learning,
            betas = (BETA1, BETA2))

        self.train_obj = self.train_dataset.num_obj

        # --------------------------------TEST------------------------------
        self.test_dataset = PRSDataset(data_dir = data_dir, split_dir = split_dir, bad_list = bad_list, transform = torch.Tensor, mode = "test")
        self.test_dataloader = DataLoader(dataset = self.test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4, drop_last = True)
        self.test_obj = self.test_dataset.num_obj

    def save_params(
        self,
        label: List[str],
        plane_params: torch.Tensor,
        rot_params: torch.Tensor,
        each_ref_dist: torch.Tensor,
        each_rot_dist: torch.Tensor,
        valid_mask: Dict[str, torch.Tensor]
        ):
        # plane/rot_params: (N_batch, N_sym, 4)
        # each_ref/rot_dist: (N_batch, N_sym)
        plane_params = plane_params.detach().cpu().numpy()
        rot_params = rot_params.detach().cpu().numpy()
        each_ref_dist = each_ref_dist.detach().cpu().numpy()
        each_rot_dist = each_rot_dist.detach().cpu().numpy()
        plane_loss_mask = valid_mask['plane_sdl'].detach().cpu().numpy()
        rot_loss_mask = valid_mask['rot_sdl'].detach().cpu().numpy()
        plane_angle_mask = valid_mask['plane_angle'].detach().cpu().numpy()
        rot_angle_mask = valid_mask['rot_angle'].detach().cpu().numpy()
        # Save each model in this batch
        for i in range(0, self.batch_size):
            _class, obj = label[i].split("_", maxsplit = 1)
            obj_path = os.path.join(self.eval_dir, _class, obj)
            param_path = os.path.join(obj_path, "param.npz")

            if not os.path.isdir(obj_path):
                os.makedirs(obj_path)

            save_dict = {
                'plane_params': plane_params[i],
                'rot_params': rot_params[i],
                'each_ref_dist': each_ref_dist[i],
                'each_rot_dist': each_rot_dist[i],
                'plane_loss_mask': plane_loss_mask[i],
                'rot_loss_mask': rot_loss_mask[i],
                'plane_angle_mask': plane_angle_mask[i],
                'rot_angle_maks': rot_angle_mask[i]
            }

            np.savez(param_path, **save_dict)

    def save_model(self, iter):
        torch.save(self.model, self.ckpt_path + self.start_time + "_" + str(iter) + ".pkl")

    def save_log(self, mode: str, iter: int, log_dict: dict):
        for key, value in log_dict.items():
            writer.add_scalar(key + "/" + mode, value, iter)

        writer.flush()

    def visual_new(self, mode: str, sample_points: torch.Tensor, label: str):
        # points: (N_sample, 3)
        _class, obj = label.split(sep = "_", maxsplit = 1)
        self.visualizer = self.vis_strategy(
            sample_points.detach().cpu().numpy(),
            os.path.join(self.aug_dir, mode, _class, obj, "model_normalized.obj"),
            label
            )

    def visual_reflect(self, trans_points: torch.Tensor, params: torch.Tensor, mask: torch.Tensor = None):
        # points: (N_plane, N_sample, 3)
        # params: (N_plane, 4)
        # mask: (N_plane)
        # Notice: Check validity if necessary
        for i in range(self.num_plane):
            if mask == None or mask[i] >= 0.5:
                self.visualizer.add_reflect(trans_points[i].detach().cpu().numpy(), params[i].detach().cpu().numpy())

    def visual_rotate(self, trans_points: torch.Tensor, params: torch.Tensor, mask: torch.Tensor = None):
        # points: (N_rot, N_sample, 3)
        # params: (N_rot, 4)
        # mask: (N_rot)
        for i in range(self.num_rot):
            if mask == None or mask[i] >= 0.5:
                self.visualizer.add_rotate(trans_points[i].detach().cpu().numpy(), params[i].detach().cpu().numpy())

    def visual_match(self, trans_points: torch.Tensor, close_points: torch.Tensor):
        # points: (N_sym, N_sample, 3)
        self.visualizer.match_point(trans_points[0].detach().cpu().numpy(), close_points[0].detach().cpu().numpy())

    def visual_save(
        self,
        iter,
        name,
        ref_points: torch.Tensor = None,
        plane_params: torch.Tensor = None,
        rot_points: torch.Tensor = None,
        rot_params: torch.Tensor = None
        ):
        #self.visual_reflect(ref_points, plane_params)
        self.visualizer.save_fig(os.path.join(self.result_dir, str(iter) + "_" + name + ".svg"))

    def train_stage(self):
        self.model.train()
        iter = self.last_iter + 1
        end_iter = self.total_iter
        log_step = self.log_step

        avg_loss = 0.0
        while (iter < end_iter):
            print("\n[ITER]\n", iter)
            loop = tqdm(enumerate(self.train_dataloader), total = len(self.train_dataloader))

            for index, (closest, sample, voxel, label) in loop:
                #
                closest = closest.to(device)
                sample = sample.to(device)
                voxel = voxel.to(device)
                # For batch
                self.optimizer.zero_grad()
                plane_params, rot_params = self.model(voxel)

                total_loss, loss_dict = self.model.get_loss(sample, plane_params, rot_params, closest)

                total_loss.backward()
                self.optimizer.step()
                avg_loss += float(total_loss)

                log_dict = {
                    "total_loss": total_loss,
                    "total_ref_dist": loss_dict['total_ref_dist'],
                    "total_rot_dist": loss_dict['total_rot_dist'],
                    "total_ref_reg": loss_dict['total_ref_reg'],
                    "total_rot_reg": loss_dict['total_rot_reg'],
                    "lr": self.optimizer.state_dict()['param_groups'][0]['lr']
                }
                self.save_log(mode = "train", iter = iter, log_dict = log_dict)

                if ((iter + 1) % log_step) == 0:
                    print("\n[INDEX]", index, " [LOSS] %.4f"%(avg_loss / log_step))
                    avg_loss = 0.0

                    self.save_model(iter)

                    self.visual_new("train", sample[0], label[0])
                    #self.visual_match(self.model.ref_trans_points[0], self.model.reflected_closest[0])
                    self.visual_reflect(self.model.ref_trans_points[0], plane_params[0])
                    self.visual_save(iter, "p0")
                    #self.visual_match(self.model.rot_trans_points[0], self.model.rotated_closest[0])
                    #self.visual_rotate(self.model.rot_trans_points[0], rot_params[0])
                    self.visual_save(iter, "r0")

                iter += 1

                if (iter >= end_iter):
                    break


    def test_stage(self):
        self.model.eval()
        print("Start testing the model...")

        with torch.no_grad():
            loop = tqdm(enumerate(self.test_dataloader), total = len(self.test_dataloader))

            iter = 0
            for index, (closest, sample, voxel, label) in loop:
                closest = closest.to(device)
                sample = sample.to(device)
                voxel = voxel.to(device)
                plane_params, rot_params = self.model(voxel)

                total_loss, loss_dict = self.model.get_loss(sample, plane_params, rot_params, closest)

                log_dict = {
                    "total_loss": total_loss,
                    "total_ref_dist": loss_dict['total_ref_dist'],
                    "total_rot_dist": loss_dict['total_rot_dist'],
                    "total_ref_reg": loss_dict['total_ref_reg'],
                    "total_rot_reg": loss_dict['total_rot_reg'],
                }
                self.save_log(mode = "test", iter = iter, log_dict = log_dict)

                mask_dict = self.model.validate_remove(
                    plane_params,
                    rot_params,
                    loss_dict['each_ref_dist'],
                    loss_dict['each_rot_dist'],
                    self.loss_threshold,
                    self.angle_threshold
                    )
                self.save_params(label, plane_params, rot_params, loss_dict['each_ref_dist'], loss_dict['each_rot_dist'], mask_dict)

                if ((iter + 1) % 3) == 0:
                    self.visual_new("test", sample[0], label[0])
                    self.visual_reflect(self.model.ref_trans_points[0], plane_params[0], mask_dict['plane_sdl'][0])
                    self.visual_rotate(self.model.rot_trans_points[0], rot_params[0], mask_dict['rot_sdl'][0])
                    self.visual_save(0, label[0])

                iter += 1

