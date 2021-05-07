import re
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from .utils import save, load, plot_prob
from .config import JsonConfig
sys.path.append('./motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
import scipy.ndimage.filters as filters
from .models import Glow
from . import thops
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import tikzplotlib
from sklearn.decomposition import PCA
import json

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from scipy.spatial.transform import Rotation as R

stylenames = ["angry", "childlike", "depressed", "neutral", "old", "proud", "sexy", "strutting"]

def get_translations(clip, trans):
    rot = R.from_quat([0, 0, 0, 1])
    translation = np.array([[0, 0, trans]])
    translations = np.zeros((clip.shape[0], 3))

    joints, root_dx, root_dz, root_dr = clip[:, :-3], clip[:, -3], clip[:, -2], clip[:, -1]
    joints = joints.reshape((len(joints), -1, 3))
    for i in range(len(joints)):
        joints[i, :, :] = rot.apply(joints[i])
        joints[i, :, 0] = joints[i, :, 0] + translation[0, 0]
        joints[i, :, 2] = joints[i, :, 2] + translation[0, 2]
        rot = R.from_rotvec(np.array([0, -root_dr[i], 0])) * rot
        translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
        translations[i, :] = translation
    return translations, joints

def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled

def fit_and_standardize(data):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler
def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)

def process_mxiafile(filename, window=32, window_step=8):
    anim, names, frametime = BVH.load(filename)

    """ Convert to 60 fps """
    anim = anim[::4]

    """ Do FK """
    global_positions = Animation.positions_global(anim)

    """ Remove Uneeded Joints """
    positions = global_positions[:, np.array([
        0,
        2, 3, 4, 5,
        7, 8, 9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]

    """ Put on Floor """
    fid_l, fid_r = np.array([4, 5]), np.array([8, 9])
    foot_heights = np.minimum(positions[:, fid_l, 1], positions[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)

    positions[:, :, 1] -= floor_height

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    reference = positions[:, 0] * np.array([1, 0, 1])
    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')
    # positions = np.concatenate([reference[:, np.newaxis], positions], axis=1)

    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.05, 0.05]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)

    """ Get Root Velocity """
    velocity = (positions[1:, 0:1] - positions[:-1, 0:1]).copy()

    """ Remove Translation """
    positions[:, :, 0] = positions[:, :, 0] - positions[:, 0:1, 0]
    positions[:, :, 2] = positions[:, :, 2] - positions[:, 0:1, 2]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 14, 18, 2, 6
    across1 = positions[:, hip_l] - positions[:, hip_r]
    across0 = positions[:, sdr_l] - positions[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    positions = rotation * positions

    """ Get Root Rotation """
    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:, :, 0]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)


    return positions

class Trainer(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 data, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # set members
        # append date info
        self.log_dir = log_dir
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")

        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        self.max_checkpoints = hparams.Train.max_checkpoints

        # model relative
        self.graph = graph
        self.optim = optim

        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm

        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device

        # number of training batches
        self.batch_size = hparams.Train.batch_size
        self.train_dataset = data.get_train_dataset()
        self.data_loader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=1,
                                      shuffle=False,
                                      drop_last=False)

        self.n_epoches = (hparams.Train.num_batches + len(self.data_loader) - 1)
        self.n_epoches = self.n_epoches // len(self.data_loader)
        self.global_step = 0

        self.seqlen = hparams.Data.seqlen
        self.n_lookahead = hparams.Data.n_lookahead

        # test batch
        self.test_data_loader = DataLoader(data.get_test_dataset(),
                                           batch_size=100,
                                           num_workers=1,
                                           shuffle=False,
                                           drop_last=True)
        self.test_batch = next(iter(self.test_data_loader))


        # validation batch
        self.val_data_loader = DataLoader(data.get_validation_dataset(),
                                          batch_size=self.batch_size,
                                          num_workers=1,
                                          shuffle=False,
                                          drop_last=True)

        self.data = data

        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step

        # log relative
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.validation_log_gaps = hparams.Train.validation_log_gap
        self.plot_gaps = hparams.Train.plot_gap



    def prepare_cond(self, jt_data, ctrl_data):
        nn, seqlen, n_feats = jt_data.shape

        jt_data = jt_data.reshape((nn, seqlen * n_feats))
        nn, seqlen, n_feats = ctrl_data.shape
        ctrl_data = ctrl_data.reshape((nn, seqlen * n_feats))
        cond = torch.from_numpy(np.expand_dims(np.concatenate((jt_data, ctrl_data), axis=1), axis=-1))

        return cond.to(self.data_device)



    def test_pair(self, stylebvh, contentbvh, eps_std=1.0, frames=1, counter=0):

        batch = self.test_batch


        style = process_mxiafile(stylebvh).astype(np.float32).reshape(1, -1, 66)
        content = process_mxiafile(contentbvh).astype(np.float32).reshape(1, -1, 66)

        style_file = stylebvh.split('/')[-1]
        content_file = contentbvh.split('/')[-1]

        style_label, style_idx, _ = style_file.split('_')
        content_label, content_idx, _ = content_file.split('_')
        label = stylenames.index(style_label)
        scaler = self.data.get_scaler()
        style = standardize(style, scaler)

        content_outfile = content_file.split('.')[0]
        style_outfile = style_file.split('.')[0]
        outfile = 'style_' + str(style_outfile) + '_content_' + str(content_outfile)



        if style.shape[1] >= content.shape[1]  and style_label != content_label:


            content = standardize(content, scaler)
            autoreg_all = batch["x"].cpu().numpy()
            control_all = batch["cond"].cpu().numpy()


            # Initialize the pose sequence with ground truth test data
            seqlen = self.seqlen
            n_lookahead = self.n_lookahead

            _, n_timesteps, _ = content.shape
            n_feats = 63
            nn = 1
            sampled_all = np.zeros((nn, n_timesteps - n_lookahead, n_feats))
            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32)  # initialize from a mean pose
            sampled_all[0, :seqlen, :] = content[:, :seqlen, :-3]

            # Loop through control sequence and generate new data
            for i in range(0, content.shape[1] - seqlen - n_lookahead):

                control = content[:, i:(i + seqlen + 1 + n_lookahead), -3:] #.reshape(1, -1, 3)

                if i == 0:
                    if style_idx == content_idx:
                        autoreg[:, 0:10, :] = style[:, 0:10, :-3]
                    else:
                        autoreg[:, 0:10, :] = content[:, 0:10, :-3]


                control2 = style[:, i:(i + seqlen + 1 + n_lookahead), -3:]
                autoreg2 = style[:, i:(i + seqlen + n_lookahead), :-3]

                ar2 = style[:, (i + seqlen + n_lookahead):(i + seqlen + n_lookahead + 1), :-3]
                ar2 = torch.from_numpy(ar2.copy())
                ar2 = np.swapaxes(ar2, 1, 2).cuda().to(self.data_device)

                ar1 = content[:, (i + seqlen + n_lookahead):(i + seqlen + n_lookahead + 1), :-3]
                ar1 = torch.from_numpy(ar1.copy())
                ar1 = np.swapaxes(ar1, 1, 2).cuda().to(self.data_device)

                cond = self.prepare_cond(autoreg.copy(), control.copy())
                cond2 = self.prepare_cond(autoreg2.copy(), control2.copy())
                label = stylenames.index(style_label)


                z_base = self.graph.generate_z(ar2, cond2)
                z_base0 = self.graph.generate_z(ar1, cond)

                sampled = self.graph(z=z_base, cond=cond, eps_std=eps_std, reverse=True)
                sampled = sampled.cpu().numpy()[:, :, 0]

                # store the sampled frame
                sampled_all[0, (i + seqlen), :] = sampled

                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:, 1:, :].copy(), sampled[:, None, :]), axis=1)


            sidx = seqlen
            control_show = content[:, sidx:(n_timesteps - n_lookahead), -3:]
            autoreg_show = content[:, sidx:(n_timesteps - n_lookahead), :-3]
            sample_show = sampled_all[0, sidx:(n_timesteps - n_lookahead), :].reshape(1, -1, 63)
            control_show2 = style[:, sidx:(n_timesteps - n_lookahead), -3:]
            autoreg_show2 = style[:, sidx:(n_timesteps - n_lookahead), :-3]

            self.data.save_animation_compare(control_show, sample_show, control_show, autoreg_show, control_show2,
                                      autoreg_show2,
                                      os.path.join(self.log_dir,
                                                   outfile))


    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self):

        self.global_step = self.loaded_step

        # begin to train
        for epoch in range(self.n_epoches):
            print("epoch", epoch)
            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):

                # set to training state
                self.graph.train()

                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])

                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)

                # # get batch data
                # for k in batch:
                #     batch[k] = batch[k].to(self.data_device)
                x = batch["x"].to(self.data_device) #100,63,70

                cond = batch["cond"].to(self.data_device) # 100, 663, 70

                # init LSTM hidden
                if hasattr(self.graph, "module"):
                    self.graph.module.init_lstm_hidden()
                else:
                    self.graph.init_lstm_hidden()

                # at first time, initialize ActNorm
                if self.global_step == 0:
                    self.graph(x[:self.batch_size // len(self.devices), ...],
                               cond[:self.batch_size // len(self.devices), ...] if cond is not None else None)
                    # re-init LSTM hidden
                    if hasattr(self.graph, "module"):
                        self.graph.module.init_lstm_hidden()
                    else:
                        self.graph.init_lstm_hidden()

                # print("n_params: " + str(self.count_parameters(self.graph)))

                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])

                # forward phase
                z, nll = self.graph(x=x, cond=cond)

                # loss
                loss_generative = Glow.loss_generative(nll)
                loss_classes = 0
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", loss_generative, self.global_step)
                loss = loss_generative

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss.backward()

                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                # step
                self.optim.step()

                if self.global_step % self.validation_log_gaps == 0:
                    # set to eval state
                    self.graph.eval()

                    # Validation forward phase
                    loss_val = 0
                    n_batches = 0
                    for ii, val_batch in enumerate(self.val_data_loader):
                        # for k in val_batch:
                        #     val_batch[k] = val_batch[k].to(self.data_device)

                        with torch.no_grad():

                            # init LSTM hidden
                            if hasattr(self.graph, "module"):
                                self.graph.module.init_lstm_hidden()
                            else:
                                self.graph.init_lstm_hidden()

                            z_val, nll_val = self.graph(x=val_batch["x"].to(self.data_device), cond=val_batch["cond"].to(self.data_device))

                            # loss
                            loss_val = loss_val + Glow.loss_generative(nll_val)
                            n_batches = n_batches + 1

                    loss_val = loss_val / n_batches
                    self.writer.add_scalar("val_loss/val_loss_generative", loss_val, self.global_step)

                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)



                # global step
                self.global_step += 1
            print(
                f'Loss: {loss.item():.5f}/ Validation Loss: {loss_val:.5f} '
            )

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
