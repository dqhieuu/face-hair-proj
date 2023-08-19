# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread


from models.FLAME import FLAME, FLAMETex
from models.decoders import Generator
from models.encoders import ResnetEncoder
from utils import util
from utils.config import cfg
from utils.renderer import SRenderY, set_rasterizer
from utils.rotation_converter import batch_euler2axis

from datasets import datasets
from utils.config import cfg as deca_cfg
from tqdm import tqdm
import torch

torch.backends.cudnn.benchmark = True


class DECA(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super().__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

    def _setup_renderer(self, model_cfg):
        set_rasterizer(self.cfg.rasterizer_type)
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size,
                               rasterizer_type=self.cfg.rasterizer_type).to(self.device)
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32) / 255.
        mean_texture = torch.from_numpy(mean_texture.transpose(2, 0, 1))[None, :, :, :].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam,
                         model_cfg.n_light]
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)
        self.my_emotion_encoder = ResnetEncoder(outsize=self.n_param).to(self.device)
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.device)
        self.D_detail = Generator(latent_dim=self.n_detail + self.n_cond, out_channels=1, out_scale=model_cfg.max_z,
                                  sample_mode='bilinear').to(self.device)
        # resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
        else:
            print(f'please check model path: {model_path}')
            # exit()
        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    # @torch.no_grad()
    # emotion is a tensor of shape [batch_size, 6] range from -1 to 1
    def encode(self, images, emotion, use_detail=True):
        if use_detail:
            # use_detail is for training detail model, need to set coarse model as eval mode
            with torch.no_grad():
                parameters = self.E_flame(images)
        else:
            parameters = self.E_flame(images)
        # parameters = self.my_emotion_encoder(parameters, emotion)


        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images
        if use_detail:
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode
        if self.cfg.model.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:, 3:].clone()  # x for yaw (open mouth), y for pitch (left ang right), z for roll
            posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose
        return codedict

    def decode(self, codedict, rendering=True, iddict=None, vis_lmk=True, return_vis=True, use_detail=True,
               render_orig=False, original_image=None, tform=None):
        images = codedict['images']
        batch_size = images.shape[0]

        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'],
                                                     pose_params=codedict['pose'])
        if self.cfg.model.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device)
        landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:, :, :2];
        landmarks2d[:, :, 1:] = -landmarks2d[:, :,
                                 1:]  # ; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']);
        landmarks3d[:, :, 1:] = -landmarks3d[:, :,
                                 1:]  # ; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        trans_verts = util.batch_orth_proj(verts, codedict['cam']);
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'landmarks3d_world': landmarks3d_world,
        }

        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo

        return opdict

    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        util.write_obj(filename, vertices, faces)

    def model_dict(self):
        return {
            'E_flame': self.E_flame.state_dict(),
            'E_detail': self.E_detail.state_dict(),
            'D_detail': self.D_detail.state_dict()
        }

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    savefolder = 'encoder_output'
    device = 'cuda'
    os.makedirs(savefolder, exist_ok=True)

    dataset = datasets.TestData('vkist_expression_test/output')

    deca_cfg.rasterizer_type = 'standard'

    deca = DECA(config=deca_cfg, device=device)

    with torch.no_grad():
        for dataunit in dataset:
            images = dataunit['image'].to(device)[None, ...]
            name = dataunit['imagename']
            print(name)

            # V000131_S002_L2_E01_C7.JPG -> match the expression code in 'E01' = 1
            emotion_code = int(name.split('_')[3][1:3])
            emotion_arr = np.zeros(6).astype(np.float32)
            # mapping
            # 1: neutral -> [0, 0, 0, 0, 0, 0]
            # 2: happy -> [1, 0, 0, 0, 0, 0]
            # 3: sad -> [0, 1, 0, 0, 0, 0]
            # 4: fear -> [0, 0, 1, 0, 0, 0]
            # 5: angry -> [0, 0, 0, 1, 0, 0]
            # 6: surprise -> [0, 0, 0, 0, 1, 0]
            # 7: disgust -> [0, 0, 0, 0, 0, 1]
            if emotion_code >= 2:
                emotion_arr[emotion_code - 2] = 1.0

            codedict = deca.encode(images, torch.from_numpy(emotion_arr))
            opdict = deca.decode(codedict)

            res = {
                'shape': codedict['shape'].cpu().numpy()[0],
                'exp': codedict['exp'].cpu().numpy()[0],
                'pose': codedict['pose'].cpu().numpy()[0],
                'my_emotion': emotion_arr,
            }

            os.makedirs(os.path.join(savefolder, name), exist_ok=True)

            np.save(os.path.join(savefolder, name, 'params.npy'), res)
            with open(os.path.join(savefolder, name, 'params.json'), 'w') as f:
                json.dump(res, f, cls=NumpyEncoder)
            deca.save_obj(os.path.join(savefolder, name, 'coarse.obj'), opdict)

    print(f'-- please check the results in {savefolder}')

main()