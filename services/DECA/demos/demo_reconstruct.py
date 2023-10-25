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
import os, sys
import shutil

import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points


def emotion_code_to_input_tensor(emotion_code: int):
    emotion_arr = np.zeros(6).astype(np.float32)
    if emotion_code >= 2:
        emotion_arr[emotion_code - 2] = 1.0
    return torch.tensor([emotion_arr]).to('cuda')


def emotion_code_to_meaning(emotion_code: int):
    return ['neutral', 'happy', 'sad', 'fear', 'anger', 'surprise', 'disgust'][emotion_code - 1]


from decalib.emotion_net import get_emotion_model
emotion_model = get_emotion_model(use_flame_shape=False)


def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector,
                                 sample_step=args.sample_step)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)
    # for i in range(len(testdata)):
    for i in tqdm(range(len(testdata))):
        for emo_code in range(1, 8):
            name = testdata[i]['imagename']
            images = testdata[i]['image'].to(device)[None, ...]
            with torch.no_grad():
                codedict = deca.encode(images)
                opdict_pre_exp, visdict_pre_exp = deca.decode(codedict)
                exp, pose = emotion_model(emotion_code_to_input_tensor(emo_code), codedict['shape'])
                codedict['exp'] = exp
                codedict['pose'] = pose
                opdict, visdict = deca.decode(codedict)  # tensor
                opdict['uv_texture_gt'] = opdict_pre_exp['uv_texture_gt']
                if args.render_orig:
                    tform = testdata[i]['tform'][None, ...]
                    tform = torch.inverse(tform).transpose(1, 2).to(device)
                    original_image = testdata[i]['original_image'][None, ...].to(device)
                    _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image,
                                                  tform=tform)
                    orig_visdict['inputs'] = original_image

            if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
                os.makedirs(os.path.join(savefolder, name), exist_ok=True)
            # -- save results
            if args.saveDepth:
                depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
                visdict['depth_images'] = depth_image
                cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
            if args.saveKpt:
                np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
                np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
            if args.saveObj:
                deca.save_obj(
                    os.path.join(savefolder, name, name + f'_emo_{emotion_code_to_meaning(emo_code)}' + '.obj'), opdict)
            if args.saveMat:
                opdict = util.dict_tensor2npy(opdict)
                savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
            if args.saveVis:
                cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
                if args.render_orig:
                    cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
            if args.saveImages:
                for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images',
                                 'landmarks2d']:
                    if vis_name not in visdict.keys():
                        continue
                    image = util.tensor2image(visdict[vis_name][0])
                    cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name + '.jpg'),
                                util.tensor2image(visdict[vis_name][0]))
                    if args.render_orig:
                        image = util.tensor2image(orig_visdict[vis_name][0])
                        cv2.imwrite(os.path.join(savefolder, name, 'orig_' + name + '_' + vis_name + '.jpg'),
                                    util.tensor2image(orig_visdict[vis_name][0]))
        print(f'-- please check the results in {savefolder}')


def init_my_deca():
    # run DECA
    deca_cfg.model.use_tex = True
    deca_cfg.model.extract_tex = True
    deca_cfg.rasterizer_type = 'standard'
    device = 'cuda'
    savefolder = 'output'
    os.makedirs(savefolder, exist_ok=True)
    return DECA(config=deca_cfg, device=device)


def output_from_image(img: np.array, deca, no_detect_pose=True, emotion_arr=None, exp_arr=None, pose_arr=None,
                      neck_pose_arr=None, eye_pose_arr=None):
    device = 'cuda'
    savefolder = 'output'
    tempfolder = f'temp/{time()}'
    # write image to temp folder
    os.makedirs(tempfolder, exist_ok=True)
    cv2.imwrite(os.path.join(tempfolder, 'temp.jpg'), img)

    # load test images
    testdata = datasets.TestData(tempfolder)

    # for i in tqdm(range(len(testdata))):
    name = f"{testdata[0]['imagename']}_{time()}"
    images = testdata[0]['image'].to(device)[None, ...]
    with torch.no_grad():
        codedict = deca.encode(images)
        opdict_pre_exp, visdict_pre_exp = deca.decode(codedict)

        if no_detect_pose:
            codedict['pose'] = torch.tensor([[0.0] * 6]).to(device)
            codedict['exp'] = torch.tensor([[0.0] * 50]).to(device)

        if emotion_arr is not None:
            exp, pose = emotion_model(torch.tensor([emotion_arr], dtype=torch.float32).to(device), codedict['shape'])
            codedict['exp'] = exp
            codedict['pose'][:3] = pose[:3]  # only jaw pose

        if exp_arr is not None:
            codedict['exp'] = torch.tensor([exp_arr]).to(device)

        if pose_arr is not None:
            codedict['pose'] = torch.tensor([pose_arr]).to(device)

        if neck_pose_arr is not None:
            codedict['neck_pose'] = torch.tensor([neck_pose_arr]).to(device)

        if eye_pose_arr is not None:
            codedict['eye_pose'] = torch.tensor([eye_pose_arr]).to(device)

        opdict, visdict = deca.decode(codedict)  # tensor
        # correct uv mapping
        opdict['uv_texture_gt'] = opdict_pre_exp['uv_texture_gt']

        os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
    print(os.path.join(savefolder, name, name + '.obj'))

    verts = opdict['verts'][0].cpu().numpy().tolist()

    five_perc_points_highest_z_head = sorted(verts, key=lambda x: x[1])[-int(len(verts) * 0.05):]
    mean_x_head = sum([v[0] for v in five_perc_points_highest_z_head]) / len(five_perc_points_highest_z_head)
    mean_y_head = sum([v[2] for v in five_perc_points_highest_z_head]) / len(five_perc_points_highest_z_head)
    highest_z_head = max(five_perc_points_highest_z_head, key=lambda x: x[1])[1]

    # read file with trimesh
    import trimesh
    mesh = trimesh.load('my_data/hair/short.obj')

    five_perc_points_highest_z_hair = sorted(mesh.vertices, key=lambda x: x[1])[-int(len(mesh.vertices) * 0.05):]
    mean_x_hair = sum([v[0] for v in five_perc_points_highest_z_hair]) / len(five_perc_points_highest_z_hair)
    mean_y_hair = sum([v[2] for v in five_perc_points_highest_z_hair]) / len(five_perc_points_highest_z_hair)
    highest_z_hair = max(five_perc_points_highest_z_hair, key=lambda x: x[1])[1]

    hair_calibration = [0, 0.08, 0]
    hair_translation = [mean_x_head - mean_x_hair + hair_calibration[0],
                        highest_z_head - highest_z_hair + hair_calibration[1],
                        mean_y_head - mean_y_hair + hair_calibration[2]
                        ]

    five_perc_points_highest_z_hair = sorted(mesh.vertices, key=lambda x: x[1])[-int(len(mesh.vertices) * 0.10):]
    leftmost_x_hair = min(five_perc_points_highest_z_hair, key=lambda x: x[0])[0]
    rightmost_x_hair = max(five_perc_points_highest_z_hair, key=lambda x: x[0])[0]
    width_hair = rightmost_x_hair - leftmost_x_hair

    leftmost_head = verts[3039]
    rightmost_head = verts[806]
    width_head = np.linalg.norm(np.array(leftmost_head) - np.array(rightmost_head))

    ratio_calibration = 0.05
    head_to_hair_ratio = width_head / width_hair + ratio_calibration

    metadata = {
        'lips': {
            'upper': {
                'center': verts[3543],
                'left': verts[2830],
                'right': verts[1713]
            },
            'lower': {
                'center': verts[3506],
                'left': verts[2713],
                'right': verts[1773]
            }
        },
        'hair_translation': hair_translation,
        'hair_scale': [head_to_hair_ratio] * 3,
    }

    # save metadata variable to savefolder/metadata.json
    with open(os.path.join(savefolder, name, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    # copy teeth.glb to savefolder
    shutil.copy('my_data/teeth.glb', os.path.join(savefolder, name, 'teeth.glb'))

    # copy hair emma.obj to savefolder
    shutil.copy('my_data/hair/short.obj', os.path.join(savefolder, name, 'hair.obj'))

    # zip all files and return
    shutil.make_archive(f'{savefolder}/{name}', 'zip', f'{savefolder}/{name}')

    # read zip file and return data
    with open(f'{savefolder}/{name}.zip', 'rb') as f:
        data = f.read()

    # remove temp folders and zip file
    shutil.rmtree(tempfolder)
    shutil.rmtree(f'{savefolder}/{name}')
    os.remove(f'{savefolder}/{name}.zip')

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details')
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    main(parser.parse_args())
