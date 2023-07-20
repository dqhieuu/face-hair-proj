import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import plotly.express as px
import sys
import plotly.io as pio


from src.preprocessing import gauss_noise
from src.train_new import HairNetLightning
from src.model_rewrite import DucModel, DucModelNew
from hair_detection.main import input_to_mask_image
from torchvision.transforms import transforms

net: HairNetLightning

torch.set_printoptions(profile="full")



def init(weight_path):
    global net
    print("Building Network...")
    net = DucModelNew().cuda()
    net.load_state_dict(torch.load(weight_path))
    net.eval()

def init_lightning(weight_path):
    global net
    print("Building Lightning Network...")
    net = HairNetLightning.load_from_checkpoint(weight_path)
    net.cuda().eval()

def hair_mask_from_unprocessed_image(img: np.ndarray):
    img_pil = Image.fromarray(img)

    img, mask = input_to_mask_image(img_pil)
    return img

def strands_from_processed_image(img: np.ndarray):
    # img = gauss_noise(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    img = transform(img).cuda().view(1, 3, 256, 256)
    # img = img * 2 - 1
    # img *= 255

    # print(img, img.shape)

    output = net(img)

    strands = output[0].cpu().detach().numpy()  # hair strands

    gaussian = cv2.getGaussianKernel(10, 3)
    for i in range(strands.shape[2]):
        for j in range(strands.shape[3]):
            strands[:, :3, i, j] = cv2.filter2D(strands[:, :3, i, j], -1, gaussian)
    # [100, 4, 32, 32]
    # interpolated = np.zeros((100, 4, 32, 32*16))
    #
    # for i in range(strands.shape[2]):
    #     for j in range(strands.shape[3]):
    #         for k in range(16):
    #             offset_x = random.uniform(-0.01, 0.01)
    #             offset_y = random.uniform(-0.01, 0.01)
    #             offset_z = random.uniform(-0.01, 0.01)
    #             for l in range(strands.shape[0]):
    #                 interpolated[l, :, i, j*16+k] = strands[l, :, i, j] + np.array([offset_x, offset_y, offset_z, 0])
    #
    # return interpolated
    return strands

def show3DhairPlotByStrands(strands: np.ndarray):
    """
    strands: [100, 4, 32, 32] (may be interpolated)
    """
    shape = strands.shape
    strands = strands.transpose((1, 2, 3, 0)).reshape(shape[1], shape[2] * shape[3] * shape[0])

    line_indexing = []
    for i in range(shape[2] * shape[3]):
        for j in range(shape[0]):
            line_indexing.append(i)

    fig = px.line_3d(
        x=strands[0],
        y=strands[1],
        z=strands[2],
        color=line_indexing,
        range_x=[-2, 2],
        range_y=[-2, 2],
        range_z=[-2, 2],
    )
    # fig.update_layout(scene_aspectmode='cube')
    fig.write_html("test.html", auto_open=True)

def saveObj(name, strands, R = 0):
    shape = strands.shape
    # R = check_R(R)
    # for i in range(32):
    #     for j in range(32):
    #         if sum(sum(strands[:, :, i, j])) == 0:
    #             continue
    #         strand = strands[:, :, i, j]
    #         x0, y0, z0, _ = strand[0]
    #         x = strand[:, 0]
    #         y = strand[:, 1]
    #         z = strand[:, 2]
            # c = strand[:, 3]
            # d = np.sqrt((x - x0) * 2 + (y - y0) * 2 + (z - z0) ** 2)
            # for k in range(1, 99):
            #     lap_x = x[k] * 2 - x[k - 1] - x[k + 1]
            #     lap_y = y[k] * 2 - y[k - 1] - y[k + 1]
            #     lap_z = z[k] * 2 - z[k - 1] - z[k + 1]
            #     total_dist = np.sqrt(lap_x * 2 + lap_y * 2 + lap_z ** 2)
                # if abs(c[k] - total_dist) > 0.0:
                #     if R != 0:
                #         strands[k, 0, i, j] += Wave(d[k], R=R)
    with open(name, "w") as f:
        vertex_count = 1
        for i in range(shape[2]):
            for j in range(shape[3]):
                if not np.any(strands[:, :, i, j]):
                    continue
                strand = strands[:, 0:3, i, j]
                for k in range(shape[0]):
                    line = "v {} {} {}\n".format(strand[k, 0], strand[k, 1], strand[k, 2])
                    f.write(line)
                    vertex_count += 1
                for k in range(shape[0] - 1):
                    v1 = vertex_count - shape[0] + k
                    v2 = vertex_count - shape[0] + k + 1
                    line = "l {} {}\n".format(v1, v2)
                    f.write(line)
        print("done writing into " + name)

def demo_from_mask(interp_factor, img_path):
    """ convdata_path = "convdata/" + img_path.split("/")[-1].split("_v1")[0] + ".convdata"
    convdata = np.load(convdata_path).reshape(100, 4, 32, 32) """

    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    strands = strands_from_processed_image(img)

    saveObj("{}_{}.obj".format(img_path, time.time()), strands)

    # show3DhairPlotByStrands(strands)

def demo_from_unprocessed_img(interp_factor, img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = hair_mask_from_unprocessed_image(img)


    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # save mask img with name appended "_mask"
    cv2.imwrite(img_path.split(".")[0] + "_mask.png", img)


    strands = strands_from_processed_image(img)

    saveObj("{}_{}.obj".format(img_path, time.time()), strands)


    # show3DhairPlotByStrands(strands)



def example(convdata):
    strands = np.load(convdata).reshape((100, 4, 32, 32))
    show3DhairPlotByStrands(strands)

if __name__ == "__main__":
    # init_lightning("src/lightning_logs/version_13/checkpoints/epoch=499-step=169000.ckpt")
    init("weight/duc_model_new_weight.pt")
    # demo_from_mask(1,"output.png")
    # demo_from_mask(1, "data/strands00002_00004_10000_v2.png")
    # demo_from_mask(1, "testhair/shorthair.png")
    # demo_from_unprocessed_img(1, "testhair/275967785_1166331687470898_5366165487611011380_n.png")
    # demo_from_unprocessed_img(1, "testhair/toc.jpg")
    # demo_from_unprocessed_img(1, "testhair/hieu.jpg")
    for filename in os.listdir("testhair/myhair/selected"):
        if filename.endswith(".jpg"):
            demo_from_unprocessed_img(1, "testhair/myhair/selected/" + filename)

