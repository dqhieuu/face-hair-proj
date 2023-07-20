import argparse
import math

import cv2
import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from .MobileNetV2_unet import MobileNetV2_unet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--map_color", type=bool, default=False)
    parser.add_argument("--save_img", type=bool, default=False)
    return parser.parse_args()


color_map = [
    [168, 241, 255],  ##FFF1A8
    [152, 250, 255],  ##FFFA99
    [88, 242, 255],  ##FFF258,
    [76, 233, 255],  ##FFE94C
    [62, 219, 255],  ##FFDB3E,
    [44, 192, 255],  ##FFC02C
    [24, 148, 255],  ##FF9418
    [16, 123, 255],  ##FF7B10,
    [7, 82, 255],  ##FF5207
    [0, 8, 255],  ##FF0800,
]


def map_colors(value):
    value = int(value)
    value = max(0, value)
    value = min(9, value)
    return color_map[value]


"""
getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]]) -> retval
.   @brief Returns Gabor filter coefficients.
.
.   For more details about gabor filter equations and parameters, see: [Gabor
.   Filter](http://en.wikipedia.org/wiki/Gabor_filter).
.
.   @param ksize Size of the filter returned.
.   @param sigma Standard deviation of the gaussian envelope.
.   @param theta Orientation of the normal to the parallel stripes of a Gabor function.
.   @param lambd Wavelength of the sinusoidal factor.
.   @param gamma Spatial aspect ratio.
.   @param psi Phase offset.
.   @param ktype Type of filter coefficients. It can be CV_32F or CV_64F .
"""


def build_filters():
    filters = []
    ksize = 100
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 15.0, theta, 5.0, 2.0, 0, ktype=cv2.CV_32F)
        filters.append(kern)
    return filters


def process(img, filters):
    f_size = len(filters)

    direction_idx = np.zeros_like(img)
    max_gabor = np.zeros_like(img)

    for i in range(f_size):
        fimg = cv2.filter2D(img, cv2.CV_8UC3, filters[i])
        for h in range(len(fimg)):
            for w in range(len(fimg[0])):
                if fimg[h][w] > max_gabor[h][w]:
                    max_gabor[h][w] = fimg[h][w]
                    direction_idx[h][w] = i

    res_dir = np.zeros([2, *img.shape])
    # print(np.min(max_gabor))

    for h in range(len(direction_idx)):
        for w in range(len(direction_idx[0])):
            # if max_gabor[h][w] <= 0.5:
            #     res_dir[0][h][w] = 0
            #     res_dir[1][h][w] = 255
            #     continue
            f = (((direction_idx[h][w] + 0) % f_size) / f_size) * 2 - 1
            res_dir[1][h][w] = ((1 - f*f) ** 0.5) * 255
            res_dir[0][h][w] = (f / 2.0 + 0.5) * 255


    return res_dir


def gabor_filter(image: numpy.ndarray):
    filters = build_filters()

    direction_arr = process(image, filters) # [2, h, w]
    rgb_result = np.dstack((direction_arr[0], direction_arr[1], np.full(direction_arr[0].shape, 255.0))) # [h, w, 3]

    rgb_result = rgb_result.astype(np.uint8)

    ax = plt.subplot(141)
    ax.imshow(rgb_result)

    bw_result = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2GRAY)

    bw_result = cv2.Canny(bw_result, 50, 100, None, 3)

    ax = plt.subplot(142)
    ax.imshow(bw_result, cmap='gray')

    lines = cv2.HoughLines(bw_result, 2, np.pi / 180, 50, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            f = (theta/np.pi *2) - 1  # [-1, 1]
            # print(f)
            c2 = int((1 - f*f) ** 0.5 * 255)
            c1 = int((f / 2.0 + 0.5) * 255)
            cv2.line(rgb_result, pt1, pt2, (c1,c2, 255), 2, cv2.LINE_AA)

    # lines = cv2.HoughLinesP(bw_result, 10, np.pi / 180, 100, None, 4, 5)
    #
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #
    #         rad = math.atan2(y2 - y1, x2 - x1)
    #         f = 2 * rad / np.pi  # [-1, 1]
    #
    #         print('f', f, line)
    #         c2 = int((1 - f*f) ** 0.5 * 255)
    #         c1 = int((f / 2.0 + 0.5) * 255)
    #         cv2.line(rgb_result, (x1,y1), (x2,y2), (c1, c2, 255), 2, cv2.LINE_AA)

    ax = plt.subplot(143)
    ax.imshow(rgb_result)


    return rgb_result


def load_model():
    model = MobileNetV2_unet()
    state_dict = torch.load("hair_detection/model/model.pt")
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()


def input_to_mask_image(img: Image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    torch_img = transform(img)
    torch_img = torch_img.unsqueeze(0)

    logits = model(torch_img)

    temp_logits = logits.squeeze().detach().numpy()  # 3x224x224

    transposed = np.transpose(torch_img.squeeze().numpy(), (1, 2, 0))

    ax = plt.subplot(133)
    ax.axis("off")
    img = gabor_filter(cv2.cvtColor(transposed, cv2.COLOR_RGB2GRAY))

    mask = np.argmax(logits.data.cpu().numpy(), axis=1)
    mask = mask.squeeze()

    mask_n = np.zeros((224, 224, 3))

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] == 1:  # Face
                mask_n[i][j] = [0, 0, 128]
                continue
            if mask[i][j] == 2:  # Hair
                mask_n[i][j] = img[i][j]
                continue
            if mask[i][j] == 0:  # Background
                mask_n[i][j] = [0, 0, 0]
                continue

    # it = 2000
    # for i in range(it):
    #     theta = i * np.pi / it
    #     x = int(90 * np.cos(theta))
    #     y = int(90 * np.sin(theta))
    #     tr = 100
    #
    #     f = i / it * 2 - 1  # [-1, 1]
    #
    #     c2 = int((1 - f * f) ** 0.5 * 255)
    #     c1 = int((f / 2.0 + 0.5) * 255)
    #
    #     cv2.line(mask_n, (x + tr, y + tr), (-x + tr, -y + tr), (c1, c2, 255), 1, cv2.LINE_AA)

    # output = mask_n.astype(np.uint8)
    output = cv2.resize(
        mask_n.astype(np.uint8), (256, 256), interpolation=cv2.INTER_LINEAR
    )

    return output, mask


if __name__ == "__main__":
    # imagePath = get_args().image_path
    imagePath = "./testhair/hieu.jpg"
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)

    output, mask = input_to_mask_image(pil_img)

    if get_args().save_img:
        cv2.imwrite("output.png", output)

    ax = plt.subplot(131)
    ax.axis("off")
    ax.imshow(image.squeeze())

    ax = plt.subplot(132)
    ax.axis("off")
    if get_args().map_color:
        ax.imshow(output)
    else:
        ax.imshow(mask)

    plt.show()
