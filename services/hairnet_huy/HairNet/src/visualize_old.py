from src.model_rewrite import DucModel, DucModelNew
from src.train_new import get_args

import cv2
import numpy as np
import plotly.express as px
import plotly.io as pio
import torch
from torchvision.transforms import transforms

from src.preprocessing import gauss_noise

pio.renderers.default = "browser"


def show3DhairPlotByStrands(strands: np.ndarray):
    """
    strands: [100, 4, 32, 32]
    """

    strands = strands.transpose((1, 2, 3, 0)).reshape(4, 100 * 32 * 32)

    line_indexing = []
    for i in range(32 * 32):
        for j in range(100):
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
    fig.write_html("test.html", auto_open=True)


if __name__ == "__main__":
    opt, _ = get_args()
    model = DucModelNew().cuda()
    state_dict = torch.load("../weight/duc_model_new_weight.pt")
    # state_dict = torch.load("./lightning_logs/version_9/checkpoints/epoch=99-step=8900.ckpt")["state_dict"]

    # for key in list(state_dict.keys()):
    #     if 'model.' in key:
    #         state_dict[key.replace('model.', '')] = state_dict[key]
    #         del state_dict[key]
    model.load_state_dict(state_dict)
    model.eval()

    img = cv2.imread('../data/strands00009_00026_00000_v0.png')

    img = gauss_noise(img)

    transform = transforms.ToTensor()
    img = transform(img).cuda().view(1, 3, 256, 256)
    img = img / 255 * 2 - 1

    out = model(img)

    strands = out[0].cpu().detach().numpy()  # hair strands

    gaussian = cv2.getGaussianKernel(10, 3)
    for i in range(strands.shape[2]):
        for j in range(strands.shape[3]):
            strands[:, :3, i, j] = cv2.filter2D(strands[:, :3, i, j], -1, gaussian)

    show3DhairPlotByStrands(strands)
