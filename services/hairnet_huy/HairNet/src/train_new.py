import argparse
import logging
import os
import time

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.utils.data import DataLoader

from dataloader import HairNetDataset
from model_rewrite import HairNetModelOriginal, HairNetLossRewrite, DucModel

log = logging.getLogger("HairNet")
logging.basicConfig(level=logging.INFO)


class HairNetLightning(pl.LightningModule):
    def __init__(self, args = None):
        super().__init__()
        self.args = args
        self.model = HairNetModelOriginal()
        self.loss = HairNetLossRewrite()

        self.epoch_loss_values = []

    def training_step(self, batch, batch_idx):
        img, convdata, visweight = batch
        output = self(img)
        loss = self.loss(output, convdata, visweight)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("train_loss_epoch", epoch_loss)
        self.epoch_loss_values.append(epoch_loss)


    def test_step(self, batch, batch_idx):
        img, convdata, visweight = batch
        output = self(img)
        loss = self.loss(output, convdata, visweight)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_step, gamma=0.5)
        return [optimizer], [scheduler]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)  # 32
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_step", type=int, default=250)
    parser.add_argument("--save_dir", type=str, default="./weight/")
    parser.add_argument("--data", type=str, default="..")
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument("--test_step", type=int, default=10)
    return parser.parse_known_args()


if __name__ == "__main__":
    opt, unknown = get_args()
    epochs, bs, lr, lr_step, save_dir, data, weight, test_step = (
        opt.epoch,
        opt.batch_size,
        opt.lr,
        opt.lr_step,
        opt.save_dir,
        opt.data,
        opt.weight,
        opt.test_step,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training args: {opt}")
    log.info(f"Training device: {device}")

    log.info("Initializing model and loss function ...")

    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1)
    trainer.fit(
        model=HairNetLightning(opt),
        train_dataloaders=DataLoader(HairNetDataset(project_dir=data, train_flag=1, noise_flag=1), batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True),
    )

    if test_step != 0:
        test_data = HairNetDataset(project_dir=data, train_flag=0, noise_flag=0)
        test_loader = DataLoader(dataset=test_data, batch_size=bs)
        log.info(f"Test dataset: {len(test_data)} data points")

    save_path = save_dir + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    os.mkdir(save_path)

    # train
    log.info("Training ...")