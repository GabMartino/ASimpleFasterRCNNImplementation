import glob
import os

import hydra
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from model.FasterRCNN import FasterRCNN
from datamodules.GlobalWheatDetectionDataModule import GlobalWheatDetectionDataModule
import pytorch_lightning as pl


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    datamodule = GlobalWheatDetectionDataModule(csv_data_path=cfg.train_csv_path,
                                                image_root_dir=cfg.image_path,
                                                batch_size=cfg.batch_size)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        dirpath=cfg.checkpoint_path,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_last=True,
        mode="min"
    )


    model = FasterRCNN()

    if cfg.restore_from_checkpoint:
        list_of_files = glob.glob(cfg.checkpoint_path+"*")  # * means all if need specific format then *.csv
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
        model = model.load_from_checkpoint(latest_checkpoint)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.logdir)
    trainer = pl.Trainer(max_epochs=cfg.epochs,
                         accelerator="gpu",
                         devices=1,
                         logger=tb_logger,
                         callbacks=[EarlyStopping("val_loss_epoch", patience=cfg.early_stopping_patience), checkpoint_callback])

    trainer.fit(model, datamodule)


if __name__ == "__main__":

    main()