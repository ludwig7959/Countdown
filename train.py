from lightning import Trainer
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from hmdb import HMDB51DataModule
from model import CNNLSTM
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
if __name__ == "__main__":
    hmdb = HMDB51DataModule('HMDB51')
    model = CNNLSTM(num_classes=51)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints/', filename='hmdb51-{epoch:02d}-{val_loss:.2f}')
    tb_logger = loggers.TensorBoardLogger(save_dir="logs/")

    trainer = Trainer(max_epochs=30, callbacks=[checkpoint_callback], logger=tb_logger, accelerator="auto", devices="auto")
    trainer.fit(model, datamodule=hmdb)
