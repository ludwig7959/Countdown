from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from hmdb import HMDB51DataModule
from model import CNNLSTM
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
if __name__ == "__main__":
    hmdb = HMDB51DataModule('HMDB51')
    model = CNNLSTM(51)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints/', filename='hmdb51-{epoch:02d}-{val_loss:.2f}')

    trainer = Trainer(max_epochs=10, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=hmdb)
