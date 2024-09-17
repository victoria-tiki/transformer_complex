#!/usr/bin/env python
from __future__ import print_function
import argparse
<<<<<<< HEAD
from time import time
import os
import gc
=======
import pickle
from time import time
import os
import gc

>>>>>>> master
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer,LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
<<<<<<< HEAD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import *
from data_generators import *
import pickle

import pytorch_lightning as pl

torch.backends.cudnn.benchmark = True
#torch.set_float32_matmul_precision('medium')


print(torch.__version__)
class TransformerModel(pl.LightningModule):
    def __init__(self, learning_rate,embed_dim = 128,dense_dim = 64,num_heads = 8):
        super(TransformerModel, self).__init__()
        self.model = create_transformer(embed_dim = embed_dim,dense_dim = dense_dim, num_heads = num_heads, device="cuda" if torch.cuda.is_available() else "cpu")#.to(torch.float64)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            #print(self.model.device)
=======
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler, StepLR
import pytorch_lightning as pl



from models import *
from data_generators import *


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')


print(torch.__version__)



def compute_loss(predictions, targets, mask, weight_factor=4.0):
    loss_fn = nn.MSELoss(reduction='mean')
    loss=loss_fn(predictions,targets)
    return loss


class TransformerModel(pl.LightningModule):
    def __init__(self, learning_rate,embed_dim = 160//2,dense_dim = 80,num_heads = 10):   
        super(TransformerModel, self).__init__()
        self.model = create_transformer(embed_dim = embed_dim,dense_dim = dense_dim, num_heads = num_heads, device="cuda" if torch.cuda.is_available() else "cpu",return_mask=True)#.to(torch.float64)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
>>>>>>> master
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        self.batch_count = 0

    def forward(self, x):
        return self.model(x)
<<<<<<< HEAD

=======
        
    def on_train_epoch_end(self, unused=None):
        optimizer = self.trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']
        print(f'End of epoch {self.current_epoch}: Current Learning Rate: {current_lr}')
        #self.log('learning_rate', current_lr, on_epoch=True, logger=True)

    
>>>>>>> master
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
<<<<<<< HEAD
                'scheduler': ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-10, verbose=True),
=======
                'scheduler': ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-10, verbose=True),
>>>>>>> master
                'monitor': 'val_loss',  
                'interval': 'epoch',  
                'frequency': 1,  
                'strict': True,
            }
        }
<<<<<<< HEAD

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        encoder_inputs, decoder_inputs = inputs
        outputs = self.model(encoder_inputs, decoder_inputs)
        loss = self.loss_fn(outputs, targets)
        
=======
        
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        encoder_inputs, decoder_inputs = inputs
        outputs, imag_mask = self.model(encoder_inputs, decoder_inputs)
        loss=compute_loss(outputs, targets, imag_mask)
        #self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
>>>>>>> master
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        encoder_inputs, decoder_inputs = inputs
        with torch.no_grad():
<<<<<<< HEAD
            outputs = self.model(encoder_inputs, decoder_inputs)
            loss = self.loss_fn(outputs, targets)
=======
            outputs, imag_mask = self.model(encoder_inputs, decoder_inputs)
            loss=compute_loss(outputs, targets, imag_mask)
>>>>>>> master
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gw forecasting")
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
<<<<<<< HEAD
    parser.add_argument('--checkpoint_dir', help='root directory', default='/scratch/bbke/victoria/Transformer_training/checkpoint/')
    parser.add_argument('--data_dir', help='root directory', default='/scratch/bbke/victoria/Transformer_data/') 
    parser.add_argument('--num_workers', type=int, help='number of workers in dataloader', default=1)
    parser.add_argument('--num_nodes', type=int, help='number of nodes', default=1) 
=======
    parser.add_argument('--checkpoint_dir', help='root directory', default='/scratch/bdao/victoria/Transformer_training/checkpoint/')
    parser.add_argument('--data_dir', help='root directory', default='/scratch/bdao/victoria/Transformer_data/') 
    parser.add_argument('--num_workers', type=int, help='number of workers in dataloader', default=1)
    parser.add_argument('--num_nodes', type=int, help='number of nodes', default=1) 
    parser.add_argument('--resume', type=int, choices=[0, 1], default=1, help='1 to resume training from the latest checkpoint, 0 not to')
>>>>>>> master
    args = parser.parse_args()
    
    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    num_workers=args.num_workers
    num_nodes=args.num_nodes
    
<<<<<<< HEAD
    
    # Define Call Backs
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir+'/weights/', filename='model_{epoch:02d}-{val_loss:.5f}', monitor='val_loss', save_top_k=-1)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    callbacks = [checkpoint_callback, early_stopping_callback]


    # define trainer
    devices=torch.cuda.device_count() if torch.cuda.is_available() else 2
=======
    if args.resume:
        checkpoint_dir_resume = os.path.join(checkpoint_dir, 'weights','model_separate_epoch=08-val_loss=0.00000.ckpt')
        print('loading from',checkpoint_dir_resume)
        
    # Define Call Backs
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir+'/weights/', filename='model_separate_{epoch:02d}-{val_loss:.5f}', monitor='val_loss', save_top_k=-1)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    progress_bar=pl.callbacks.TQDMProgressBar(refresh_rate=100)
    callbacks = [checkpoint_callback, early_stopping_callback,progress_bar]


    # define trainer
    devices=torch.cuda.device_count() if torch.cuda.is_available() else 1
>>>>>>> master
    print('devices',devices)

    trainer = Trainer(
        max_epochs=100,
<<<<<<< HEAD
        num_nodes=num_nodes,devices=devices,accelerator="gpu", strategy="ddp",     
        precision="16-mixed",
        enable_progress_bar=True,
=======
        num_nodes=num_nodes,devices=devices,accelerator="gpu", strategy='ddp_find_unused_parameters_true',     
        precision="16-mixed",
>>>>>>> master
        enable_model_summary=True, 
        callbacks=callbacks,
        )
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Lightning model
    initial_learning_rate = 0.001
<<<<<<< HEAD
    model = TransformerModel(initial_learning_rate,embed_dim = 128,dense_dim = 64,num_heads = 8)
    
    #Define data
    data_module = WaveformDataModule(data_dir=data_dir, batch_size=batch_size, normalize=True, num_workers=num_workers)


    # Train 
=======
    model = TransformerModel(initial_learning_rate,embed_dim = 160//2,dense_dim = 80,num_heads = 10)
    
    if args.resume:
        print('starting from checkpoint')  
        checkpoint = torch.load(checkpoint_dir_resume)
        optimizer_state_dict = checkpoint['optimizer_states'][0]
        learning_rate = optimizer_state_dict['param_groups'][0]['lr']

        new_learning_rate = learning_rate
        model = TransformerModel.load_from_checkpoint(checkpoint_dir_resume,learning_rate=new_learning_rate)
    else: 
        print('starting from scratch')

    # Load Data Generators
    print('define data module')
    data_module = WaveformDataModule(data_dir=data_dir, batch_size=batch_size, normalize=False, num_workers=num_workers)


    # Train using PyTorch Lightning Trainer
>>>>>>> master
    t0 = time()
    trainer.fit(model, datamodule=data_module)
    t1 = time()

    print('**Evaluation time: %s' % (t1-t0))
