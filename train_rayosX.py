
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import  ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from m_finetuning import FeatureExtractorFreezeUnfreeze
import os
from argparse import ArgumentParser


import torch
torch.set_float32_matmul_precision('medium')

import yaml
import sys

import json



from pl_datamodule import ListFileDataModule


from pl_model import SegmentadorMonoclase
import random
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import pickle




    
def main(config):
    
    tam_erode_training=1
    
    cov_mode="full"
    
    train_dataplaces=config['dataset_params']['train_dataplaces']
    val_dataplaces=config['dataset_params']['val_dataplaces']
    images_root_folder=config['dataset_params']['images_root_folder']
    normalization_json=config['dataset_params']['normalization_json']
    
    image_size=config['model_params']['image_size']
    
    batch_size=config['training_params']['batch_size']
    num_epochs=config['training_params']['num_epochs']    


    random_seed=config['training_params']['random_seed']
    maxvalues=[2.0**16, 2.0**16, 2.0**16]

        
   
    print('>>> Train dataplaces', train_dataplaces)
    print('>>> Configured Images root folder', images_root_folder)
 
    imagesize= (image_size,image_size)
    in_memory= config['training_params']['in_memory']

    if normalization_json is not None:
        with open(normalization_json) as json_file:
            normalization_params=json.load(json_file)
    else:
        normalization_params=None
    
    user=os.getenv('USER')
    if user=='csanchis':
        images_root_folder=os.path.join("/home/csanchis",images_root_folder)
    elif user=="aalbiol":
        images_root_folder=os.path.join("/home/aalbiol/owc",images_root_folder)
    
    print(">>>>> Full Path Images Root Folder: ",images_root_folder, " user: ",user)
    
    defect_simulation_params=config['defect_simulation_params']
    
    print(">>>>>> Defect simulation params: ",defect_simulation_params)
    datamodule=ListFileDataModule(train_dataplaces=train_dataplaces, val_dataplaces=val_dataplaces, 
                                images_root_path=images_root_folder, 
                                in_memory=in_memory, 
                                imagesize=imagesize,
                                normalization_params=normalization_params,
                                batch_size=batch_size,
                                delimiter=".",
                                terminaciones=[".png"],
                                max_values=maxvalues,
                                params_simulacion_defectos=defect_simulation_params,
                                     )
    
    normalizacion=datamodule.getNormalizationParams()
    print('nornmalizacion: ',normalizacion)
 

    model = SegmentadorMonoclase(
                            optimizer = config['training_params']['optimizer'], 
                            lr = config['training_params']['learning_rate'],
                            num_channels_in=3,
                            model_version=config['model_params']['model_version'],
                            warmup_iter=config['training_params']['warmup'],
                            
                            p_dropout=config['model_params']['p_dropout'],
                            weight_decay=config['training_params']['weights_decay'],
                            num_epochs=num_epochs,

                            gamma_param=config['training_params']['gamma_param'],
                            pos_weights=config['training_params']['pos_weights'],)
    
# Continuar entrenamiento a partir de un punto
    if'initial_model' in config and config['initial_model'] is not None:
        checkpoint = torch.load(config['initial_model'])
        model.load_state_dict(checkpoint['state_dict'])


    # Instantiate lightning trainer and train model   

    if config['log_name'] is not None:
        miwandb= WandbLogger(name=config['log_name'], project=config['wandb_project'],config=config)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    model_name=config['training_params']['out_model_name']
    save_path = os.path.join(config['training_params']['save_path'] if config['training_params']['save_path'] is not None else '' ,model_name)
    # checkpoint_saver=ModelCheckpoint(dirpath=config['save_path'],  save_top_k=1,  every_n_epochs=5, monitor='val_loss', 
    #                                  filename='{epoch}-{val_loss:.2f}',mode='min')
   
    trainer_args = { 'max_epochs': num_epochs}
    if config['log_name'] is not None:
        trainer_args['logger']=miwandb
        
    # log_cimg=config['log_cimg']
    # if user=='csanchis':
    #     log_cimg=os.path.join("/home/csanchis",log_cimg)
    # elif user=="aalbiol":
    #     log_cimg=os.path.join("/home/aalbiol/owc",log_cimg)
    
    # imagelogger=ImageLogger(epoch_interval=1,num_samples=1,fname=log_cimg,defect_types=tipos_defecto,
    #                         image_size=image_size,model=model,normalization=dict_norm,num_channels_in=config['num_channels_in'])
       

    
    # callbacks2=[lr_monitor, checkpoint_saver]
    # callbacks3=[lr_monitor, checkpoint_saver,FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=config['unfreeze_epoch'],initial_denom_lr=1)]
    # callbacks4=[lr_monitor, 
    #             FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=config['training_params']['unfreeze_epoch'],initial_denom_lr=1), 
    #             imagelogger]
    
    callbacks5=[lr_monitor, 
                FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=config['training_params']['unfreeze_epoch'],initial_denom_lr=1)]
    trainer = pl.Trainer(callbacks=callbacks5  ,**trainer_args) 

  
    trainer.fit(model, datamodule=datamodule)

    # Save trained model
    trainer.save_checkpoint(save_path) 

    #Save ts model
    print('Saved  model...',save_path)
  

    
    

    
        

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", help="""File containing training configuration.""", default = './configs/config.yaml')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)

    print(">>>>>>")
    print(config)
    print(">>>>>>")
    main(config)        

