import warnings

from pathlib import Path
from argparse import ArgumentParser
import argparse
import os
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import  ModelCheckpoint
from m_finetuning import FeatureExtractorFreezeUnfreeze
import json

from pl_datamodule import TomatoDataModule
from pl_constrained_model import ConstrainedSegmentMIL

from imgcallback import ImageLogger

import pathlib

import yaml

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", help="""File containing training configuration.""", default = './configs/config.yaml')
    args = parser.parse_args()
    #Leer archivo YAML
    with open(args.config,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
   
    print ('\n Config:\n',config)
    # Comprobar si hay GPU
    cuda_available=torch.cuda.is_available()
    if cuda_available:
        cuda_count=torch.cuda.device_count()    
        cuda_device=torch.cuda.current_device()
        tarjeta_cuda=torch.cuda.device(cuda_device)
        tarjeta_cuda_device_name=torch.cuda.get_device_name(cuda_device)
        print(f'Cuda Disponible. Num Tarjetas: {cuda_count}. Tarjeta activa:{tarjeta_cuda_device_name}\n\n')
        device='cuda'
        gpu=1
    else:
        device='cpu'
        gpu=0


    train_dataplaces=config['train_dataplaces']
    val_dataplaces=config['val_dataplaces']
    
    delimiter= config['delimiter'] # para separar num vista de fruit_id
    terminaciones=config['terminaciones']
   
    maxValues=config['maxValues']
    defect_types= config['defect_types']
    
    images_root_folder = config['root_folder']
    print('Config Root Folder:',images_root_folder)
    user=os.getenv('USER')
    if user=='csanchis':
        images_root_folder=os.path.join("/home/csanchis",images_root_folder)
    elif user=="aalbiol":
        images_root_folder=os.path.join("/home/aalbiol/owc",images_root_folder)

    print('Root Folder:',images_root_folder)
    print('Train Dataplaces=',train_dataplaces)
    print('Val Dataplaces=',val_dataplaces)
    image_size=(config['InputTensorSize'],config['InputTensorSize'])
    datamodule =  TomatoDataModule( training_path = images_root_folder, 
                 set_categories=None, 
                 train_dataplaces=train_dataplaces,
                 val_dataplaces=val_dataplaces,
                 suffixes=terminaciones,
                 defect_types=defect_types,                                                 
                 predict_set_folder = None , 
                 batch_size=config['batch_size'], 
                 imagesize=image_size,
                 normalization_means=None,
                 normalization_stds= None,
                 size_rect=None,
                 maxValues=maxValues,
                 delimiter=delimiter,
                 use_balanced_sampler=True,
                 in_memory=config['in_memory'],
                 num_workers=config['num_workers'],
                 use_masks=config['use_masks'])
    
    num_classes=datamodule.get_num_classes()
    print(num_classes)
    
    pathlib.Path(config['save_path']).mkdir(parents=True, exist_ok=True)    

# Saving info
    fname=os.path.join(config['save_path'],'clases_last_train.json')
    print('Saving '+fname)
    with open(fname, 'w') as outfile:
        json.dump({'clases':list(datamodule.tipos_defecto) },outfile,indent=4)
    


    medias_norm=datamodule.medias_norm.tolist()
    stds_norm=datamodule.stds_norm.tolist()
    size_rect=str(datamodule.size_rect)
    print('medias_norm:',medias_norm)
    print('stds_norm:',stds_norm)
    print('Categorias:',datamodule.tipos_defecto)

    dict_norm={'medias_norm': medias_norm,
    'stds_norm': stds_norm }
    
    fname=os.path.join(config['save_path'],'normalization.json')
    print('Saving '+fname)
  
    with open(fname, 'w') as outfile:
        json.dump(dict_norm,outfile,indent=4)

    print("Saving params in json file...")
    cadena_medias = ";".join([str(valor) for valor in medias_norm])
    cadena_stds = ";".join([str(valor) for valor in stds_norm])
    
    
    #crear una lista con los nombres de los TDAtas TDataNames = ['img_size','planes']
    TDataNames = ['InputTensorSize',"RequiredPlanes","Means","Stds","ClassTypes","SizeRect"]
    datos_json= {"TDataParams":{"Means":cadena_medias,
                                "Stds": cadena_stds,
                                "SizeRect": size_rect,
                                "ClassNames": ";".join(defect_types)},

                 "TrainParams": {}}
    
    fichero_json_parametros=config['model_name'].split('.')[0]+'.json'
    print('Saving json file:',fichero_json_parametros)
    with open(fichero_json_parametros,"w") as json_file:

        for key, value in config.items():
            if key in TDataNames:
                datos_json["TDataParams"][key]= str(value)
                continue
            else:
                datos_json["TrainParams"][key]= str(value)
           
        json.dump(datos_json, json_file,indent=4)



    tipos_defecto=datamodule.tipos_defecto
    print(tipos_defecto)
    
    min_negative_fraction=config['min_neg_fraction']

    print("Min Negative Fraction:",min_negative_fraction)
    
    if 'bottom_constrain_weight' not in config:
        config['bottom_constrain_weight']=0.0
        
        
    print('bottom_constrain_weight:',config['bottom_constrain_weight'])    
    model = ConstrainedSegmentMIL(
                            optimizer = config['optimizer'], lr = config['learning_rate'],
                            num_channels_in=config['num_channels_in'],
                            model_version=config['model_version'],
                            warmup_iter=config['warmup'],
                            class_names=defect_types,
                            p_dropout=config['p_dropout'],
                            weight_decay=config['weights_decay'],
                            num_epochs=config['num_epochs'],
                            area_minima=config['area_minima_defecto'],
                            min_negative_fraction=min_negative_fraction,
                            constrain_weight=config['constrain_weight'],
                            bottom_constrain_weight=config['bottom_constrain_weight'],
                            gamma_param=config['gamma_param'],
                            binmask_weight=config['binmask_weight'],
                            lista_pesos_clase=config['lista_pesos_clase'])
        
    # Continuar entrenamiento a partir de un punto
    if config['initial_model'] is not None:
        checkpoint = torch.load(config['initial_model'])
        model.load_state_dict(checkpoint['state_dict'])


    # Instantiate lightning trainer and train model   

  
    miwandb= WandbLogger(name=config['log_name'], project=config['wandb_project'],config=config)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    model_name=config['model_name']
    save_path = (config['save_path'] if config['save_path'] is not None else '/') + model_name
    checkpoint_saver=ModelCheckpoint(dirpath=config['save_path'],  save_top_k=1,  every_n_epochs=5, monitor='val_loss', 
                                     filename='{epoch}-{val_loss:.2f}',mode='min')
   
    trainer_args = { 'max_epochs': config['num_epochs'], 'logger' : miwandb}
    log_cimg=config['log_cimg']
    if user=='csanchis':
        log_cimg=os.path.join("/home/csanchis",log_cimg)
    elif user=="aalbiol":
        log_cimg=os.path.join("/home/aalbiol/owc",log_cimg)
    
    imagelogger=ImageLogger(epoch_interval=1,num_samples=1,fname=log_cimg,defect_types=tipos_defecto,
                            image_size=image_size,model=model,normalization=dict_norm,num_channels_in=config['num_channels_in'])
       
    print('num_epochs:',config['num_epochs'])
    
    callbacks2=[lr_monitor, checkpoint_saver]
    callbacks3=[lr_monitor, checkpoint_saver,FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=config['unfreeze_epoch'],initial_denom_lr=1)]
    callbacks4=[lr_monitor, 
                FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=config['unfreeze_epoch'],initial_denom_lr=1), 
                imagelogger]
    trainer = pl.Trainer(callbacks=callbacks4  ,**trainer_args) 

  
    
    trainer.fit(model, datamodule=datamodule)

    #trainer.predict_dataloaders() #Pendiente de mirar
    
    # Save trained model
    trainer.save_checkpoint(save_path) 

    #Save ts model
    print('Saving ts model...')
  
    model_name_ts=config['model_name'].split('.')[0]+'.ts'
    

    model.modelo.eval()
    print('Saving model_ts:',model_name_ts)
    #input_torch=torch.randn(1,4,args.image_size,args.image_size)
    model_scripted = torch.jit.script(model.modelo)
    save_path_ts = (config['save_path'] if config['save_path'] is not None else '/') + model_name_ts

    model_scripted.save(save_path_ts)
    
    
