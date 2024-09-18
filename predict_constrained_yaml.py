import warnings

from pathlib import Path
from argparse import ArgumentParser
import argparse
import os
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import  ModelCheckpoint
from m_finetuning import FeatureExtractorFreezeUnfreeze
import json

from pl_datamodule import TomatoDataModule
from pl_model import ConstrainedSegmentMIL, write_names



from imgcallback import ImageLogger

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


    
    with open(config['normalizacion']) as json_file:
        normalizacion = json.load(json_file)    
  
    
    delimiter= config['delimiter'] # para separar num vista de fruit_id
    terminaciones=config['terminaciones']
   
    defect_types= config['defect_types']

    images_root_folder = config['root_folder']
    print('Config Root Folder=',images_root_folder)
    user=os.getenv('USER')
    if user=='csanchis':
        images_root_folder=os.path.join("/home/csanchis",images_root_folder)
    elif user=="aalbiol":
        images_root_folder=os.path.join("/home/aalbiol/owc",images_root_folder)
    print('Used Root Folder=',images_root_folder)      
      
    print('Pred Dataplaces=',config['pred_dataplaces'])
    
    image_size=(config['InputTensorSize'],config['InputTensorSize'])
    datamodule =  TomatoDataModule( training_path = images_root_folder, 
                 set_categories=None, 
                 train_dataplaces=None,
                 val_dataplaces=None,
                 pred_dataplaces=config['pred_dataplaces'],
                 suffixes=terminaciones,
                 defect_types=defect_types,                                                 
                 predict_set_folder = None , 
                 batch_size=config['batch_size'], 
                 imagesize=image_size,
                 normalization_means=normalizacion['medias_norm'],
                 normalization_stds= normalizacion['stds_norm'],
                 max_value=255,
                 delimiter=delimiter,
                 use_balanced_sampler=True,
                 in_memory=config['in_memory'],
                 num_workers=config['num_workers'])
    
    num_classes=datamodule.get_num_classes()
    print(num_classes)

    model = ConstrainedSegmentMIL(
                            num_channels_in=config['num_channels_in'],
                            model_version=config['model_version'],
                            class_names=defect_types,
                            p_dropout=config['p_dropout'],
                            )
    # Continuar entrenamiento a partir de un punto
    if config['initial_model'] is not None:
        checkpoint = torch.load(config['initial_model'])
        model.load_state_dict(checkpoint['state_dict'])


    # Instantiate lightning trainer and train model   

    trainer = pl.Trainer(callbacks=None) 
    
    out=trainer.predict(model, datamodule=datamodule)
    # print out type and shape
    print('Type out:',type(out))
    print('Len out:',len(out))
    
    allpreds=[]
    alllabels=[]
    allcasos=[]
    allprobs_pixels=[]
    fname = config['filenames']
    for b in out: # Los distintos batches
        preds=b['preds']
        labels=b['labels']
        casos=b['casos']
        probs_pixels=b['probs_pixels']   
        allpreds.append(preds)
        alllabels.append(labels)
        allcasos.append(casos)
        allprobs_pixels.append(probs_pixels)
        #write_names(fname,casos)
        # with open(fname,'w') as archivo:
        #     lista_casos='\n'.join(map(str,allcasos))
        #     archivo.write(lista_casos)
        # print('casos', casos)        
        
    #print('allcasos',allcasos)
    allpreds=torch.cat(allpreds)
    alllabels=torch.cat(alllabels)
    allprobs_pixels=torch.cat(allprobs_pixels)
    casosall=[]
    for caso in allcasos:
        casosall += caso
    with open(fname,'w') as archivo:
            lista_casos='\n'.join(map(str,casosall))
            archivo.write(lista_casos)
    torch.save(allpreds,config['preds'])
    torch.save(alllabels,config['labels'])
    torch.save(allprobs_pixels,config['pixel_probs'])

    

    
    

    
    
