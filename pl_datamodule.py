import warnings
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import pytorch_lightning as pl

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pycimg
import multiprocessing
from typing import Tuple,Any

import os
from PIL import Image

import cv2
import pandas as pd
import torch.nn.functional as F

import m_dataLoad_json



from transformaciones import Aumentador_Imagenes_y_Mascaras

from dataset import DataplacesDataSetSimulaDefectos

def pad_image(image_tensor, max_height, max_width,pad_value=1):
    _, h, w = image_tensor.shape
    pad_height = max_height - h
    pad_width = max_width - w
    
    # Padding is done in (left, right, top, bottom) order
    padded_image = F.pad(image_tensor, (0, pad_width, 0, pad_height), mode='constant', value=pad_value)  # zero padding
    return padded_image

def centercrop_image(image_tensor, min_height, min_width):
    _, h, w = image_tensor.shape
    pad_height = h - min_height
    pad_width = w - min_width
    x0=pad_width//2
    y0=pad_height//2
    
    # Padding is done in (left, right, top, bottom) order
    cropped=  image_tensor[:,y0:y0+min_height,x0:x0+min_width]
    return cropped


def my_collate_fn(data): # Crear batch a partir de lista de casos
    '''
    images: tensor de batch_size x num_channels_in x height x width
    casos
    mascaras_defectos
    '''
    images = [d[0] for d in data]
    image_sizes = [tensor.shape[-2:] for tensor in images]  # H, W dimensions
    max_height = max([size[0] for size in image_sizes])
    max_width = max([size[1] for size in image_sizes])
    
    min_height = min([size[0] for size in image_sizes])
    min_width = min([size[1] for size in image_sizes])
    #padded_images = [pad_image(image, max_height, max_width,1) for image in images]
    padded_images=[centercrop_image(image, min_height, min_width) for image in images]

# Stack them into a single batch
    images = torch.stack(padded_images,dim=0)
    

    casos = [d[1] for d in data]
 
    bin_masks = [d[2].unsqueeze(0) for d in data]
    
    # padded_masks = [pad_image(image, max_height, max_width,0) for image in bin_masks]
    padded_masks = [centercrop_image(image, min_height, min_width) for image in bin_masks]
    bin_masks = torch.stack(padded_masks, dim = 0) # tendra dimensiones numvistastotalbatch, 1,imgsize,imgsize
    return { 
        'images': images, 
        'casos': casos,
        'bin_masks': bin_masks #del batch
    }
 
       




class ListFileDataModule(pl.LightningDataModule):
    def __init__(self, images_root_path = None, 
                 train_dataplaces=None,
                 val_dataplaces=None,
                 pred_dataplaces=None,                 
                 batch_size: int =25, 
                 num_workers=-1,
                 logaritmo=False,
                 normalization_params=None,
                 normalization_image_params=None,
                 in_memory=True,
                 terminaciones=[".png"],
                 max_values=None,
                 delimiter='.',
                 params_simulacion_defectos=None,
                 scale=1.0,
                 monocanal=False
                 ):
        '''

        '''
        super().__init__()


        
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers > 0 else multiprocessing.cpu_count()-1
        self.terminaciones=terminaciones
        self.max_values=max_values
        self.delimiter=delimiter
        self.images_root_path=images_root_path
        self.normalization_params = normalization_params
        self.normalization_image_params = normalization_image_params
        #self.target_image_size =imagesize
        self.train_dataplaces=train_dataplaces  
        self.val_dataplaces=val_dataplaces
        self.pred_dataplaces=pred_dataplaces
        self.params_simulacion_defectos=params_simulacion_defectos
        self.scale=scale
        self.monocanal=monocanal
        self.logaritmo=logaritmo
        
        assert self.params_simulacion_defectos is not None
        print('self.normalization params at input of ListFileDataModule', self.normalization_params)
        

        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        if train_dataplaces is not None:
            
            self.train_dataset = DataplacesDataSetSimulaDefectos(root_folder=self.images_root_path, dataplaces=self.train_dataplaces, transform = None, 
                                                   in_memory=in_memory, 
                                                   terminaciones=self.terminaciones,
                                                   max_values=self.max_values,
                                                   normalization_params=self.normalization_params,
                                                   normalization_image_params=self.normalization_image_params,
                                                   #normalization_image_size=self.target_image_size, 
                                                   delimiter=self.delimiter,
                                                   params_simulacion_defectos=params_simulacion_defectos,scale=self.scale,
                                                   monocanal=self.monocanal,
                                                   logaritmo=self.logaritmo)
        # Si por configuracion es none, lo calcula con el trainset
        if self.normalization_params is None :
            self.normalization_params=self.train_dataset.normalization_params
            print('self.normalization params computed at ListFileDataModule:', type(self.normalization_params), self.normalization_params)
        
        
        self.medias_norm=self.normalization_params['medias_norm']
        self.stds_norm=self.normalization_params['stds_norm']
        
        if val_dataplaces is not None:
            self.val_dataset = DataplacesDataSetSimulaDefectos(root_folder=self.images_root_path, dataplaces=self.val_dataplaces, transform = None, 
                                                 in_memory=in_memory, terminaciones=self.terminaciones,max_values=self.max_values,
                                                 normalization_params=self.normalization_params, 
                                                 #normalization_image_size=self.target_image_size
                                                 delimiter=self.delimiter,
                                                 params_simulacion_defectos=params_simulacion_defectos,scale=self.scale,
                                                 monocanal=self.monocanal,
                                                 logaritmo=self.logaritmo)
        if pred_dataplaces is not None:
            params_simulacion_defectos_pred=params_simulacion_defectos.copy()
            params_simulacion_defectos_pred['prob_no_change']=1.0
            self.pred_dataset = DataplacesDataSetSimulaDefectos(root_folder=self.images_root_path, dataplaces=self.pred_dataplaces, transform = None, in_memory=in_memory, terminaciones=self.terminaciones,max_values=self.max_values,
                                                 normalization_params=self.normalization_params, 
                                                 #normalization_image_size=self.target_image_size,
                                                 delimiter=self.delimiter,
                                                 params_simulacion_defectos=params_simulacion_defectos_pred,scale=self.scale,
                                                 monocanal=self.monocanal,
                                                 logaritmo=self.logaritmo)
# Transformaciones para el entrenamiento
        transform_geometry_train= transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(15,fill=1),
            # transforms.Resize(self.target_image_size)
            ])
        
        
        transform_normalize=None # transforms.Compose([transforms.Normalize(self.medias_norm, self.stds_norm),])

        transform_train=Aumentador_Imagenes_y_Mascaras(transform_geometry_train,
                                                       None,transform_normalize)

        transform_val = Aumentador_Imagenes_y_Mascaras(None,
                                                       None,transform_normalize) 

        if self.train_dataset is not None:
            self.train_dataset.setTransform(transform_train)
        if self.val_dataset is not None:
            self.val_dataset.setTransform(transform_val)
        if self.pred_dataset is not None:
            self.pred_dataset.setTransform(transform_val)            

        if self.train_dataset is not None:
            print(f"len total trainset =   {self.train_dataset.__len__() }")

        if self.val_dataset is not None:
            print(f"len total valset =   {self.val_dataset.__len__()  }")

        if self.pred_dataset is not None:
            print(f"len total predset =   {self.pred_dataset.__len__()  }")            

        print("batch_size in ListFileDataModule", self.batch_size)
        
    def get_len_trainset(self):
        return len(self.train_dataset)
    def get_len_valset(self):
        return len(self.val_dataset)
    
    def getNormalizationParams(self):
        return self.normalization_params
    


    def get_viewids_val(self):
        return self.view_ids_val


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        return None

    def train_dataloader(self):
        print("batch_size in Dataloader train", self.batch_size)
        return DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True,collate_fn=my_collate_fn)
    
    def val_dataloader(self):
        print("batch_size in Dataloader val", self.batch_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False,shuffle=False, collate_fn=my_collate_fn)
    def predict_dataloader(self):
        print("batch_size in Dataloader pred", self.batch_size)
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False,shuffle=False, collate_fn=my_collate_fn)

        
