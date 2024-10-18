
import warnings
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torchmetrics import ConfusionMatrix
from torchmetrics import F1Score, AUROC
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR
import torchmetrics

from torch.optim.lr_scheduler import LinearLR

import numpy as np

import modelos
import metricas
import os

torch.set_printoptions(precision=3)


   
def write_names(fichero,nombres):
    with open(fichero, 'w') as fp:
        for item in nombres:
            fp.write("%s\n" % item)
    #print(f' Finished writing {fichero}')

class  SegmentadorMonoclase(pl.LightningModule):
    '''
    Clase para segmentar defectos en RayosX
    El labeling son imágenes binarias con un solo tipo de defecto que se simulan sobre producto sin defecto
    '''
    def __init__(self, num_channels_in=3,
                optimizer='sgd', 
                lr=1e-3,
                weight_decay=1e-3,
                model_version = "deeplabv3_resnet50",
                warmup_iter=0,
                p_dropout=0.5,
                num_epochs=None,
                gamma_param=0.1,
                ):
        super().__init__()



        self.__dict__.update(locals())
     
        self.num_channels_in=num_channels_in

#Learning params
        self.weight_decay=weight_decay
        self.lr=lr
        self.warmup_iter=warmup_iter    
        self.p_dropout=p_dropout        
        self.optimizer_name =optimizer        
        self.num_epochs=num_epochs

        self.gamma_param=gamma_param
        print('SegmentadorMonoclase Num epochs:',self.num_epochs)
           
        if model_version ==  "deeplabv3_resnet50":
            self.modelo = modelos.deeplabv3resnet50(num_classes=1,num_channels_in=self.num_channels_in, p_dropout=self.p_dropout)

       
        else:
            print(f"\n***** Warning. Version  solicitada {model_version} no contemplada. Usando deeplabv3_resnet50")
            self.modelo=modelos.deeplabv3resnet50(num_classes=1,num_channels_in=self.num_channels_in, p_dropout=self.p_dropout)


        self.epoch_counter=1
        
        #self.pos_weights = self.pos_weights # El área de los defectos es muy pequeña comparada con la de los no defectos
        

        

    
    
                

    
    def forward(self, X):# Para training
        logits_pixeles=self.modelo(X)
        
        return logits_pixeles


    def criterion(self, logits_pixeles, bin_masks):#Recibe batch_size x (numclases)
        '''
        Calcula el loss medio de los pixeles
        La única cosa rara es que como hay pocos pixeles con defecto se ponderan más

        '''
        #bcelogitsloss=nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(self.pos_weights).to(self.device))
        bcelogitsloss=nn.BCEWithLogitsLoss(reduction='mean')
        #print(">>>> CRITERIOON",type(logits_pixeles),type(bin_masks))
        # loss0=bcelogitsloss(logits_pixeles[bin_masks>0.5],bin_masks[bin_masks>0.5])
        # loss1=bcelogitsloss(logits_pixeles[bin_masks<=0.5],bin_masks[bin_masks<=0.5])    
        loss0=bcelogitsloss(logits_pixeles,bin_masks)
        loss1=0    
        return loss0,loss1


    def configure_optimizers(self):
        
        parameters = list(filter(lambda x: x.requires_grad, self.parameters()))
        if self.optimizer_name.lower() == 'sgd':
            optimizer = SGD(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == 'adam':
            optimizer = Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        else:
            print(f'**** WARNING : Optimizer configured to {self.optimizer_name}. Falling back to SGD')
            optimizer = SGD(parameters, lr=self.lr, weight_decay=self.weight_decay)
                
        gamma=self.gamma_param**(1/self.num_epochs)
        if self.warmup_iter > 0:           
            warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_iter)
            schedulers = [warmup_lr_scheduler, ExponentialLR(optimizer, gamma=gamma) ]
        else:
           schedulers = [ ExponentialLR(optimizer, gamma=0.99) ] 
        return [optimizer],schedulers


    def training_step(self, batch):
        
        images = batch['images']
        bin_masks=batch['bin_masks']
        
        logits_pixels = self.forward(images)
        
        loss0,loss1 = self.criterion(logits_pixels, bin_masks)
        
        # num_positives=bin_masks.sum()
        # num_negatives=bin_masks.numel()-num_positives
        
        # num_positives*=8
        
        # num_total=num_positives+num_negatives
        # weight_positives=num_positives/num_total
        # weight_negatives=num_negatives/num_total
        
        
        #loss=loss0*weight_negatives+loss1*weight_positives
        loss=(loss0+loss1)/2
        
        log_dict={'train_loss':loss, "train_loss0":loss0, "train_loss1":loss1}

        for k,v in log_dict.items():
            self.log(k,v, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss



    def validation_step(self, batch, batch_idx):
        images = batch['images']
        bin_masks=batch['bin_masks']
        
        logits_pixels = self.forward(images)
        
        
        loss0,loss1 = self.criterion(logits_pixels, bin_masks)
        loss=(loss0+loss1)/2
        
        
        log_dict={'val_loss':loss,"val_loss0":loss0, "val_loss1":loss1}

        for k,v in log_dict.items():
            self.log(k,v, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss




    def predict_step(self, batch, batch_idx, dataloader_idx=0):    
        if 'casos' in batch:
            casos=batch['casos']
        else:
            casos=batch['view_ids']
        labels = batch['labels']
        bin_masks=batch['bin_masks']
        batchn = batch['images'] # Vienen normalizadas del dataloader
        aggregation = self.forward_predict(batchn)
        probs_defecto = aggregation[1]
        probs_pixels=aggregation[2]
        out={'preds':probs_defecto,'labels':labels,'casos':casos, 'probs_pixels':probs_pixels,'bin_masks':bin_masks}
        
        # for m in zip(casos,probs_vista,labels):
        #     zz=np.round(m[1]*100).astype('int')
        #     lista_probs = zz.tolist() 
        #     lista_labels=m[2].tolist()
        #     dict_predictions={ ww[0]:ww[1] for ww in zip(self.class_names,lista_probs)}
        #     dict_annotations={ ww[0]:ww[1] for ww in zip(self.class_names,lista_labels)}
        #     out[m[0]]= {'predictions':dict_predictions,'annotations':dict_annotations}
        return out

    def on_validation_epoch_end(self, ) -> None:        
        self.epoch_counter += 1 
        return    

    def on_training_epoch_end(self) -> None:
        pass
            
