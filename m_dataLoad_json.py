import os
import sys
import json
import glob
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from PIL import Image


def parse_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        #print('Data JSON',data)
    return data

def extract_tipos_defecto(d):
    anot =d['annotations']
    tipos_defecto=list(anot.keys())
    return tipos_defecto

def extract_one_hot(d,tipos_defecto):
    anot =d['annotations']
    v=[]
    for defecto in tipos_defecto:
        if defecto in anot:
            tmp=anot[defecto]
            if isinstance(tmp,str):
                tmp=float(tmp)
            if tmp <0 :
                tmp=math.nan
                print('********label is <0**********',tmp)
            v.append(tmp)
        else:
            v.append(math.nan)
    return torch.tensor(v)

def extract_masks(d, tipos_defecto):
    ''' Extrae del diccionario del json las posibles máscaras de segmentación
        Si el json no tiene mascaras devolvemos None
    '''
    if 'masks' not in d:
        return None
    return d['masks']
 

def split_train_val(lista,p):
    n1=int(len(lista)*p)
    train=lista[:n1]
    val=lista[n1:]
    return train,val

def lee_pngChannel(filename,max_value):
    #print(f'Reading {filename}...')
    im=cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    
    if im is not None:
        if im.ndim ==3 :
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im=im.astype('float32')/max_value
        im[im>1]=1

        im=torch.tensor(im)
        if im.ndim ==2:# convertir de hxw --> 1 x h x w
            im=im.unsqueeze(0)
        else:
            im=im.permute((2,0,1))
    return im






def lee_vista(images_folder,view_id,terminaciones,maxValues,crop_size):
    nombre_base=os.path.join(images_folder,view_id) 
    canales=[]
    indice =0
    for t in terminaciones:
        nombre=nombre_base+t
        canal=lee_pngChannel(nombre,maxValues[indice]) # Devuelve un tensor de la imagen normalizada
        canales.append(canal)
        indice += 1
    canales=torch.concat(canales,0) # Se concatena la lista de imagenes en un tensor quedando -> (n_canales,H,W)
    if crop_size is not None:
        h=canales.shape[1]
        w=canales.shape[2]
        fila_ini= int((h - crop_size[0])//2)
        fila_fin=-fila_ini
        col_ini=int(w-crop_size[1])//2
        col_fin=-col_ini
        canales=canales[:,fila_ini:fila_fin,col_ini:col_fin]     
    return canales

def lee_im_mask_segmentacion(images_folder, view_id, terminacion,max_value,crop_size):
    nombre_base=os.path.join(images_folder,view_id)
    canales=[]
    nombre=nombre_base+terminacion
    if not os.path.exists(nombre):
        return None
    
    anotada=lee_pngChannel(nombre,max_value)
            
    if crop_size is not None:
        h=anotada.shape[1]
        w=anotada.shape[2]
        fila_ini= int((h - crop_size[0])//2)
        fila_fin=-fila_ini
        col_ini=int(w-crop_size[1])//2
        col_fin=-col_ini
        anotada=anotada[:,fila_ini:fila_fin,col_ini:col_fin]    
    #print('ANOTADA SHAPE', anotada.shape)  
    return anotada

def mask_bin_segmentacion(masks_RGB_values,im_anotada,defectos,max_value,v_id,tolerancia=0.01):
    '''masks_RGB_values: diccionario cuyos keys son los nombres de los defectos y cuyos values son el RGB con que se ha anotado ese defecto
       Ejemplo: masks_RGB_values = {"Hoja": (0, 0, 1)}
       Los valores de RGB están normalizados
       Devuelve: 
       Una lista de imagenes binarias, una por cada tipo de defecto. Si defectos = ['Ped', 'Contraped', 'Hoja'] devolveria
       [None, None, mascara_binaria_hoja]
       tolerancia: similitud entre el color en la imagen y el color especidicado en la etiqueta
       '''
    if masks_RGB_values is None:
        bin_masks=[None]*len(defectos)
        return bin_masks
    bin_masks=[]
    for d in range(len(defectos)):
        if defectos[d] not in masks_RGB_values:
            bin_masks.append(None)
        else:
            if len(masks_RGB_values[defectos[d]]) < 3:
                bin_masks.append(None)
                continue
            color=torch.tensor(masks_RGB_values[defectos[d]])/max_value
            color=torch.unsqueeze(color,1)
            color=torch.unsqueeze(color,2)
            m=(im_anotada-color).abs().sum(0)<tolerancia # Tolerancia
            #print('BIN MASK ONES',m.float().sum())
            if m.float().sum()==0:
                print(f'Warning: {v_id} Mascara con todos los valores a 0')
            bin_masks.append(m.float())
    return bin_masks

def lee_mascaras(imags_folder,v_id,terminacion_im_mascara,max_value,crop_size,json_dict,tipos_defecto,use_masks):
    '''Devuelve una lista con tantos elemntos como tipos de defecto
        Si una mascara no existe pone None
    '''
    bin_masks=[None]*len(tipos_defecto)
    if use_masks:
        im_mask=lee_im_mask_segmentacion(imags_folder,v_id,terminacion=terminacion_im_mascara,max_value=max_value,crop_size=crop_size)
        if im_mask is not None:
            d=json_dict
            masks = extract_masks(d, tipos_defecto)
            bin_masks=mask_bin_segmentacion(masks,im_mask,tipos_defecto,max_value,v_id)
   
    return bin_masks

def fruit_id(filename,delimiter):
    '''
    Dado un nombre de fichero, 
    obtiene el basename
    elimina el _xx.json

    '''
    basename=os.path.basename(filename)
    id=basename.split(delimiter)

    id=id[:-1]
    id=delimiter.join(id)

    return id

def view_id(filename):
    '''
    Dado un nombre de fichero, 
    obtiene el basename
    elimina el _xx.json

    '''
    basename=os.path.basename(filename)
    id=basename.split('.')
    return ''.join(id[:-1])

def add_good_category(onehot):
     v= (1 if onehot.sum()==0 else 0)
     return torch.concat((torch.tensor(v).unsqueeze(0),onehot))

def GetMaxValues( images_folder, json_file, sufijos):
    v_id=view_id(json_file)
    nombre_base=os.path.join(images_folder,v_id) 
    maxValues =[]
    for sufijo in sufijos:
        #if RGB, AuxB,...
        if sufijo == '_RGB.png':
            maxValues.append(255)
        else:
            
            nombre=nombre_base+sufijo
            print('nombre: ', nombre)
            im = Image.open(nombre)
            if im is not None:
                bitCount = -1
                if "bitCount" in im.info:
                    bitCount=im.info['bitCount']
                    maxValue = 2**int(bitCount) -1
                else:
                    maxValue = 1023
            else:
                maxValue=1023
            maxValues.append(maxValue)

    return maxValues



def  genera_ds_jsons_multilabel(root,  dataplaces, sufijos=None,maxValues=None, size_rect=None, crop_size=(120,120),defect_types=None,splitname_delimiter='-',
                               multilabel=True, in_memory=True, use_masks=False):
    assert sufijos is not None

    assert sufijos is not None
    json_files=[]
    imags_directorio=[]
    for place in dataplaces:
        anotaciones = place[0]
        imagenes=place[1]
        anot_folder=os.path.join(root,anotaciones)
        imags_folder=os.path.join(root,imagenes)
        fichs=glob.glob('*.json',root_dir=anot_folder)
        ficheros=[os.path.join(anot_folder,f) for f in fichs]
        
        json_files +=ficheros#FullPath
        imagenes= [imags_folder]*len(fichs)
        imags_directorio += imagenes


    if defect_types is None:
        tipos_defecto=set()
        for json_file in json_files:
            d=parse_json(json_file)
            defects=extract_tipos_defecto(d)
            defects=set(defects)   
            tipos_defecto = tipos_defecto.union(defects)
       
        tipos_defecto=list(tipos_defecto)
        tipos_defecto.sort()
        print('Tipos Defecto de JSONS:', tipos_defecto)
    else:
        tipos_defecto=defect_types
        print('Tipos de Defecto por configuracion:', tipos_defecto)
            
    #df=pd.DataFrame()
    #df['json_files']=json_files
    #df['fruit_ids']=[fruit_id(a,splitname_delimiter) for a in df['json_files']]
    #df['view_ids']=[view_id(a) for a in df['json_files']]
    #df['imags_folder']=imags_directorio


    #frutos =pd.Series(list(set(df['fruit_ids'])))
    #train=frutos.sample(frac=0.7)
    #val=frutos.drop(train.index)
    #dftrain=df[df['fruit_ids'].isin( list(train) )]
    #dfval=df[df['fruit_ids'].isin( list(val) )]


    out=[]
   
    max_value_mask=255
    if maxValues is None:
        maxValues = GetMaxValues(imags_directorio[0], json_files[0], sufijos) #valores para la normalizacion, se selecciona la primera imagen del dataset
    else:
        maxValues=maxValues
    print("maxValues: ", maxValues)
    #exit(0)
    for fruto in zip(json_files,imags_directorio):
        json_file=fruto[0]
        #print('json_file',json_file)
        f_id=fruit_id(json_file,splitname_delimiter)
        #print('f_id',f_id)
        d=parse_json(json_file)
        #print('d',d)
        #print('type d',type(d))
        v_id=view_id(json_file)
        #print('v_id',v_id)
        imags_folder= fruto[1]
        #print('imags_folder',imags_folder)
        
        if in_memory:
            #channels=lee_vistaRGB_NIR(imags_folder,v_id,crop_size=crop_size)
            channels=lee_vista(imags_folder,v_id,sufijos,maxValues,crop_size=crop_size)
            bin_masks=lee_mascaras(imags_folder,v_id,'_RGB_mask.png',max_value_mask,crop_size,d,tipos_defecto,use_masks)
        else:
            channels=None
            bin_masks=None
        onehot=extract_one_hot(d,tipos_defecto)
        if multilabel==False and onehot.sum()> 1: #skip instances with multiple labels if multiclass
            continue
        if multilabel==False:
            onehot = add_good_category(onehot)
        
        dict_vista={'fruit_id':f_id, 'view_id':v_id, 'image': channels, 'labels': onehot, 
                    'imag_folder': imags_folder, 'sufijos': sufijos, 'maxValues':maxValues, 'crop_size':crop_size, 
                    'bin_masks':bin_masks, 'dict_json':d, 'tipos_defecto': tipos_defecto
                    } 
        
        if bin_masks is None:
            #print('Posiblemente not in memory')
            pass
        else:
            for m in bin_masks:
                if m is None:
                    pass
                    #print('Mascara no disponible para el defecto')
                else:
                    print('BIN MASK for fruit_id',v_id)
            
            #im_name='./enmascarado/mascaras/'+v_id+'.png'
            #cv2.imwrite(im_name,np.array(bin_masks[0]*255))
        out.append(dict_vista)

    
    
    if multilabel ==False:
        tipos_defecto.insert(0,'bueno')        
    return out,tipos_defecto
        


# =================== Salvar listas en jsons ============            
    
def write_list(a_list,filename):
    print("Started writing list data into a json file")
    with open(filename, "w") as fp:
        json.dump(a_list, fp)
        print(f"Done writing JSON data into {filename}")

# Read list to memory
def read_list(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = json.load(fp)
        return n_list
# ==================== Obtener valor del tamaño del recorte =========
def GetImgRectSize(caso):
    image=caso['image']
    view_id=caso['view_id']
    if image is None: #Cuando no está en memoria
        imags_folder=caso['imag_folder']
        sufijos=caso['sufijos']
        maxValues=caso['maxValues']
        crop_size=caso['crop_size']
        #print("Reading ", view_id)
        image=lee_vista(imags_folder,view_id,sufijos,maxValues,crop_size=crop_size)
    
    print("****VIew id***", view_id)
    print("****Image shape***",image.shape)
    return image.shape[1] # Height, en teoría son imgs cuadradas de tamaño del recorte fijo.    

# ===================================== NORMALIZACION ==================
# 
def calcula_media_y_stds(trainset):
    medias=[]
    medias2=[]
    pix_dimensions=(1,2)
    area_total=0
    
    for caso in trainset:
        image=caso['image']
        view_id=caso['view_id']
        if image is None: #Cuando no está en memoria
            imags_folder=caso['imag_folder']
            sufijos=caso['sufijos']
            maxValues=caso['maxValues']
            crop_size=caso['crop_size']
            #print("Reading ", view_id)
            image=lee_vista(imags_folder,view_id,sufijos,maxValues,crop_size=crop_size)
        
        mask=torch.all(image>0,dim=0)
        
        area=torch.sum(mask)
        suma=torch.sum(image,axis=pix_dimensions)
        image2=image*image
        suma2=torch.sum(image2,axis=pix_dimensions)
        area_total +=area
        medias.append(suma)
        medias2.append(suma2)
        
    medias=torch.stack(medias).sum(axis=0)/area_total
    medias2=torch.stack(medias2).sum(axis=0)/area_total
    stds=np.sqrt(medias2 - medias*medias)

    return medias,stds

def normalizar(dataset,medias,stds):
    pix_dimensions=(1,2)
    mediass=medias.unsqueeze(pix_dimensions[0]).unsqueeze(pix_dimensions[1])
    stdss=stds.unsqueeze(pix_dimensions[0]).unsqueeze(pix_dimensions[1])
    out=[]
    for caso in dataset:
        
        image=caso['image']
        image2 = image-mediass
        image2 = image2/ stdss
        dict_vista={'fruit_id':caso['fruit_id'], 'view_id':caso['view_id'], 'image': image2, 'labels': caso['labels'] }

        out.append(dict_vista)
    return out

def normalizar_imagen(im,medias,stds):
    pix_dimensions=(1,2)
    medias=medias.unsqueeze(pix_dimensions[0]).unsqueeze(pix_dimensions[1])
    stds=stds.unsqueeze(pix_dimensions[0]).unsqueeze(pix_dimensions[1])
 
    im2 = im - medias
    im2  = im2/stds
        
    return im2