import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt


def spline(P0,P1,P2,t):
    return (1-t)*(1-t)*P0 +2*t*(1-t)*P1 + t*t*P2


def draw_line(im,P0,P1,radio=2,color=1,flat=True):
    '''
    Modifica im
    '''
    #print("color=",color)
    d=np.sqrt(np.sum((P0-P1)**2))

    f=np.expand_dims(np.linspace(0,1,int(d)+10),0)
    
    v=np.expand_dims(P1-P0,1)
    #print(v)
    P0o=np.expand_dims(P0,1)

    puntos=P0o+f*v
    copia=im
    puntos=puntos

    vn=v/np.linalg.norm(v)
    w=np.array([-vn[1],vn[0]])#Perpendicular
    
    

    for e in np.linspace(-radio,radio,round(3*radio+1)):
        offset=e*w
        puntoso=np.round(puntos+offset).astype('int')
        #print(puntoso.shape,)
        if flat==False:
            d=math.sqrt((radio+1)**2-e**2)/(radio+1)
        else:
            d=1
        
        copia[puntoso[1,:],puntoso[0,:]]=np.array(color*d)
    return copia


def draw_polyline(im,puntos,radio=2,color=1,flat=True):
    '''
    puntos lista de arrays de 2 elementos o matriz de Nx2
    '''
    
    npuntos=len(puntos)
    
    copia=im.copy()
    for k in range(1,npuntos):
        P0=puntos[k-1]
        P1=puntos[k]
        draw_line(copia,P0,P1,radio,color,flat)

        
    return copia


def draw_spline(im,P0,P1,P2,radio=2,color=1,flat=True):
    puntos=[]
    for t in np.linspace(0,1.0,20):
        p=spline(P0,P1,P2,t)
        puntos.append(p)
    #print(puntos)
    copia=im.copy()
    out=draw_polyline(copia,puntos,radio,color,flat)
    return out

class SimulaDefectoRayos:
    '''
    Clase para simular defectos en una imagen rayos X
    El constructor recibe unos parámetros que definen la probabilidad de que haya un defecto y sus características
    así como la imagen a la que se le van a simular los defectos
    
    El método call simula los defectos en la imagen y devuelve la imagen con los defectos y la máscara de los defectos
    
    
    params={
    'prob_no_change':0.5,
    'min_number_of_defects':1,
    'max_number_of_defects':5,
    'alpha_low': 0.012
    'alpha_high': 0.009
    'min_defect_intensity':0.1,
    'max_defect_intensity': 0.3,
    'min_defect_size':20,
    'max_defect_size':50,
    'min_defect_width':3,
    'max_defect_width':10,
    'defect_types':{'FlatLine': 1.0, 'CylLine': 1.0, 'FlatSpline': 1.0, 'CylSpline':1.0}


    En defect_type tenemos un diccionario con tipos de defecto y su probabilidad relativa
    '''
    def __init__(self,params):

        self.params=params
        self.defectos=[]
        probs=[]
        for k,v in params['defect_types'].items():
            self.defectos.append(k)
            probs.append(v)
        self.probs = probs
        
    def __call__(self,im):
        '''
        Si monocroma (w,h)
        Si color (3,w,h)
        
        Esta trabaja con numpy
        '''
                
        a=np.random.rand()

        num_defects=np.random.randint(self.params['min_number_of_defects'],self.params['max_number_of_defects']+1)
        #print('Num defects:',num_defects)   
        
        tipos_defecto = random.choices(self.defectos,weights=self.probs,k=num_defects)
        

        if im.ndim==2:
            img_width=im.shape[1]
            img_height=im.shape[0]
        else:
            img_width=im.shape[2]
            img_height=im.shape[1]
        #im_defects=np.zeros((img_height*3,img_width*3))            
        im_defects=np.zeros((img_height,img_width))            
        if a<self.params['prob_no_change']:
            return im,np.zeros((img_height,img_width))
        for k in range(num_defects):
            tipo_defecto = tipos_defecto[k]
            #print(tipo_defecto)

            defect_size=np.random.uniform(self.params['min_defect_size'],self.params['max_defect_size'])
            defect_width=np.random.uniform(self.params['min_defect_width'],self.params['max_defect_width'])

            guarda=self.params['max_defect_width']+self.params['max_defect_size']+ max(img_height,img_height)//12
            center_x=np.random.uniform( guarda,img_width-guarda)
            center_y=np.random.uniform(guarda,img_height -guarda)
            orientation=np.random.uniform(0,2*np.pi)
            color_low=defect_width*self.params['alpha_low']
            color_high=defect_width*self.params['alpha_low']
            
            
            
            pp=np.array([center_x,center_y])
            vv=np.array([np.cos(orientation),np.sin(orientation)])
            P0=pp-defect_size*vv/2
            P2=pp+defect_size*vv/2
            P1=(P0+P2)/2+np.random.uniform(-defect_size/3,defect_size/3)*np.array([-vv[1],vv[0]])
            
            flat= [True if 'Flat' in tipo_defecto else False]
            copia=np.zeros((img_height,img_width))        
            
            if 'Line' in tipo_defecto:
                out=draw_line(copia,P0,P2,defect_width,color_low,flat)
            elif 'Spline' in tipo_defecto:
                out=draw_spline(copia,P0,P1,P2,defect_width,color_low,flat)
            #print("max out=", out.max())    
            im_defects += out
            
            #print("global max:",im_defects.max())
        #_=plt.imshow(im_defects,cmap='gray')    
        im_defects_low = im_defects
        im_defects_high = color_high/color_low * im_defects_low
        im_defects_media=(im_defects_low+im_defects_high)/2
        
        stack_defects=np.stack((im_defects_low,im_defects_high,im_defects_media),axis=0)
        
        gain=np.exp(-stack_defects)
                      
        mascara=(im_defects>0)

        return im*gain , mascara
    
    def processTensor(self,imtensor):
        im=imtensor.numpy()
        im,mask=self(im)
        im=torch.from_numpy(im.astype(float)).to(torch.float32)
        mask=torch.from_numpy(mask.astype(float)).to(torch.float32)
        
        return im,mask