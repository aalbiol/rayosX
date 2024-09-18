import cv2
import numpy as np
import torch


def spline(P0,P1,P2,t):
    return (1-t)*(1-t)*P0 +2*t*(1-t)*P1 + t*t*P2


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
}
    '''
    def __init__(self,params):

        self.params=params
        
        
    def __call__(self,im):
        '''
        Si monocroma (w,h)
        Si color (3,w,h)
        '''
        a=np.random.rand()

        num_defects=np.random.randint(self.params['min_number_of_defects'],self.params['max_number_of_defects']+1)
        #print('Num defects:',num_defects)   

        if im.ndim==2:
            img_width=im.shape[1]
            img_height=im.shape[0]
        else:
            img_width=im.shape[2]
            img_height=im.shape[1]
        im_defects=np.zeros((img_height*3,img_width*3))            
        if a<self.params['prob_no_change']:
            return im,np.zeros((img_height,img_width))
        for k in range(num_defects):
            defect_intensity=np.random.uniform(self.params['min_defect_intensity'],self.params['max_defect_intensity'])
            defect_size=np.random.uniform(self.params['min_defect_size'],self.params['max_defect_size'])*3
            defect_width=np.random.uniform(self.params['min_defect_width'],self.params['max_defect_width'])*3
            center_x=np.random.uniform(0.15,0.85)*img_width*3
            center_y=np.random.uniform(0.15,0.85)*img_height*3
            orientation=np.random.uniform(0,2*np.pi)
            
            pp=np.array([center_x,center_y])
            vv=np.array([np.cos(orientation),np.sin(orientation)])
            P0=pp-defect_size*vv/2
            P2=pp+defect_size*vv/2
            P1=(P0+P2)/2+np.random.uniform(-defect_size/3,defect_size/3)*np.array([-vv[1],vv[0]])
            
            puntos=[]
            for t in np.linspace(0,1.0,20):
                p=spline(P0,P1,P2,t)
                puntos.append(p)
            puntos=np.array(puntos,dtype=int)
            copia=np.zeros_like(im_defects)
            #print('Dibujando with ',defect_intensity, defect_width, defect_size, " at ",center_x//3,", ",center_y//3)
            im_defects += cv2.polylines(copia,[puntos],False,defect_intensity, int(defect_width+0.5))
            
        mascara=(im_defects>0)
        mascara=cv2.resize(mascara.astype(float),(img_width,img_height),interpolation=cv2.INTER_NEAREST)
        im_defects=cv2.resize(im_defects,(img_width,img_height),interpolation=cv2.INTER_NEAREST)
        return im-im_defects , im_defects>0
    
    def processTensor(self,imtensor):
        im=imtensor.numpy()
        im,mask=self(im)
        im=torch.from_numpy(im.astype(float)).to(torch.float32)
        mask=torch.from_numpy(mask.astype(float)).to(torch.float32)
        
        return im,mask