import cv2
import numpy as np

import os



def loadRayosX(f,maxval=2**16-1,sigma=2):
    '''
    A partir del nombre de la imagen LOW o HIGH
    carga ambas imagenes y las junta en una sola imagen
    con los canales LOW, HIGH y LOW+HIGH/2
    
    Devuelve tanto la imagen original como la imagen filtrada
    '''
    
    if "LOW" in f:
        namelow=f
        namehigh=f.replace("LOW","HI_ORI")
    else:
        namehigh=f
        namelow=f.replace("HI_ORI","LOW")
        
    imlow=cv2.imread(namelow,cv2.IMREAD_UNCHANGED)
    if not os.path.exists(namehigh):
        print("No existe",namehigh)
        namehigh=namelow
    imhigh=cv2.imread(namehigh,cv2.IMREAD_UNCHANGED)
    
    imlow=imlow.astype(np.float32)
    imhigh=imhigh.astype(np.float32)
    
    if maxval is not None:
        imlow/=maxval
        imhigh/=maxval
    img=np.stack([imlow,imhigh,(imlow+imhigh)/2],axis=2)
    tam=2*round(3*sigma)+1
    imgfilt=cv2.GaussianBlur(img,(tam,tam),sigma)
    #zeff=Zeff(imgfilt,maxval)
    # img[:,:,2]=zeff 
    # imgfilt[:,:,2]=zeff
    return img,imgfilt



def recorte(im,imf,threshold=0.85,orla=2,min_area=150000):
    '''
    Devuelve el recorte de máxima área dentro de una imagem
    
    Tanto de la imagen original como de la imagen filtrada
    
    orla: número de pixeles de orla que se añaden alrededor del recorte. Además se hace una erosión de la máscara de ese tamaño
    para asegurarnos de estar dentro de la carne para la normalidad
    '''
    recorte_binaria=imf[:,:,2]<threshold
    # plt.imshow(recorte_binaria)
    contr,_=cv2.findContours(recorte_binaria.astype(np.uint8),cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    contr2=[]
    max_area=0
    contorno_max_area=None
    for c in contr:
        area=cv2.contourArea(c)
        if area<min_area:
            continue
        if area>max_area:
            max_area=area
            contorno_max_area=c

    if contorno_max_area is None:
        return None
    
    bb_recorte=cv2.boundingRect(contorno_max_area)
    h,w=im.shape[:2]
    
    bb_recorte=np.array(bb_recorte)
    bb_recorte[:2]-=orla
    bb_recorte[:2]=np.maximum(bb_recorte[:2],0)
        
    
    bb_recorte[-2:]+=2*orla
    bb_recorte[-2]=min(bb_recorte[-2],w-bb_recorte[0])
    bb_recorte[-1]=min(bb_recorte[-1],h-bb_recorte[1])
    
    x1,y1,w,h=bb_recorte
    out=im[y1:y1+h,x1:x1+w,:]
    outf=imf[y1:y1+h,x1:x1+w,:]
    
    recorte_binaria[:,0]=0
    recorte_binaria[0,:]=0
    
    recorte_binaria[:,-1]=0
    recorte_binaria[-1,:]=0    

    # distancia=cv2.distanceTransform(recorte_binaria.astype(np.uint8),cv2.DIST_L2,5)
    # mascara_interior=distancia>orla
    mascara_interior=recorte_binaria[y1:y1+h,x1:x1+w]
    return out,outf,mascara_interior