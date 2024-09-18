import torch
from torchvision import transforms


class Aumentador_Imagenes_y_Mascaras():
    def __init__(self, geometric_transforms, color_transforms,normalize_transforms):
        self.geometric_transforms = geometric_transforms
        self.color_transforms = color_transforms
        self.normalize_transforms=normalize_transforms

    def __call__(self, img, mask):
        ncanales=img.shape[0]
        if self.color_transforms is not None:
            RGB=img[:3,:,:]
            if img.shape[0] >3 :
                canales_extra=img[3:,:,:]
                canales_extra1=self.color_transforms(canales_extra)
            RGB1 = self.color_transforms(RGB) # Solo a la imagen
            if img.shape[0] >3 :
                img1=torch.concat([RGB1,canales_extra1],dim=0)
            else:
                img1=RGB1
        else:
            img1=img
        
        
        taco=img1


        taco=torch.cat((taco,1-mask.unsqueeze(0)),0)
        taco=self.geometric_transforms(taco)
        
        img2=taco[:ncanales,:,:]
        mask2=1-taco[ncanales,:,:]    
        
        if self.normalize_transforms is not None:
            img2=self.normalize_transforms(img2)
        

            
        return img2, mask2