from torch.utils.data import Dataset
import cv2
import torch
import os
import json
import simulador_defectos
from torchvision import transforms
import pathlib
import math


def read_image_rayosX(filename,max_value,scale=0.7):
	'''
	El primer canal es low, el segundo es high y el tercero la media
	'''
	#print("Reading image", filename		)
	im=cv2.imread(filename,cv2.IMREAD_UNCHANGED)
	if im is not None:
		if im.ndim ==3 :
			im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
   		
		widht=im.shape[1]
		height=im.shape[0]
		widthn=int(scale*widht)
		heightn=int(scale*height)
		im=cv2.resize(im,(widthn,heightn),interpolation=cv2.INTER_AREA)

		im=im.astype('float32')/max_value
		im[im>1]=1

		im=torch.tensor(im)
		if im.ndim ==2:# convertir de hxw --> 1 x h x w
			im=im.unsqueeze(0)
		else:
			im=im.permute((2,0,1))

	
	return im



	
def get_dataset(dataplaces, images_root_folder="",in_memory=False, terminaciones=None, max_values=None,delimiter='.'):
	'''
	Recibe una lista de dataplaces
	Cada dataplace es una tupla de dos elementos
		El primer elemento es el nombre de un fichero de texto con una lista de imagenes
		El segundo elemento es el directorio del que cuelgan las imágenes

	Devuelve una lista de diccionarios con dos elementos
	{'filename': nombre del fichero de texto, 'pixels': tensor } 
	con la imagen 3 x h x w
	filename es el path completo
	'''
	
	dataset=[]
	for dataplace in dataplaces:	
		filename=dataplace[0]
		directory=dataplace[1]
		
		with open(filename) as f:
			images = f.readlines()
		for im in images:
			im = im.strip()
			view_id=im.split(delimiter)[:-1]

			
			filenames=[]
			pixels=None # De momento no he leido nada
			for t in range(len(terminaciones)):
				channel=delimiter.join(view_id)
				channel=channel+terminaciones[t]
				
				channel=os.path.join(images_root_folder,directory, channel)
				filenames.append(channel)
	
				if in_memory:
					print("Reading image", channel)
					pixels_channel = read_image_rayosX(channel,max_values[t])
					if pixels is None:
						pixels=pixels_channel
					else:
						pixels=torch.cat((pixels,pixels_channel),0)
				else:
					pixels = None
			#print(">>>>>>>>>>>>>>>>>>> view_id: ",view_id, "im: ",im, filenames)
			dataset.append({'filenames': filenames, 'pixels': pixels})
	return dataset



		
def compute_normalization(dataset,terminaciones,max_values):
	'''
	Dado un dataset, lista de diccionarios, devuelve un diccionario con la media y la desviación estándar de cada canal
	'''
	medias=[]
	stds=[]
	suma=0
	suma2=0

	for el in dataset:
		pixels = el['pixels']
		if pixels is  None:
			pixels=[]
			#print('el[filenames]',el['filenames'])
			for t in range(len(terminaciones)):
				pixels_channel = read_image_rayosX(el['filenames'][t],max_values[t])
				pixels.append(pixels_channel)
			pixels=torch.cat((pixels),0)
		media=torch.mean(pixels, dim=(1,2))
		stdev=torch.std(pixels, dim=(1,2))
		medias.append(media)
		stds.append(stdev)

	

	out={'medias_norm': torch.mean(torch.stack(medias,axis=0),axis=0).tolist(), 'stds_norm': torch.mean(torch.stack(stds,axis=0),axis=0).tolist()}
	#print("out compute_normalization", out)
	return out



class DataplacesDataSetSimulaDefectos(Dataset):
	'''
	Clase para suministrar archivos dataplaces
	Incluye la adición de defectos de acuerdo con los parámetros de simulación recibidos
	'''
	
	def __init__(self,root_folder=None, dataplaces=None ,transform=None,in_memory=False,terminaciones=None,normalization_params=None,
					normalization_image_params=None,normalization_image_size=(224,224),
					max_values=None,delimiter='.',
					params_simulacion_defectos=None,
					scale=1.0,
					logaritmo=True,
					monocanal=False):
		#super().__init__()

		self.root_folder=root_folder
		self.dataplaces=dataplaces
		self.transform = transform
		self.terminaciones=terminaciones
		self.max_values=max_values
		self.delimiter=delimiter
		self.scale=scale
		self.monocanal = monocanal

		self.logaritmo=logaritmo
		print(">>>>>>>>>>>> DataplacesDataset terminaciones", terminaciones, "delimiter", delimiter)	
		self.dataset = get_dataset(dataplaces, images_root_folder=root_folder,in_memory=in_memory, 
									terminaciones=terminaciones,max_values=max_values,delimiter=self.delimiter)
		self.normalization_image_size=normalization_image_size
		assert params_simulacion_defectos is not None

		self.params_simulacion_defectos=params_simulacion_defectos
		self.simulador=simulador_defectos.SimulaDefectoRayos(self.params_simulacion_defectos)

		print(">>>>>>>>>> scale in DataSet", self.scale)

		# if normalization_params is None or normalization_image_params is None:
		#     self.normalization_params,self.normalization_image_params=dataset.compute_normalization(self.dataset,self.terminaciones,self.max_values,normalization_image_size)
		if normalization_params is None:
			self.normalization_params=compute_normalization(self.dataset,self.terminaciones,self.max_values)    
			
			pathlib.Path('modelos').mkdir(parents=True, exist_ok=True)
			with open('modelos/normalization.json','w') as json_file:
				json.dump(self.normalization_params,json_file, indent=4)
			print('Datos de normalizacion calculados y guardados en modelos/normalization.json')
		else:
			self.normalization_image_params=normalization_image_params
			self.normalization_params=normalization_params
			print('Datos de normalizacion cargados')
			
	def setTransform(self, transform):
		self.transform = transform

				
	def __getitem__(self, index: int) :              
		caso=self.dataset[index]
		max_values=self.max_values
		
		
		
		pixels=caso['pixels']

		if pixels is None:
			pixels=[]
			for t in range(len(self.terminaciones)):
				#print("getting item", caso['filenames'][t],index,self.__len__())
				assert os.path.exists(caso['filenames'][t]), caso['filenames'][t]
				pixels_channel=read_image_rayosX(caso['filenames'][t],max_values[t],scale=self.scale)
				pixels.append(pixels_channel)
			pixels=torch.cat((pixels),0)
	
		pixels, defect_mask = self.simulador.processTensor(pixels)
		


		if self.transform is not None:  
			pixels , defect_mask = self.transform(pixels,defect_mask) # Transforma pixeles y mascaras inteligentemente
			
		if self.logaritmo:
			pixels=torch.log(pixels+1/max_values[0])/math.log(0.2)            

		if self.monocanal:
			pixels=torch.mean(pixels,dim=0,keepdim=True)	
			pixels=torch.concat([pixels]*3,dim=0)
		return pixels, caso['filenames'],defect_mask
				
	def __len__(self) -> int:
		return len(self.dataset)

