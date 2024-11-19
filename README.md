# Herbarium Segmentation

This repository contains the code,  dataset and pretrained models from the following publication:

*Enhancing Plant Morphological Trait Identification in Herbarium Collections through Deep Learning-based Segmentation*

## Dataset

Download link : https://figshare.com/s/ab9813a29f0ce2e4e8d5

* 2,277 (image, mask)
* Two different backgrounds for masks : Black Background (BB)  and White  Background (WB).
* 11 different families and genera:

	1.  Amborella (91 images), 
	2. *Castanea* (161 images), 
	3. *Desmodium* (164 images), 
	4. *Ulmus* (352 images), 
	5. *Rubus* (184 images), 
	6. *litsea* (199 images),
	7.  *eugenia* (219 images), 
	8. *laurus* (250 images), 
	9. *Convolvulaceae* (177 images),
	10.  *Magnolia* (162 images) and 
	11. *Monimiaceae* (318 images))

* 1821 images for training
* 456 imgaes for validation

## Cite the dataset

SKLAB, Youcef; Sklab, Hanane; Prifti, Edi; Chenin, Eric; Zucker, Jean-Daniel (2024). Herbarium Image Segmentation Dataset with Plant Masks for Enhanced Morphological Trait Analysis. figshare. Dataset. https://doi.org/10.6084/m9.figshare.27685914.v1

## Pretrained models

Download link: https://drive.google.com/drive/folders/1jvUWrNkLfECmdCBDJ0f7bFOo1j8qQGxv?usp=sharing

## Inference and Training: 

	For training: 
	1. 	Install requirements packages: pip install requirements.txt
	2. 	run train.py and configure your datasets. 

	For Inference: 
	1. 	run predict-WB.py for white background or 
	2. 	run predict-BB.py for black background. 




## Citation
If you use this work, including the code or dataset, please cite it as follows:

Hanane Ariouat, Youcef Skla, Edi Prifti, Jean-Daniel Zucker, Eric Chenin.  
*Enhancing Plant Morphological Trait Identification in Herbarium Collections through Deep Learning-based Segmentation: Application in Plant Science. 2024.* In Press.

