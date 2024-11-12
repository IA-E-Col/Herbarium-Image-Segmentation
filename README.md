**Herbarium Segmentation**

This repository contains the code,  dataset and retrained models from the following publication:

*Enhancing Plant Morphological Trait Identification in Herbarium Collections through Deep Learning-based Segmentation*

**Dataset**

See: Download link : [https://drive.google.com/drive/folders/1XX6LNxqnETBTrV54XxykF_N38r4glZBc?usp=sharing](https://drive.google.com/drive/folders/1XX6LNxqnETBTrV54XxykF_N38r4glZBc?usp=sharing)

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

**Pretrained models**
Inference and Training Code: [https://drive.google.com/file/d/1ZDtxJwfK2OqjCcdiZpMeMTeAkFW3XTji/view?usp=drive_link](https://drive.google.com/file/d/1ZDtxJwfK2OqjCcdiZpMeMTeAkFW3XTji/view?usp=drive_link)

	For training: 
	1. 	Install requirements packages: pip install requirements.txt
	2. run train.py and configure your datasets. 

	For Inference: 
	1. 	run predict-WB.py for white background or 
	2. run predict-BB.py for black background. 


![https://drive.google.com/file/d/1-osHGQUvql2jXh1Ck1kyRVSsx2xaDlcC/view?usp=sharing](https://drive.google.com/file/d/1-osHGQUvql2jXh1Ck1kyRVSsx2xaDlcC/view?usp=sharing)


**Citation**
If you use this dataset or code in your research, please use the following BibTeX entry:

A faire


