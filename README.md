# Edge Computing et IA embarquée
### Objectifs
- Choisir un modèle de vision compatible avec les différents matériels utilisés
- Appliquer sur ce modèle différentes optimisations
- Mesurer différentes métriques sur les différents matériels et avec les différentes versions du modèle de vision
- Écrire un rapport sur ces travaux
### Modèle choisi
- [YOLO11](https://github.com/ultralytics/ultralytics), modèle de détection/classification
### Matériels disponibles
- [Luxonis OAK-D Pro](https://docs.luxonis.com/hardware/products/OAK-D%20Pro)
- Nvidia Jetson Orin
- Raspberry Pi 4B
### Optimisations envisagées
##### Optimisations agnostiques du matériel
- Élagage structuré (*structured pruning*)
- Quantification (*quantisation*)
- Fusion de couches (*layer fusion*)
- Distillation de connaissances (*knowledge distillation*)
- Réduction de la résolution d'entrée (*input resolution scaling*)
##### Optimisations spécifiques à chaque matériel
- À compléter
### Roadmap
- [x] Trouver un modèle compatible avec les différents matériels, qu'on puisse ***fine-tuner***  et **manipuler** (appliquer les optimisations)
	- [YOLO11](https://github.com/ultralytics/ultralytics)
- [x] *Fine-tuner* ce modèle pour détecter des **cubes rouges**
	- [x] Créer le jeu de données
		- Photos prises puis on utilise [CVAT AI](https://www.cvat.ai/)
	- [x] Créer le *pipeline* d'entraînement
		- Fait avec `notebooks/train.ipynb`
	- [x] Entraîner, déployer sur la caméra, tester
		- Fait avec `scripts/detect_oak.py`
- [ ] Développer le *pipeline* d'optimisations/manipulations du modèle
	- [ ] Développer le *pipeline* d'**élagage**
	- [ ] Développer le *pipeline* de **quantification**
	- [ ] Développer le *pipeline* de **fusion de couche**
	- [ ] Développer le *pipeline* de **distillation**
	- [ ] Développer le *pipeline* d'**optimisations spécifiques aux matériels**
	- [ ] Développer le *pipeline* de combinaison de toutes ces optimisations
- [ ] Développer le *pipeline* de mesures modèle/matériel
- [ ] Créer les scripts `.sh` pour lancer les mesures
	- [ ] Créer le `.sh` pour la caméra
	- [ ] Créer le `.sh` pour le Jetson
	- [ ] Créer le `.sh` pour le Raspberry