# Edge Computing et IA embarquée
## Objectifs
- Choisir un modèle de vision compatible avec les différents matériels utilisés
- Appliquer sur ce modèle différentes optimisations
- Mesurer différentes métriques sur les différents matériels et avec les différentes versions du modèle de vision
- Écrire un rapport sur ces travaux
## Modèle choisi
- [`YOLO11`](https://github.com/ultralytics/ultralytics), modèle de détection/classification
## Matériels disponibles
- [PC de Nathan](http://ldlc.com/fiche/PB00591624.html) (i9-14900HX, 4070 8GB, 32GB RAMA)
- [Luxonis OAK-D Pro](https://docs.luxonis.com/hardware/products/OAK-D%20Pro)
- ReComputer J3010-Edge avec NVIDIA Jetson Orin Nano 4 G
- Raspberry Pi 4B
## Optimisations envisagées
### Optimisations agnostiques du matériel
- Élagage structuré (*structured pruning*)
- Distillation de connaissances (*knowledge distillation*)
- Quantification (*quantisation*)
- Réduction de la résolution d'entrée (*input resolution scaling*)
- Fusion de couches (*layer fusion*)
### Optimisations spécifiques à chaque matériel
- À compléter
## Métriques mesurées
### Métriques *hardware* (côté matériel)
- Consommation instantanée (puissance en Watts)
- Énergie par inférence (en Joules) : $Joules = Watts \times Temps(secondes)$
- Taille du modèle (*storage footprint*), taille fichier et/ou nombre de paramètres du modèle
	- Fait
- Pic mémoire (*peak RAM usage*), quantité max de RAM utilisée
- Température et *throttling*
- Taux d'utilisation du CPU/GPU/...
### Métriques *software* (côté IA)
- mAP@50 (*mean average precision*), vérifie si la boîte englobante (*bounding box*) chevauche au moins à 50% la vérité terrain
	- Fait
- Precision / Recall / F1
	- Fait
- Matrice de confusion (le modèle confond quels objets avec quels objets)
- E2E
	- Fait
- Device Time
	- Fait
- Preprocess
	- Fait
- Inference
	- Fait
- Postprocess
	- Fait
- p50/p90/p95/p99
	- Fait
- FPS (*throughput*)
	- Fait
### Métriques combinées
- FPS / Watt (Efficacité), combien d'images on traite pour 1 Watt
- Densité de performance, FPS / Prix du matériel ($)
## *Roadmap*
- [x] Trouver un modèle compatible avec les différents matériels, qu'on puisse ***fine-tuner***  et **manipuler** (appliquer les optimisations)
	- [`YOLO11`](https://github.com/ultralytics/ultralytics)
- [x] *Fine-tuner* ce modèle pour détecter des **cubes rouges**
	- [x] Créer le jeu de données
		- Photos prises puis on utilise [CVAT AI](https://www.cvat.ai/)
	- [x] Créer le *pipeline* d'entraînement
		- Fait avec `notebooks/train_redcube.ipynb`
	- [x] Entraîner, déployer sur la caméra, tester
		- Fait avec `scripts/detect_oak.py`
- [x] Récupérer et préparer le modèle de base, non *fine-tuné*
	- Fait avec `notebooks/base_export.ipynb`
- [x] Développer le script MVP de *benchmark*
	- [x] Développer la partie **compilation**
		- Fait avec `scripts/compile.py`
	- [x] Développer la partie **mesures**
		- Fait avec `scripts/benchmark.py`
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
## Remarques
- Le VPU de la caméra ne peux pas faire tourner un `YOLO11n` sans aucune optimisation.
- Il faut donc définir la *baseline* la moins optimisée possible.
- La caméra OAK D Pro embarque un VPU Myriad X, le `YOLO11` le moins optimisé sera :
	- Modèle (élagage) : `YOLO11n`
	- Résolution (*input resolution*) : 640
	- Précision (quantification) : FP16
	- Fusion de couche : `ORT_DISABLE_ALL`
- Remarque : **Attention à la conversion en Blob qui risque d'appliquer des optimisations**, savoir lesquelles.
- On fera les modifications et évaluations des performances sur une version non *fine-tuné* de `YOLO11n`, car ça sera plus fiable que sur notre version *fine-tuné* (utile principalement pour le projet avec ROSS). Par contre, on pourra appliquer les optimisations trouvées ici sur la version ROSS.
- On développe d'abord le script qui fait les mesures, en mode *contract first*.
- Plus tard, il faudra améliorer ce script probablement `.py` avec les bonnes pratiques pour des mesures saines (inférence à vide, mesures sans le modèle qui tourne, etc.).