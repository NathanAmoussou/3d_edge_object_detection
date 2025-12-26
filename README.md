# Edge Computing et IA embarquée
### Objectifs
- Choisir un modèle de vision compatible avec les différents matériels utilisés
- Appliquer sur ce modèle différentes optimisations
- Mesurer différentes métriques sur les différents matériels et avec les différentes versions du modèle de vision
- Écrire un rapport sur ces travaux
### Modèle choisi
- [`YOLO11`](https://github.com/ultralytics/ultralytics), modèle de détection/classification
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

| Optimisation       | Paramètre                               | Granularité                                                                                                                                                    |
| ------------------ | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Élagage            | `model scale` ou `depth/width multiple` | Discret :<br>- Nano (`yolo11n`)<br>- Small (`yolo11s`)<br>- Medium (`yolo11m`)<br>Continu :<br>Modifier le `.yaml`, par ex `width_multiple` (0.25, 0.50, 0.75) |
| Quantification     | `precision`                             | - FP32 (Base, aucune quantif)µ<br>- FP16 (Half-precision, idéal Jetson)<br>- INT8 (Entier 8-bits, idéal NPU/Coral)                                             |
| Fusion de couches  | `GraphOptimizationLevel` (ONNX Runtime) | - `ORT_DISABLE_ALL` (aucune fusion)<br>- `ORT_ENABLE_BASIC` (fusion de base)<br>- `ORT_ENABLE_ALL` (fusion complexe)                                           |
| Distillation       | Taille du `Teacher`                     | - Teacher = `yolo11s` (petit prof)<br>- Teacher = `yolo11m` (moyen prof)<br>- Teacher = `yolo11x` (grand prof)                                                 |
| *Input resolution* | `imgsz` (taille image en pixels)        | Continu (par pas de 32) :<br>- 640 (standard)<br>- 512<br>- 416<br>- 320 (rapide)<br>- 256 (très rapide)                                                       |

##### Optimisations spécifiques à chaque matériel
- À compléter
### Métriques mesurées
- À compléter
### Roadmap
- [x] Trouver un modèle compatible avec les différents matériels, qu'on puisse ***fine-tuner***  et **manipuler** (appliquer les optimisations)
	- [`YOLO11`](https://github.com/ultralytics/ultralytics)
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
### Remarques
- Le VPU de la caméra ne peux pas faire tourner un `YOLO11n` sans aucune optimisation.
- Il faut donc définir la *baseline* la moins optimisée possible.
- La caméra OAK D Pro embarque un VPU Myriad X, le `YOLO11` le moins optimisé sera :
	- Modèle (élagage) : `YOLO11n`
	- Résolution (*input resolution*) : 640
	- Précision (quantification) : FP16
	- Fusion de couche : `ORT_DISABLE_ALL`
- Remarque : Attention à la conversion en Blob qui risque d'appliquer des optimisations, savoir lesquelles.
- On fera les modifications et évaluations des performances sur une version non *fine-tuné* de `YOLO11n`, car ça sera plus fiable que sur notre version *fine-tuné* (utile principalement pour le projet avec ROSS). Par contre, on pourra appliquer les optimisations trouvées ici sur la version ROSS.