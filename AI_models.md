# Modèles d'IA testés
## Axes d'optimisation
### Optimisations agnostiques du matériel
- Élagage structuré (*structured pruning*)
- Distillation de connaissances (*knowledge distillation*)
- Quantification (*quantisation*)
- Réduction de la résolution d'entrée (*input resolution scaling*)
- Fusion de couches (*layer fusion*)
### Optimisations spécifiques à chaque matériel
- [Luxonis OAK-D Pro](https://docs.luxonis.com/hardware/products/OAK-D%20Pro)
	- Modification du nombre de SHAVEs alloués lors de la compilation
	- Délestage du post-traitement (Yolo Node), à explorer
- ReComputer J3010-Edge avec NVIDIA Jetson Orin Nano 4 G
	- Le paramètre "Builder Optimization Level" (optimisation)
	- Le paramètre "Sparsity" (accélération matérielle spécifique à Orin)

| Optimisation                     | Paramètre                                | Granularité                                                                                                                                                    |
| -------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Élagage                          | `model scale` ou `depth/width multiple`  | Discret :<br>- Nano (`yolo11n`)<br>- Small (`yolo11s`)<br>- Medium (`yolo11m`)<br>Continu :<br>Modifier le `.yaml`, par ex `width_multiple` (0.25, 0.50, 0.75) |
| Distillation                     | Taille du `Teacher`                      | - Teacher = `yolo11s` (petit prof)<br>- Teacher = `yolo11m` (moyen prof)<br>- Teacher = `yolo11x` (grand prof)                                                 |
| Quantification                   | `precision`                              | - FP32 (Base, aucune quantif)<br>- FP16 (Half-precision, idéal Jetson)<br>- INT8 (Entier 8-bits, idéal NPU/Coral)                                              |
| *Input resolution*               | `imgsz` (taille image en pixels)         | Continu (par pas de 32) :<br>- 640 (standard)<br>- 512<br>- 416<br>- 320 (rapide)<br>- 256 (très rapide)                                                       |
| Fusion de couches                | `GraphOptimizationLevel` (ONNX Runtime)  | - `ORT_DISABLE_ALL` (aucune fusion)<br>- `ORT_ENABLE_BASIC` (fusion de base)<br>- `ORT_ENABLE_ALL` (fusion complexe)                                           |
| SHAVEs OAK                       |                                          | - 4<br>- 5<br>- 6<br>- 7                                                                                                                                       |
| Délestage du post-traitement OAK |                                          | - False<br>- True                                                                                                                                              |
| Optimisation Orin                | `trtexec --builderOptimizationLevel=N`   | - 3<br>- 5                                                                                                                                                     |
| Accélération matérielle Orin     | `trtexec --sparsity=enable` (ou `force`) | - False<br>- True                                                                                                                                              |

## Ordre des optimisations
- Sans **distillation** :
	1. Élagage
	2. Quantification + *Input resolution*
	3. Fusion
- Avec **distillation** : 
	1. Élagage + Distillation
	2. Quantification + *Input resolution*
	3. Fusion
## Algorithme de productions des variantes
```
elagage_bins = [yolo11m, yolo11s, yolo11n]
distillation_bins = [False, True]
quantification_bins = [fp32, fp16, int8]
resolution_bins = [640, 512, 416, 320, 256]
fusion_bins = [ORT_ENABLE_ALL, ORT_ENABLE_BASIC, ORT_DISABLE_ALL]

for d in distillation_bins:
	for e in elagage_bins:
		for q in quantification_bins:
			for r in resolution_bins:
				for f in fusion_bins:
					ProduceVariant(d, e, q, r, f)
```
### Remarques
- Pour un *hardware* donné, définir le point de départ, la variante la moins optimisée, puis appliquer progressivement des optimisations.
- Tous les *hardwares* n'auront pas le même point de départ.
- La couche distillation risque d'être lente. D'abord développer tous les modèles sans distillation, ensuite aviser.
- **Se renseigner sur les optimisations obligatoires ou par défaut appliquées lors des différentes compilations (Blob, ONNX...).**
### Nombres de variantes et *bins* par *hardware*
```
if hardware == "4070": # onnx runtime
	elagage_bins = [yolo11m, yolo11s, yolo11n]
	quantification_bins = [fp32, fp16, int8]
	resolution_bins = [640, 512, 416, 320, 256]
	fusion_bins = [ORT_ENABLE_ALL, ORT_ENABLE_BASIC, ORT_DISABLE_ALL]
	
	nb_variantes = 3 * 3 * 5 * 3 = 135

if hardware == "oak": # depthai runtime
	elagage_bins = [yolo11m, yolo11s, yolo11n]
	quantification_bins = [fp16]
	resolution_bins = [640, 512, 416, 320, 256]
	shaves_bins = [4, 5, 6, 7, 8]
	fusion_bins = [False, True]
	
	nb_variantes = 3 * 1 * 5 * 5 * 2 = 150

if hardware == "orin":
	elagage_bins = [yolo11m, yolo11s, yolo11n]
	quantification_bins = [fp32, fp16, int8]
	resolution_bins = [640, 512, 416, 320, 256]
	
	if runtime == "ort":
		fusion_bins = [ORT_ENABLE_ALL, ORT_ENABLE_BASIC, ORT_DISABLE_ALL]
		
		nb_variantes = 3 * 3 * 5 * 3 = 135
	
	elif runtime == "trt":
		optim_bins = [3, 4, 5]
		sparsity = [False, True]
		
		nb_variantes = 3 * 3 * 5 * 2 * 2 = 180
```
## Point de départ de chaque *hardware*

| *Hardware*                | Élagage min.          | Quantification min. | Résolution min. | Fusion min.        |
| ------------------------- | --------------------- | ------------------- | --------------- | ------------------ |
| 4070 laptop 8GB, 32GB RAM | `yolo11m`             | `fp32`              | `640`           | `ORT_DISABLE_ALL`  |
| i9-14900HX, 32GB RAM      | `yolo11m`             | `fp32`              | `640`           | `ORT_DISABLE_ALL`  |
| OAK-D Pro (Myriad X)      | `yolo11m`             | `fp16`              | `640`           | Obligatoire (Blob) |
| Jetson Orin Nano 4GB      | `yolo11m` ? `yolo11s` | `fp16`              | `640`           | `ORT_DISABLE_ALL`  |
| Raspberry Pi 4B           | `yolo11n` ?           | `fp32`              | `640`           | `ORT_DISABLE_ALL`  |
### Remarques
- Pour le OAK-D Pro (Myriad X) :
	- Le `yolo11m` est la limite haute théorique qui compile souvent, parfois au prix d'une réduction de résolution ou de shaves, si `m` ne compile pas, descendre à `s`.
	- Myriad X ne supporte pas le `fp32` nativement, notre *baseline* la moins optimisée est donc obligatoirement le `fp16`.
	- Le compilateur **BlobConverter** fusionne obligatoirement les couches pour créer le binaire, on ne peut pas le désactiver (d'où "Obligatoire").
- Pour Jetson Orin Nano 4GB : 
	- la RAM est partagée CPU et GPU, l'OS prend déjà 1-2 Go, `m` crachera probablement, mais à tester.
	- `fp16` et la pratique courante, mais à tester `fp32`.
- Pour Raspberry Pi 4B :
	-  On peut techniquement lancer un `yolo11s` ou `m` en `fp32`, mais on aura une latence de 2 à 10 secondes par image.
- La fusion de couches et la quantification `int8` seront faites à l'inférence.
- Du coup, peut-être utiliser les `.json` lors du *bench* pour formater les images à la bonne taille, par exemple.
## *Roadmap*
- [x] Implémenter les optimisations avant le *runtime*
	- Fait avec `scripts/generate_variants.py`
- [x] Implémenter les optimisations aux *runtime*
	- Fait dans `scripts/benchmark.py`
	- Attention : On a uniquement fait la fusion et `int8` pour ORT
- [ ] Implémenter les optimisations spécifiques à chaque *hardware*, notamment la fusion sur TRT et DRT
## Optimisations supportées actuellement

| *Hardware* (et *runtime*) | Élagage | Quantification                                    | Résolution | Fusion / optimisations “type fusion”                                                     | Nombre de modèles testables          |
| ------------------------- | ------- | ------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------- | ------------------------------------ |
| 4070 (ORT)                | Tous    | FP32/FP16 + (INT8 si modèle QDQ et EP compatible) | Tous       | ORT Graph Optimizations (basic/extended/all) au *runtime*                                | $3 \times 3 \times 5 \times 3 = 135$ |
| Orin (TRT natif)          | Tous    | FP32/FP16 + INT8 (au build)                       | Tous       | Optimisations TensorRT au *build* (tactics, fusion, etc.) (**à finir d'implém.**)        | Au moins $3 \times 3 \times 5 = 45$  |
| Orin (ORT)                | Tous    | FP32/FP16 + INT8 (QDQ)                            | Tous       | ORT Graph Optimizations (basic/extended/all) au *runtime*                                | $3 \times 3 \times 5 \times 3 = 135$ |
| OAK-D (DepthAI + MyriadX) | Tous    | FP16 (MyriadX/RVC2 en pratique)                   | Tous       | Optimisations OpenVINO au *build* ONNX→blob (pas de niveaux ORT) (**à finir d'implém.**) | Au moins $3 \times 1 \times 5 = 15$  |
### Test rapide
```
python scripts/sweep.py --targets 4070 --ort-levels disable all
```

```
python scripts/sweep.py --targets oak --compile-oak
```
## Mesures sur 4070 ORT
- Générer les 30 `fp32`/`fp16` :
```
python scripts/generate_variants.py \
  --output models/variants \
  --models n s m \
  --resolutions 640 512 416 320 256
```
- Générer les 15 `int8` :
```
python scripts/transform.py \
  --input-dir models/variants \
  --pattern "yolo11*_fp32.onnx" \
  --int8 \python scripts/sweep.py --targets 4070 --ort-levels disable basic all

  --int8-act-type qint8 \
  --calib-dataset coco128 \
  --calib-size 100 \
  --output models/transformed \
  --skip-existing
```
- Vérifier les *runtime providers* :
```
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
Retourne : `python scripts/sweep.py --targets 4070 --ort-levels disable basic all --dry-run`
- *Dry-run* :
```
python scripts/sweep.py --targets 4070 --ort-levels disable basic all --dry-run
```
Retourne : `...Expected runs: 135...`
- *Run* :
```
python scripts/sweep.py --targets 4070 --ort-levels disable basic all
```