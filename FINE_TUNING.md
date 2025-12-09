# Pipeline de développement pour détection “cube + profondeur” avec OAK-D

## 1. Préparer les données  
- [x] Collecter des photos du cube rouge dans des conditions variées (angles, distances, éclairages, arrière-plans, occlusions éventuelles, tailles différentes).  
- [x] Annoter chaque image : délimiter un bounding box autour du cube + assigner le label (par exemple “cube”).  
- [ ] (Optionnel) Appliquer des augmentations d’images : rotations, recadrages, variations de luminosité/contraste, zooms, changements d’arrière-plan, etc.  

## 2. Entraîner / fine-tuner le modèle de détection  
- [ ] Organiser les images annotées dans le format attendu par l’outil d’entraînement (par exemple pour une architecture SSD ou autre).  
- [ ] Partir d’un modèle pré-entraîné (backbone + head de détection) — par exemple un modèle basé sur MobileNet‑SSD ou un détecteur compatible — et modifier la tête de classification pour n’avoir qu’une seule classe (“cube”).  
- [ ] Lancer l’entraînement (fine-tuning) sur le dataset annoté.  
- [ ] Valider les performances du modèle sur un jeu de validation : vérifier que les bounding boxes détectent correctement les cubes, avec des scores de confiance acceptables, et peu de faux positifs / faux négatifs.  

## 3. Convertir le modèle en format compatible pour OAK-D  
- [ ] Exporter le modèle entraîné dans un format standard (par exemple ONNX, ou un format supporté selon le framework).  
- [ ] Convertir ce modèle en un format “IR OpenVINO” (fichiers `.xml` + `.bin`).  
- [ ] Compiler le modèle IR en un fichier `.blob` compatible avec la VPU de l’OAK-D. Cette conversion est nécessaire pour déployer le modèle sur l’OAK.
- [ ] Vérifier que la conversion s’est bien déroulée : le modèle accepte les bonnes dimensions d’entrée, les types d’entrée/sortie, etc.  

## 4. Construire le pipeline d’inférence sur OAK-D avec profondeur (spatial)  
- [ ] Initialiser les caméras de l’OAK-D : caméra RGB (couleur) + caméras mono (gauche/droite) pour la stéréo.  
- [ ] Configurer le traitement de profondeur (stéréo → disparity / depth map).  
- [ ] Charger le modèle custom (`.blob`) dans un nœud de détection spatiale approprié (par exemple un nœud de type “SpatialDetectionNetwork”, ou équivalent, selon le modèle et le framework utilisé).  
- [ ] Lier l’entrée du réseau à la sortie de la caméra couleur (image RGB pré-traitée).  
- [ ] Lier la carte de profondeur (depth) au réseau pour permettre la fusion 2D (bbox) + 3D (coordonnées spatiales).  
- [ ] Configurer la sortie : obtenir pour chaque détection la bounding box 2D **et** les coordonnées spatiales (X,Y,Z) / profondeur.  

## 5. Tester et valider sur l’OAK-D  
- [ ] Lancer le pipeline sur l’OAK-D : observer les résultats en temps réel via la sortie RGB + overlay bounding boxes + affichage de la profondeur (distance).  
- [ ] Tester dans plusieurs conditions : différentes distances, angles, éclairages, arrière-plans, positions du cube.  
- [ ] Noter les cas d’échec (faux positifs, manques de détection, profondeur incorrecte, etc.).  

## 6. Itérer / améliorer  
- [ ] Si des erreurs ou des manques apparaissent : revenir au dataset, ajouter des images couvrant les cas problématiques, annoter, relancer l’entraînement / conversion / test.  
- [ ] Ajuster les **paramètres du pipeline** si besoin : seuil de confiance, taille d’entrée, pré-traitement, scale / mean, échelle de bounding box, configuration profondeur, etc.  
- [ ] Répéter les cycles entraînement → conversion → test → collecte de données complémentaires jusqu’à obtenir une robustesse satisfaisante.
