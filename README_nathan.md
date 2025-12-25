# README de Nathan

On part sur YoloV11.

Faudra le modèle non fine-tuné pour pouvoir tester sa performance avec les jeux de tests de base.

Faudra aussi une version fine-tuné sur le cube rouge pour détecter avec le robot.

On aura plusieurs versions du modèle non fine-tuné, en fonction des optimisations qu'on applique et des architectures (luxonis, jetson...) visées.

Pour le test sur mon matériel assigné, la luxonis, faudra pouvoir utiliser le VPU de la caméra pour faire tourner le modèle sur son jeu de test.

Du coup faut récupérer le jeu de test de YoloV11.

Faut mettre au clair aussi les optims qu'on fera et les métriques qu'on mesurera.

---

## Partie A - Gazebo

- Fine-tuner YoloV11n avec le cube. On va utiliser CVAT AI.
- Tester que la caméra peut en effet détecter le cube et la profondeur.
- Demander à Noé quoi faire par rapport à Gazebo une fois que le modèle fonctionne.

## Partie B - Benchmarks

- Mettre au clair quelles métriques on va mesurer, et quelles optim on va faire.
- Aviser, quand la partie A est finie.
