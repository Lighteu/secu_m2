# Pour kyber
Dans le fichier kyber_pke.py
Pour la visualisation il faut lancer full_kyber_vis.py avec la librarie manim 
Il y a une image docker afin d'éviter d'installer manim 

# Pour dilithium
Trois versions de l'algorithme de sign  ture dilithium ont été implémenté :
- La première est dans le fichier dilithium_toy_version.py
  - Ce fichier est une implémentation d'un algorithme de version basique.
  - Cette version n'est pas fonctionnelle, dû à une erreur lors du calcul des bits de poids forts et bits de poids faibles.

- La seconde version est disponible dans le fichier dilithium_b_version.py.
  - Cette version implémente une version largement améliorée de la version précédente, en précisant comment sont calculées les bits de poids faire et faible.
  - Cette version est fonctionnelle, et peu être exécuter.

- La troisième version est disponible dans le fichier dilithium_final_version.py.
    - Cette version contient une légère amélioration de la précédente, en ajoutant des vecteurs indices (HintBits) et la compression du vecteur t en (t1,t0).
    - Cette version n'est pas fonctionnelle, dû à une erreur probablement causée lors du calcul des bits (encore une fois).
