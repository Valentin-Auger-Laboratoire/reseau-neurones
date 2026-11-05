# Un réseau de neurones écrit à la main

🇫🇷 [Français](#français) · 🇬🇧 [English](#english)

**Essayer l'outil :** https://valentin-auger-laboratoire.github.io/portfolio/outils/reseau-neurones/

---

## Français

### L'idée
Aucune bibliothèque d'apprentissage : chaque neurone, chaque multiplication, la **rétropropagation du gradient** et les **optimiseurs** sont écrits à la main en JavaScript, puis **vérifiés**. On voit le réseau apprendre en direct, on suit la rétropropagation calcul par calcul, et on l'entraîne sur de vrais chiffres manuscrits avant de lui faire lire les siens.

### Le moteur (`nn.js`)
- **Perceptron multicouche** (couches denses), de 0 à plusieurs couches cachées.
- **Activations :** ReLU, Leaky ReLU, tanh, sigmoïde, linéaire.
- **Sorties et pertes :**
  - sigmoïde + entropie croisée binaire ;
  - softmax (stable numériquement) + entropie croisée ;
  - linéaire + erreur quadratique.
- **Initialisation :** He (ReLU) ou Glorot (tanh, sigmoïde).
- **Rétropropagation** écrite à la main : à la sortie δ = ŷ − y, puis δ remonte couche par couche, multiplié par la dérivée de l'activation.
- **Optimiseurs :** descente de gradient par mini-lots, moment, RMSprop, Adam (avec correction de biais) ; régularisation L2.
- **Vérification du gradient** par différences centrées.

### Les quatre onglets
1. **Le voir apprendre**
   - 7 jeux de points 2D : cercles, XOR, lunes, spirale, 3 spirales, nuages, damier ; bruit et part de test réglables.
   - Entrées au choix : x, y, x², y², x·y, sin x, sin y.
   - Carte de décision ; schéma du réseau où **chaque neurone affiche ce qu'il calcule** (un clic l'agrandit) ; courbes de perte apprentissage / test.
   - **Expériences :**
     - XOR avec un seul neurone (impossible), avec 4 neurones, ou avec x·y en entrée ;
     - la spirale, qui demande de la profondeur ;
     - le **surapprentissage**, puis sa correction par L2 ;
     - un pas d'apprentissage trop grand, qui fait diverger.
2. **La rétropropagation pas à pas**
   - Un réseau 2-2-1, un exemple, et les quatre étapes : propagation avant, perte, rétropropagation, mise à jour.
   - Toutes les formules sont affichées **avec les nombres**.
   - Les 9 gradients sont comparés un à un aux différences finies (écart ≈ 10⁻¹¹).
3. **Chiffres manuscrits**
   - Apprentissage sur 1 438 chiffres 8×8 réels et test sur 359 chiffres jamais vus : **environ 96 %** de bonnes réponses avec un réseau 64-32-10.
   - Matrice de confusion, erreurs du réseau, poids de la première couche affichés comme des images.
   - **Dessinez un chiffre :** il passe par le même principe de prétraitement que la base (image binaire 32×32, comptage par blocs de 4×4), puis le réseau donne ses probabilités.
4. **Vérifications** : contrôles recalculés en direct dans le navigateur.

### Données
« Optical Recognition of Handwritten Digits », E. Alpaydin et C. Kaynak (1998), UCI Machine Learning Repository, licence **CC BY 4.0**. Il s'agit de la copie du jeu de test UCI fournie par scikit-learn (`load_digits`) : 1 797 chiffres. Fichier : `data/chiffres.js`.

### Fichiers
| Fichier | Rôle |
|---|---|
| `nn.js` | Moteur (testable avec Node) |
| `index.html` | Interface bilingue FR / EN |
| `data/chiffres.js` | Les 1 797 chiffres 8×8 (UCI, CC BY 4.0) |
| `tests/test.js` | 526 tests |

### Lancer les tests
```bash
node tests/test.js
```
- **Rétropropagation = différences finies :**
  - 5 activations × 3 sorties × 0, 1 ou 3 couches cachées × avec ou sans L2 ;
  - critères : erreur relative < 10⁻⁶ sur les gradients non négligeables, écart absolu < 10⁻⁸ ;
  - exemples tirés loin du point anguleux de ReLU ;
  - même vérification sur un réseau 64-12-10 avec de vrais chiffres.
- **Dérivées** des activations contre la dérivée numérique.
- **Softmax** stable, pertes exactes ; **initialisation** He et Glorot ; reproductibilité.
- **Optimiseurs :** Adam (premier pas = η × signe), moment (somme géométrique), RMSprop.
- **Descente de gradient = moindres carrés**, à 10⁻⁶ près ; pas trop grand → pas de convergence.
- **XOR :** impossible sans couche cachée, résolu avec 4 neurones ou avec x·y en entrée.
- **Spirale** apprise ; **surapprentissage** mesuré, réduit par L2.
- **Jeux de données :** classes équilibrées, règles des classes, découpage disjoint.
- **Chiffres :** plus de 99 % sur l'apprentissage, plus de 94 % sur le test, plus de 90 % sans couche cachée ; matrice de confusion cohérente.

### Limites
- Couches denses seulement (pas de convolution) ; calcul en JavaScript sur le processeur ; petits jeux de données.
- Le cadrage d'un dessin à la souris est approché : une écriture très différente de celle de 1998 peut tromper le réseau.

### Utiliser en local
Ouvrir `index.html` dans un navigateur.

---

## English

### The idea
No machine-learning library: every neuron, every multiplication, **gradient back-propagation** and the **optimisers** are written by hand in JavaScript, then **checked**.

### What is in it
- **Multilayer perceptron:**
  - ReLU, Leaky ReLU, tanh, sigmoid activations;
  - sigmoid + binary cross-entropy, softmax + cross-entropy, linear + squared error;
  - He / Glorot initialisation.
- **Training:** SGD, momentum, RMSprop and Adam optimisers, L2 regularisation.
- **Gradient check** by central differences.

### The four tabs
1. **Watch it learn:** 7 2D datasets, selectable input features, a live decision map, and a network diagram where each neuron shows its own map. Experiments: XOR, spiral, overfitting and L2, a learning rate that diverges.
2. **Step-by-step back-propagation:** a 2-2-1 network with every formula shown with its numbers. Each gradient is compared with finite differences.
3. **Handwritten digits:** trained on 1,438 real 8×8 digits, about 96 % correct on 359 unseen ones.
   - Confusion matrix, mistakes, first-layer weights.
   - Draw your own digit: it goes through the same preprocessing principle as the dataset.
4. **Checks**, recomputed live in the browser.

### Data
“Optical Recognition of Handwritten Digits”, E. Alpaydin and C. Kaynak (1998), UCI Machine Learning Repository, **CC BY 4.0** licence. This is the copy of the UCI test set distributed with scikit-learn.

### Tests
`node tests/test.js`: 526 tests.
- Back-propagation against finite differences.
- Optimisers against their formulas.
- Gradient descent against least squares.
- XOR, spiral, overfitting.
- Digit accuracy.

### Limits
- Dense layers only, CPU JavaScript, small datasets.
- The framing of a mouse drawing is approximate.

---

← [Portfolio](https://valentin-auger-laboratoire.github.io/portfolio/#projets)
