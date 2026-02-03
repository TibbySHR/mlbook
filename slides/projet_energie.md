---
marp: true
theme: mila
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Projet: Prédiction de la demande énergétique
## Consignes et méthodologie

*Pierre-Luc Bacon*
IFT3395/IFT6390 – Fondements de l'apprentissage machine

---

## Plan de la présentation

1. **Contexte et objectifs** : données Hydro-Québec, compétences visées
2. **Les données** : variables, division temporelle, décalage de distribution
3. **Implémentations** : OLS et régression logistique à partir de zéro
4. **Modélisation** : caractéristiques, Ridge, classification, modèle combiné
5. **Évaluation** : Kaggle et entrevue orale

---

<!-- _class: lead -->

# Contexte et objectifs
## Un projet ancré dans le réel

---

<!-- footer: "📖 Projet: Prédiction de la demande énergétique" -->

## Le contexte : Hydro-Québec

Les données proviennent du portail de données ouvertes d'**Hydro-Québec** :
- Consommation électrique horaire de clients
- Programme de gestion de la demande en période de pointe
- Données météorologiques intégrées

**Votre mission** : Prédire la consommation énergétique (kWh) en utilisant **uniquement** les méthodes des chapitres 1 à 5.

| Interdit | Autorisé |
|----------|----------|
| Arbres, forêts aléatoires | Régression linéaire (OLS) |
| Réseaux de neurones | Régression Ridge |
| XGBoost, LightGBM | Régression logistique |

---

## Objectifs d'apprentissage

À la fin de ce projet, vous serez en mesure de :

1. **Implémenter OLS** à partir de zéro (solution analytique)
2. **Implémenter la régression logistique** avec descente de gradient
3. **Appliquer Ridge** et interpréter l'effet de la régularisation
4. **Construire un modèle à deux étages** : classification → régression
5. **Utiliser les probabilités prédites** comme caractéristiques

Ce projet teste votre **compréhension profonde**, pas seulement l'utilisation de sklearn.

---

## Évaluation : l'oral est la clé

| Composante | Pondération | Description |
|------------|-------------|-------------|
| **Entrevue orale** | **60%** | Vérification de la compréhension |
| Code soumis | 20% | Complétion des parties 1-7 |
| Kaggle | 10% | Position au classement |
| Rapport écrit | 10% | Analyse et réflexion |

**Pourquoi 60% à l'oral?** Nous voulons évaluer votre **compréhension**, pas votre capacité à copier du code.

---

## Barème de l'entrevue orale (60%)

| Critère | Points | Ce qu'on évalue |
|---------|--------|-----------------|
| Dérivation OLS au tableau | 15 | Solution analytique sans notes |
| Explication descente de gradient | 10 | Règle de mise à jour, convergence |
| Justification des choix | 15 | Caractéristiques, TimeSeriesSplit |
| Questions théoriques | 10 | Ridge = MAP, entropie croisée |
| Modifications en direct | 10 | Changer λ, prédire l'effet |

**Vous devez pouvoir expliquer chaque ligne de code que vous soumettez.**

---

<!-- _class: lead -->

# Les données
## Division temporelle et décalage de distribution

---

<!-- footer: "📖 Chapitre 4 : Généralisation" -->

## Variables disponibles

**Météorologiques** : `temperature_ext`, `humidite`, `vitesse_vent`, `neige`, `irradiance_solaire`

**Temporelles** : `heure`, `mois`, `jour`, `jour_semaine`
- Encodage cyclique : `heure_sin`, `heure_cos`, `mois_sin`, `mois_cos`
- Indicateurs : `est_weekend`, `est_ferie`

**Variable importante** :
$$\boxed{\texttt{clients\_connectes} : \text{nombre de clients connectés}}$$

**Variable cible** : `energie_kwh` (consommation en kWh)

---

## Pourquoi encoder l'heure avec sin/cos?

Avec l'heure brute (0-23), un modèle linéaire a deux problèmes :

1. Il ne peut apprendre qu'une relation **monotone** (la cible augmente ou diminue avec l'heure)
2. Les heures 23 et 0 sont **adjacentes** mais numériquement distantes de 23

Or, la consommation d'énergie est **cyclique** : haute le matin et le soir, basse la nuit.

| Encodage | Distance 22h→23h | Distance 23h→0h |
|----------|------------------|-----------------|
| Brut : $h$ | 1 | 23 |
| Cyclique : $(\sin, \cos)$ | 0.26 | 0.26 |

Un modèle linéaire avec l'heure brute ne peut pas capturer ce comportement.

---

## Encodage cyclique : la solution

Projeter l'heure sur un cercle avec sin et cos :

$$\boxed{\phi(h) = \left(\sin\frac{2\pi h}{24}, \cos\frac{2\pi h}{24}\right)}$$

La distance entre heures consécutives est maintenant la distance euclidienne dans le plan :

$$d(h_i, h_{i+1}) = \sqrt{(\sin_{i+1} - \sin_i)^2 + (\cos_{i+1} - \cos_i)^2}$$

| Variable | Période $T$ | Formule |
|----------|-------------|---------|
| Heure | 24 | $\sin(2\pi h/24), \cos(2\pi h/24)$ |
| Jour semaine | 7 | $\sin(2\pi j/7), \cos(2\pi j/7)$ |
| Mois | 12 | $\sin(2\pi m/12), \cos(2\pi m/12)$ |

Deux composantes nécessaires : $\sin$ seul ne distingue pas 6h de 18h.

---

## Division temporelle — Ne pas mélanger!

C'est une **série temporelle**. Une division aléatoire cause une **fuite d'information**.

```
Janvier 2022                                    Juillet 2024
    |==================== Train ====================|==== Test ====|
                                                 1er fév 2024
```

| Interdit | Correct |
|----------|---------|
| `train_test_split(random_state=42)` | Division chronologique |
| `KFold` standard | `TimeSeriesSplit` |

Le futur ne peut pas prédire le passé!

---

## Pourquoi `TimeSeriesSplit`?

La validation croisée standard mélange passé et futur :

```
KFold standard (INTERDIT):
Fold 1: [Test] [Train] [Train] [Train] [Train]  ← Utilise le futur!
Fold 2: [Train] [Test] [Train] [Train] [Train]  ← Utilise le futur!
```

`TimeSeriesSplit` respecte l'ordre chronologique :

```
TimeSeriesSplit (CORRECT):
Fold 1: [Train] [Test]
Fold 2: [Train] [Train] [Test]
Fold 3: [Train] [Train] [Train] [Test]
```

```python
from sklearn.model_selection import TimeSeriesSplit
model = RidgeCV(alphas=[0.1, 1, 10], cv=TimeSeriesSplit(n_splits=5))
```

---

## Décalage de distribution

Les données d'entraînement et de test couvrent des **saisons différentes**.

| Ensemble | Période | Consommation moyenne |
|----------|---------|---------------------|
| Train | Janvier 2022 – Janvier 2024 | **216 kWh** (hiver) |
| Test | Février 2024 – Juillet 2024 | **84 kWh** (printemps/été) |

**Conséquences** :
- Le R² sur le test peut être négatif
- C'est un défi **réaliste** : généraliser à travers les saisons
- Utilisez des caractéristiques qui **généralisent** (ex: `clients_connectes`)

Ne paniquez pas si votre R² est bas. Concentrez-vous sur la **méthodologie**.

---

<!-- _class: lead -->

# Partie 1 : Implémentation OLS
## La solution analytique des moindres carrés

---

<!-- footer: "📖 Chapitre 2 : Régression linéaire" -->

## Rappel : la solution OLS

Minimiser la somme des carrés des résidus :

$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$$

**Solution analytique** (équations normales) :

$$\boxed{\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}}$$

En pratique, **ne pas inverser directement**. Résoudre le système :

$$\mathbf{X}^\top \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^\top \mathbf{y}$$

---

## Ce que vous devez implémenter

Deux fonctions à écrire :

```python
def ols_fit(X, y):
    """Retourne beta de forme (p+1,) avec intercept en premier."""
    # À vous de jouer!

def ols_predict(X, beta):
    """Retourne les prédictions."""
    # À vous de jouer!
```

**Indices** :
- N'oubliez pas d'ajouter une colonne de 1 pour l'intercept
- Utilisez `np.linalg.solve` plutôt que d'inverser directement
- Vérifiez vos résultats avec `sklearn.linear_model.LinearRegression`

---

<!-- _class: lead -->

# Partie 2 : Régression logistique
## Descente de gradient

---

<!-- footer: "📖 Chapitre 3 : Classification" -->

## Rappels : régression logistique

**Fonction sigmoïde** :
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Perte d'entropie croisée binaire** :
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

**Gradient** :
$$\boxed{\nabla \mathcal{L} = \frac{1}{n} \mathbf{X}^\top (\boldsymbol{\sigma} - \mathbf{y})}$$

où $\boldsymbol{\sigma} = \sigma(\mathbf{X}\boldsymbol{\beta})$.

---

## Stabilité numérique : pièges à éviter

Deux problèmes classiques que vous devrez gérer :

| Problème | Cause | Conséquence |
|----------|-------|-------------|
| $e^{-z}$ déborde | $\|z\|$ très grand | `inf` ou `nan` |
| $\log(0)$ | probabilité = 0 ou 1 | `-inf` |

**Indice** : `np.clip` est votre ami pour borner les valeurs.

---

## Ce que vous devez implémenter

```python
def sigmoid(z):
    """Fonction sigmoïde, stable numériquement."""

def cross_entropy_loss(y_true, y_pred_proba):
    """Perte d'entropie croisée binaire."""

def logistic_gradient(X, y, beta):
    """Gradient de la perte."""

def logistic_fit_gd(X, y, lr=0.1, n_iter=1000):
    """Entraînement par descente de gradient."""
```

**Conseil** : Normalisez les caractéristiques avec `StandardScaler`.

---

## Vérifier la convergence

Tracez la perte en fonction des itérations :

```python
plt.plot(losses)
plt.xlabel('Itération')
plt.ylabel('Perte (entropie croisée)')
```

**Questions à vous poser** :

| Observation | Action |
|-------------|--------|
| Perte diminue et se stabilise | Convergence OK |
| Perte oscille | Réduire le taux d'apprentissage |
| Perte ne diminue pas | Vérifier le gradient |

---

<!-- _class: lead -->

# Parties 3-6 : Modélisation
## Caractéristiques, Ridge, classification

---

<!-- footer: "📖 Chapitres 2-5" -->

## Partie 3 : Ingénierie des caractéristiques

**À partir de maintenant, vous pouvez utiliser scikit-learn.**

Créez **au moins 3 nouvelles caractéristiques** :

| Type | Exemple |
|------|---------|
| Retards | `df['energie_kwh'].shift(1)`, `shift(24)` |
| Moyennes mobiles | `df['energie_kwh'].rolling(6).mean()` |
| Interactions | `df['temperature_ext'] * df['heure_cos']` |
| Transformations | `np.maximum(18 - df['temperature_ext'], 0)` |

Les retards créent des `NaN` — supprimez-les avec `dropna()`.

---

## Partie 4 : Régression Ridge

Ridge ajoute une pénalité L2 :

$$\boxed{\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|^2}$$

**Ce qu'on vous demande** :
1. Entraîner Ridge avec `RidgeCV` et `TimeSeriesSplit`
2. Comparer avec OLS
3. Analyser les coefficients

**Question orale** : Pourquoi Ridge aide-t-il avec des caractéristiques corrélées?

---

## Partie 5 : Classification des événements de pointe

Entraînez un classifieur pour `evenement_pointe`, puis utilisez **la probabilité** :

```python
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train_pointe)

# Extraire P(pointe), pas la prédiction binaire!
train_eng['P_pointe'] = clf.predict_proba(X_train)[:, 1]
```

**Question clé** : Pourquoi `P(pointe)` plutôt qu'un indicateur 0/1?

→ La probabilité contient plus d'information. P=0,7 ≠ P=0,99.

---

## Partie 6 : Modèle combiné

Assemblez tout :

```python
features_final = features_disponibles + ['P_pointe']

model_final = RidgeCV(
    alphas=[0.1, 1, 10, 100], 
    cv=TimeSeriesSplit(n_splits=5)
)
model_final.fit(X_train_final, y_train_final)
```

**Mesurez l'amélioration** apportée par `P_pointe`.

---

## Partie 7 : Extension

Choisissez **UNE** option :

| Option | Description |
|--------|-------------|
| **A** | Ajouter des données météo externes (`meteostat`) |
| **B** | Classification multiclasse (faible/moyenne/élevée) |
| **C** | Analyse d'erreur : quand le modèle échoue-t-il? |

Cette partie montre votre **initiative**.

---

<!-- _class: lead -->

# Compétition Kaggle
## Soumission et évaluation

---

<!-- footer: "📖 Projet: Prédiction de la demande énergétique" -->

## La compétition Kaggle

**Métrique** : RMSE (Root Mean Squared Error)

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Format de soumission** :
```
id,energie_kwh
0,245.5
1,312.8
...
```

**Limites** : 5 soumissions par jour

---

## Générer la soumission

Trois fichiers de données sont fournis :

| Fichier | Contient `energie_kwh`? | Usage |
|---------|-------------------------|-------|
| `energy_train.csv` | Oui | Entraînement |
| `energy_test_avec_cible.csv` | Oui | Évaluation locale |
| `energy_test.csv` | **Non** | Soumission Kaggle |

Le fichier pour Kaggle ne contient pas la cible : c'est ce que vous devez prédire!

**Format de soumission** (`submission.csv`) :

```
id,energie_kwh
0,123.45
1,98.76
...
```

Appliquez les mêmes transformations qu'à l'entraînement avant de prédire.

---

<!-- _class: lead -->

# Entrevue orale
## 60% de la note finale

---

<!-- footer: "⚠️ L'entrevue orale vaut 60% de la note" -->

## Pourquoi l'oral est-il si important?

L'entrevue orale est la **composante principale** de l'évaluation.

| Raison | Explication |
|--------|-------------|
| **Vérification** | S'assurer que vous comprenez votre code |
| **Compréhension profonde** | Au-delà de la simple exécution |
| **Compétence professionnelle** | Savoir expliquer son travail |

**Ce qu'on évalue** : Votre capacité à dériver, expliquer et modifier.

**Ce qu'on n'évalue pas** : La perfection du code ou le rang Kaggle.

---

## Barème détaillé de l'oral (60 points)

| Critère | Points | Attentes |
|---------|--------|----------|
| Dérivation OLS au tableau | 15 | Sans notes, proprement |
| Descente de gradient | 10 | Expliquer chaque terme |
| Justification des choix | 15 | Pourquoi TimeSeriesSplit? Quelles caractéristiques? |
| Questions théoriques | 10 | Ridge = MAP, entropie croisée |
| Modifications en direct | 10 | Changer λ, prédire l'effet |

**Durée** : ~15 minutes par étudiant.

---

## Questions sur les fondamentaux

1. **Dérivez la solution OLS** sur le tableau.
   - Partir de $\min \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$
   - Calculer le gradient, l'annuler

2. **Pourquoi une division temporelle?**
   - Éviter la fuite d'information

3. **Que voyez-vous dans vos résidus?**
   - Centrés autour de 0?
   - Motifs (hétéroscédasticité)?

---

## Questions sur la régularisation

4. **Pourquoi Ridge aide-t-il avec des caractéristiques corrélées?**
   - Instabilité des estimateurs OLS
   - Ridge « partage » le poids

5. **Comment avez-vous choisi λ?**
   - Validation croisée avec `TimeSeriesSplit`

6. **Quel coefficient a été le plus réduit? Pourquoi?**
   - Les coefficients instables sont réduits

---

## Questions sur la classification et la théorie

7. **Votre classifieur donne P=0,7. Qu'est-ce que cela signifie?**
   - 70% de probabilité, pas une certitude

8. **Pourquoi P(pointe) plutôt qu'un indicateur 0/1?**
   - Plus d'information (incertitude préservée)

9. **Expliquez Ridge comme estimation MAP.**
   - Prior gaussien sur les coefficients
   - $\lambda$ contrôle la variance du prior

---

## Questions de synthèse

10. **Parcourez votre modèle complet étape par étape.**
    - Chargement → Caractéristiques → Classification → Régression

11. **Quelle amélioration était la plus importante?**
    - Comparer les différentes étapes

12. **Modifiez ce seuil en direct — que prédisez-vous?**
    - Montrer la compréhension des hyperparamètres

---

## Récapitulatif

| À faire | À ne pas faire |
|---------|----------------|
| Implémenter OLS et logistique à la main | Utiliser seulement sklearn |
| Utiliser `TimeSeriesSplit` | Division aléatoire |
| Comprendre le décalage de distribution | Paniquer si R² est négatif |
| **Préparer l'entrevue orale** | **Copier sans comprendre** |

---

## Avertissement sur l'utilisation d'outils IA

Les outils comme ChatGPT, Cursor, Copilot peuvent vous aider, **mais** :

- Vous devez comprendre **chaque ligne** de code
- L'entrevue orale révélera si vous comprenez ou non
- **60% de la note** dépend de votre capacité à expliquer

$$\boxed{\text{Code copié sans compréhension} \Rightarrow \text{échec à l'oral}}$$

**Conseil** : Utilisez ces outils pour apprendre, pas pour éviter d'apprendre.

---

<!-- _class: lead -->

# Questions?

**Ressources** :
- Notebook : `exercises/projet_energie.ipynb`
- Compétition Kaggle : lien dans le notebook
- Chapitres 1-5 du livre

**Rappel** : L'entrevue orale vaut **60%** de la note finale.
