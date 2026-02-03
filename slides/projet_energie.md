---
marp: true
theme: mila
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Projet: Prédiction de la demande énergétique
## Présentation et consignes

*Pierre-Luc Bacon*
IFT3395/IFT6390 – Fondements de l'apprentissage machine

---

## Plan de la présentation

1. **Contexte et objectifs** : Données Hydro-Québec et compétences visées
2. **Structure et évaluation** : Pondération et entrevue orale
3. **Les données** : Variables, division temporelle, décalage de distribution
4. **Partie 1-2** : Implémentations OLS et régression logistique
5. **Partie 3-6** : Caractéristiques, Ridge, classification, modèle combiné
6. **Kaggle et entrevue** : Soumission et préparation

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

**Votre mission** : Prédire la consommation énergétique (en kWh) en utilisant **uniquement** les méthodes vues dans les chapitres 1 à 5.

| Interdit | Autorisé |
|----------|----------|
| Arbres, forêts aléatoires | Régression linéaire (OLS) |
| Réseaux de neurones | Régression Ridge |
| XGBoost, LightGBM | Régression logistique |
| Tout ce qui n'est pas ch. 1-5 | Descente de gradient |

---

## Objectifs d'apprentissage

À la fin de ce projet, vous serez en mesure de :

1. **Implémenter OLS** à partir de zéro (solution analytique)
2. **Implémenter la régression logistique** avec descente de gradient
3. **Appliquer Ridge** et interpréter l'effet de la régularisation
4. **Construire un modèle à deux étages** : classification → régression
5. **Utiliser les probabilités prédites** comme caractéristiques

Ce projet teste votre **compréhension profonde**, pas juste l'utilisation de scikit-learn.

---

<!-- _class: lead -->

# Structure et évaluation
## Comment vous serez notés

---

## Pondération du projet

| Partie | Contenu | Poids |
|--------|---------|-------|
| 1 | Implémentation OLS | 10% |
| 2 | Régression logistique + descente de gradient | 15% |
| 3 | Ingénierie des caractéristiques | 15% |
| 4 | Régression Ridge | 15% |
| 5 | Sous-tâche de classification | 15% |
| 6 | Modèle combiné | 10% |
| 7 | Extension (choisir 1 option) | 10% |
| **Kaggle** | Position au classement | **10%** |

⚠️ L'**entrevue orale** peut ajuster votre note de **±10%**.

---

## L'entrevue orale : ce qui compte vraiment

L'entrevue n'est pas un interrogatoire, c'est une **conversation** sur votre travail.

**Ce qu'on évalue** :
- Votre **compréhension** des méthodes utilisées
- Votre capacité à **expliquer** vos choix
- Votre capacité à **dériver** les formules de base

**Exemples de questions** :
- « Dérivez la solution OLS sur le tableau. »
- « Pourquoi Ridge aide-t-il avec des caractéristiques corrélées? »
- « Votre classifieur donne P=0,7. Qu'est-ce que cela signifie? »

On ne vous piégera pas. Préparez-vous simplement à **expliquer votre code**.

---

<!-- _class: lead -->

# Les données
## Division temporelle et décalage de distribution

---

## Description des variables

**Variables météorologiques** :
- `temperature_ext` : Température extérieure (°C)
- `humidite`, `vitesse_vent`, `neige`, `irradiance_solaire`

**Variables temporelles** :
- `heure`, `mois`, `jour`, `jour_semaine`
- `heure_sin`, `heure_cos`, `mois_sin`, `mois_cos` (encodage cyclique)
- `est_weekend`, `est_ferie`

**Variable importante** ⚠️ :
- `clients_connectes` : Nombre de clients connectés (très prédictif!)

**Variable cible** : `energie_kwh`

---

## ⚠️ Division temporelle — Ne pas mélanger!

C'est une **série temporelle**. Une division aléatoire cause une **fuite d'information**.

```
Janvier 2022                                    Juillet 2024
    |==================== Train ====================|==== Test ====|
                                                 1er fév 2024
```

**Interdit** :
- `train_test_split(X, y, random_state=42)` ❌
- Validation croisée standard (`KFold`) ❌

**Correct** :
- Division chronologique ✅
- `TimeSeriesSplit` pour la validation croisée ✅

---

## ⚠️ Décalage de distribution

Les données d'entraînement et de test couvrent des **saisons différentes**.

| Ensemble | Période | Consommation moyenne |
|----------|---------|---------------------|
| Train | Janvier 2022 – Janvier 2024 | **216 kWh** (hiver) |
| Test | Février 2024 – Juillet 2024 | **84 kWh** (printemps/été) |

**Conséquences** :
- Le R² sur le test peut être négatif (le modèle fait pire que la moyenne)
- C'est un défi **réaliste** : le modèle doit généraliser à travers les saisons
- Utilisez des caractéristiques qui **généralisent** (ex: `clients_connectes`)

Ne paniquez pas si votre R² est bas. Concentrez-vous sur la **méthodologie**.

---

## Pourquoi `TimeSeriesSplit`?

La validation croisée standard mélange passé et futur :

```
Standard KFold (INTERDIT pour séries temporelles):
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
tscv = TimeSeriesSplit(n_splits=5)
model = RidgeCV(alphas=[0.1, 1, 10, 100], cv=tscv)
```

---

<!-- _class: lead -->

# Partie 1 : Implémentation OLS
## La solution analytique des moindres carrés

---

## Rappel : la solution OLS

Minimiser la somme des carrés des résidus :

$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$$

**Solution analytique** (équations normales) :

$$\boxed{\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}}$$

En pratique, **ne pas inverser directement** — résoudre le système :

$$\mathbf{X}^\top \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^\top \mathbf{y}$$

```python
# Stable numériquement
beta = np.linalg.solve(X.T @ X, X.T @ y)
```

---

## Ce qu'on vous demande

Implémenter deux fonctions :

```python
def ols_fit(X, y):
    """
    X : (n, p) - caractéristiques SANS colonne de 1
    y : (n,) - cible
    Retourne: beta de forme (p+1,) avec intercept en premier
    """
    # 1. Ajouter une colonne de 1 pour l'intercept
    # 2. Résoudre X^T X beta = X^T y
    pass

def ols_predict(X, beta):
    """Retourne X_aug @ beta"""
    pass
```

**Vérification** : Vos coefficients doivent correspondre à `LinearRegression()`.

---

<!-- _class: lead -->

# Partie 2 : Régression logistique
## Implémentation avec descente de gradient

---

## Rappels : régression logistique

**Fonction sigmoïde** :
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Perte d'entropie croisée binaire** :
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

où $p_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})$.

**Gradient** :
$$\nabla \mathcal{L} = \frac{1}{n} \mathbf{X}^\top (\boldsymbol{\sigma} - \mathbf{y})$$

---

## Stabilité numérique — Attention!

**Problème** : $e^{-z}$ déborde pour $z$ très négatif ou très positif.

```python
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Éviter le débordement
    return 1 / (1 + np.exp(-z))
```

**Problème** : $\log(0) = -\infty$

```python
def cross_entropy_loss(y_true, y_pred_proba):
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred_proba) + 
                    (1 - y_true) * np.log(1 - y_pred_proba))
```

---

## Descente de gradient

```python
def logistic_fit_gd(X, y, lr=0.1, n_iter=1000):
    n, p = X.shape
    X_aug = np.column_stack([np.ones(n), X])  # Ajouter intercept
    beta = np.zeros(p + 1)
    losses = []
    
    for i in range(n_iter):
        proba = sigmoid(X_aug @ beta)
        loss = cross_entropy_loss(y, proba)
        losses.append(loss)
        
        grad = (X_aug.T @ (proba - y)) / n
        beta = beta - lr * grad
    
    return beta, losses
```

**Conseil** : Normalisez les caractéristiques avec `StandardScaler` avant!

---

## Courbe de convergence

Tracez la perte en fonction des itérations pour vérifier la convergence :

```python
plt.plot(losses)
plt.xlabel('Itération')
plt.ylabel('Perte (entropie croisée)')
plt.title('Convergence de la descente de gradient')
```

**Questions à vous poser** :
- La perte diminue-t-elle?
- Converge-t-elle vers un plateau?
- Si elle oscille : réduisez le taux d'apprentissage

---

<!-- _class: lead -->

# Parties 3-6 : Le reste du projet
## Caractéristiques, Ridge, classification, modèle combiné

---

## Partie 3 : Ingénierie des caractéristiques

**À partir de maintenant, vous pouvez utiliser scikit-learn.**

Créez **au moins 3 nouvelles caractéristiques** :

| Type | Exemple |
|------|---------|
| Retards (lags) | `df['energie_kwh'].shift(1)`, `shift(24)` |
| Moyennes mobiles | `df['energie_kwh'].rolling(6).mean()` |
| Interactions | `df['temperature_ext'] * df['heure_cos']` |
| Transformations | `np.maximum(18 - df['temperature_ext'], 0)` |

⚠️ Les retards créent des `NaN` — supprimez-les avec `dropna()`.

---

## Partie 4 : Régression Ridge

Ridge ajoute une pénalité L2 aux coefficients :

$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|^2$$

**Ce qu'on vous demande** :
1. Entraîner Ridge avec `RidgeCV` et `TimeSeriesSplit`
2. Comparer les performances avec OLS
3. Analyser comment les coefficients changent

**Questions pour l'oral** :
- Pourquoi Ridge aide-t-il avec des caractéristiques corrélées?
- Comment interpréter Ridge comme estimation MAP?

---

## Partie 5 : Classification des événements de pointe

Entraînez un classifieur pour prédire `evenement_pointe`, puis utilisez **la probabilité** comme caractéristique.

```python
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train_pointe)

# Extraire P(pointe), pas la prédiction binaire!
train_eng['P_pointe'] = clf.predict_proba(X_train)[:, 1]
```

**Question clé** : Pourquoi `P(pointe)` plutôt qu'un indicateur 0/1?

→ La probabilité contient plus d'information. Un P=0,7 n'est pas traité comme un P=0,99.

---

## Partie 6 : Modèle combiné

Assemblez tout :
1. Caractéristiques de base + vos nouvelles caractéristiques
2. Ajoutez `P_pointe` comme caractéristique supplémentaire
3. Entraînez Ridge final

```python
features_final = features_disponibles + ['P_pointe']
model_final = RidgeCV(alphas=[0.1, 1, 10, 100], cv=tscv)
model_final.fit(X_train_final, y_train_final)
```

**Mesurez l'amélioration** apportée par P_pointe.

---

## Partie 7 : Extension (choisir UNE option)

| Option | Description |
|--------|-------------|
| **A** | Ajouter des données météo externes (`meteostat`) |
| **B** | Classification multiclasse (faible/moyenne/élevée) avec softmax |
| **C** | Analyse d'erreur approfondie : quand le modèle échoue-t-il? |

Cette partie montre votre **initiative** et votre **curiosité**.

---

<!-- _class: lead -->

# Compétition Kaggle
## Soumission et classement

---

## La compétition Kaggle

**Lien** : Disponible dans le notebook (invitation privée)

**Métrique** : RMSE (Root Mean Squared Error)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Format de soumission** :
```
id,energie_kwh
0,245.5
1,312.8
2,189.3
...
```

**Limites** : 5 soumissions par jour

---

## Conseil pour la soumission

Le fichier `test_kaggle` n'a **pas** la colonne `energie_kwh`.

```python
# Charger le test Kaggle (sans cible)
test_kaggle = pd.read_csv(BASE_URL + "energy_test.csv")

# Appliquer les mêmes transformations
test_kaggle_eng = creer_caracteristiques(test_kaggle)
test_kaggle_eng['P_pointe'] = clf.predict_proba(X_kaggle_pointe)[:, 1]

# Prédire
X_kaggle = test_kaggle_eng[features_final].values
y_pred_kaggle = model_final.predict(X_kaggle)

# Créer la soumission
submission = pd.DataFrame({
    'id': range(len(test_kaggle)),
    'energie_kwh': y_pred_kaggle
})
submission.to_csv('submission.csv', index=False)
```

---

<!-- _class: lead -->

# Préparation à l'entrevue orale
## Les questions auxquelles vous devez pouvoir répondre

---

## Questions sur les fondamentaux

1. **Dérivez la solution OLS** sur le tableau.
   - Partir de $\min \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$
   - Calculer le gradient, l'annuler

2. **Pourquoi une division temporelle?**
   - Éviter la fuite d'information
   - Le futur ne peut pas prédire le passé

3. **Que voyez-vous dans vos résidus?**
   - Sont-ils centrés autour de 0?
   - Y a-t-il des motifs (hétéroscédasticité)?

---

## Questions sur la régularisation

4. **Pourquoi Ridge aide-t-il avec des caractéristiques corrélées?**
   - Instabilité des estimateurs OLS
   - Ridge « partage » le poids entre caractéristiques corrélées

5. **Comment avez-vous choisi λ?**
   - Validation croisée (`RidgeCV`)
   - Avec `TimeSeriesSplit`

6. **Quel coefficient a été le plus réduit? Pourquoi?**
   - Analyser le tableau de comparaison OLS vs Ridge
   - Les coefficients instables sont les plus réduits

---

## Questions sur la classification

7. **Votre classifieur donne P=0,7. Qu'est-ce que cela signifie?**
   - 70% de probabilité d'événement de pointe
   - Pas une certitude!

8. **Pourquoi utiliser P(pointe) plutôt qu'un indicateur 0/1?**
   - Plus d'information (incertitude préservée)
   - Permet des décisions graduées

9. **Expliquez Ridge comme estimation MAP.**
   - Prior gaussien sur les coefficients
   - λ contrôle la variance du prior

---

## Questions de synthèse

10. **Parcourez votre modèle complet étape par étape.**
    - Chargement → Caractéristiques → Classification → Régression

11. **Quelle amélioration de R² était la plus importante?**
    - Comparer les différentes étapes
    - Identifier la contribution de chaque partie

12. **Modifiez ce seuil en direct — que prédisez-vous?**
    - Montrer que vous comprenez l'impact des hyperparamètres

---

## Récapitulatif

| À faire | À ne pas faire |
|---------|----------------|
| Implémenter OLS et logistique à la main | Utiliser seulement sklearn |
| Utiliser `TimeSeriesSplit` | Utiliser `train_test_split` aléatoire |
| Comprendre le décalage de distribution | Paniquer si R² est négatif |
| Préparer l'entrevue orale | Copier du code sans comprendre |
| Documenter vos choix | Soumettre sans vérifier le format |

**Bon projet!** 🎯

---

<!-- _class: lead -->

# Questions?
