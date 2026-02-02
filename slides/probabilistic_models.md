---
marp: true
theme: mila
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Modèles probabilistes
## Théorie de l'information, Naive Bayes et GMM

*Pierre-Luc Bacon*
IFT6390 – Fondements de l'apprentissage machine

---

## Plan de la présentation

1. **Stabilité numérique** : l'astuce log-sum-exp
2. **Théorie de l'information** : entropie, divergence KL, entropie croisée
3. **Classifieur naïf bayésien** : approche générative pour la classification
4. **Modèles de mélange gaussien** : partitionnement probabiliste et algorithme EM

---

<!-- _class: lead -->

# Stabilité numérique
## L'astuce log-sum-exp

---

<!-- footer: "📖 Chapitre 3 : Classification" -->

## Le problème : débordement numérique

La fonction **softmax** transforme des scores en probabilités :

$$\text{softmax}(\mathbf{a})_c = \frac{e^{a_c}}{\sum_{c'} e^{a_{c'}}}$$

**Problème** : Si les logits sont grands, l'exponentielle déborde.

| Logit $a_c$ | $e^{a_c}$ | Résultat |
|-------------|-----------|----------|
| 10 | 22 026 | OK |
| 100 | $2.7 \times 10^{43}$ | OK |
| 1000 | $\infty$ | **Débordement!** |

En Python, `np.exp(1000)` retourne `inf`, rendant le calcul inutilisable.

---

## L'astuce log-sum-exp

**Idée** : Soustraire le maximum avant d'exponencier.

$$\text{softmax}(\mathbf{a})_c = \frac{e^{a_c - a_{\max}}}{\sum_{c'} e^{a_{c'} - a_{\max}}}$$

où $a_{\max} = \max_c a_c$.

**Pourquoi ça fonctionne?** Le facteur $e^{-a_{\max}}$ apparaît au numérateur et au dénominateur :

$$\frac{e^{a_c - a_{\max}}}{\sum_{c'} e^{a_{c'} - a_{\max}}} = \frac{e^{a_c} \cdot e^{-a_{\max}}}{\sum_{c'} e^{a_{c'}} \cdot e^{-a_{\max}}} = \frac{e^{a_c}}{\sum_{c'} e^{a_{c'}}}$$

Le résultat est identique, mais le plus grand exposant vaut maintenant **0**.

---

## Implémentation stable

```python
def softmax_naive(a):
    """Version naïve - déborde pour grands logits"""
    return np.exp(a) / np.sum(np.exp(a))

def softmax_stable(a):
    """Version stable - soustrait le max"""
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a)
```

| Entrée | `softmax_naive` | `softmax_stable` |
|--------|-----------------|------------------|
| `[1, 2, 3]` | `[0.09, 0.24, 0.67]` | `[0.09, 0.24, 0.67]` |
| `[1000, 1001, 1002]` | `[nan, nan, nan]` | `[0.09, 0.24, 0.67]` |

En pratique, utilisez `scipy.special.softmax` qui gère automatiquement la stabilité.

---

## Visualisation du softmax

![w:850](_static/softmax_transform.png)

Le softmax transforme des scores arbitraires en probabilités valides ($\sum_c \mu_c = 1$, $\mu_c > 0$).

---

## La fonction log-sum-exp

Pour calculer $\log \sum_c e^{a_c}$ de manière stable :

$$\boxed{\text{logsumexp}(\mathbf{a}) = a_{\max} + \log \sum_{c} e^{a_c - a_{\max}}}$$

**Applications** : softmax, entropie croisée, GMM, responsabilités EM.

---

## Entraînement : logsumexp suffit!

L'entropie croisée pour la vraie classe $y$ est $-\log \text{softmax}(\mathbf{a})_y$. Développons :

$$-\log \frac{e^{a_y}}{\sum_c e^{a_c}} = -a_y + \log \sum_c e^{a_c} = \boxed{-a_y + \text{logsumexp}(\mathbf{a})}$$

**Pas besoin de calculer le softmax!** La perte se calcule directement à partir des logits.

```python
def cross_entropy_stable(logits, y):
    """Entropie croisée sans calculer softmax"""
    return -logits[y] + logsumexp(logits)
```

C'est ce que fait `torch.nn.CrossEntropyLoss` : il prend des **logits**, pas des probabilités.

---

<!-- _class: lead -->

# Théorie de l'information
## Entropie, divergence KL et entropie croisée

---

<!-- footer: "📖 Chapitre 5 : Le cadre probabiliste" -->

## L'entropie : mesurer l'incertitude

Considérons une pièce de monnaie :
- Pièce équilibrée ($p = 0{,}5$) : incertitude **maximale**
- Pièce truquée ($p = 0{,}99$) : incertitude **faible**

L'**entropie** quantifie cette incertitude :

$$\mathbb{H}(p) = -\sum_y p(y) \log p(y)$$

| Distribution | Entropie | Interprétation |
|--------------|----------|----------------|
| $p = [0{,}5, 0{,}5]$ | 1 bit | Incertitude maximale |
| $p = [0{,}99, 0{,}01]$ | 0,08 bits | Presque certain |
| $p = [1, 0]$ | 0 bits | Déterministe |

L'entropie est maximale pour la distribution **uniforme**.

---

## Pourquoi des bits? La base du logarithme

L'entropie = **nombre minimal de questions oui/non** pour identifier un résultat.

- Pièce équilibrée : 1 question → 1 bit
- Dé à 6 faces : $\log_2 6 \approx 2{,}58$ questions

| Base | Unité | Usage |
|------|-------|-------|
| $\log_2$ | **bits** | Théorie de l'information, compression |
| $\ln$ | **nats** | Apprentissage automatique (gradients) |

**En ML** : on utilise $\ln$ (dérivée simple), mais l'interprétation reste la même.

Conversion : $1 \text{ nat} = \frac{1}{\ln 2} \approx 1{,}44 \text{ bits}$

---

## Entropie de Bernoulli

Pour une variable binaire avec $p(Y=1) = \theta$ :

$$\mathbb{H}(\theta) = -\theta \log_2 \theta - (1-\theta) \log_2 (1-\theta)$$

![w:650](_static/entropy_bernoulli.png)

L'entropie est **symétrique** autour de $\theta = 0{,}5$ et atteint son maximum (1 bit) quand les deux résultats sont équiprobables.

---

## Entropie croisée

Supposons que les données suivent $p$, mais nous utilisons $q$ pour prédire.

L'**entropie croisée** mesure la surprise moyenne :

$$\mathbb{H}_{\text{ce}}(p, q) = -\sum_y p(y) \log q(y)$$

| Relation | Signification |
|----------|---------------|
| $q = p$ | $\mathbb{H}_{\text{ce}}(p, q) = \mathbb{H}(p)$ |
| $q \neq p$ | $\mathbb{H}_{\text{ce}}(p, q) > \mathbb{H}(p)$ |

Utiliser le « mauvais » modèle augmente toujours la surprise moyenne.

---

## Divergence de Kullback-Leibler

La **divergence KL** mesure la différence entre deux distributions :

$$D_{\text{KL}}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)} = \mathbb{E}_{p}\left[\log \frac{p(y)}{q(y)}\right]$$

**Propriétés** :
- $D_{\text{KL}}(p \| q) \geq 0$ toujours (inégalité de Gibbs)
- $D_{\text{KL}}(p \| q) = 0$ si et seulement si $p = q$
- **Non symétrique** : $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$

L'asymétrie a du sens : la surprise de quelqu'un qui croit en $q$ mais observe $p$ diffère de l'inverse.

---

## Décomposition fondamentale

$$\boxed{D_{\text{KL}}(p \| q) = \mathbb{H}_{\text{ce}}(p, q) - \mathbb{H}(p)}$$

![w:900](_static/kl_divergence.png)

En apprentissage, nous ne pouvons pas réduire $\mathbb{H}(p)$ (irréductible), mais nous pouvons minimiser $D_{\text{KL}}$ en améliorant notre modèle.

---

## L'EMV minimise la divergence KL

Soit $\hat{p}$ la distribution empirique des données (fréquences observées).

Minimiser la **log-vraisemblance négative** :
$$\text{LVN}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^N \log p(y_i | \mathbf{x}_i; \boldsymbol{\theta})$$

équivaut à minimiser l'**entropie croisée** $\mathbb{H}_{\text{ce}}(\hat{p}, p_{\boldsymbol{\theta}})$.

Puisque $\mathbb{H}(\hat{p})$ est constante, cela revient à minimiser :
$$D_{\text{KL}}(\hat{p} \| p_{\boldsymbol{\theta}})$$

**Le maximum de vraisemblance cherche le modèle le plus « proche » des données au sens de la divergence KL.**

---

## Trois perspectives unifiées

Le même algorithme, trois interprétations :

| Perspective | Objectif | Résultat |
|-------------|----------|----------|
| **Décisionnelle** | Minimiser le risque empirique | Fonction de perte |
| **Probabiliste** | Maximiser la vraisemblance | Modèle génératif |
| **Informationnelle** | Minimiser la divergence KL | Distance aux données |

Pour la **régression** : bruit gaussien → perte quadratique
Pour la **classification** : Bernoulli/catégoriel → entropie croisée

Le choix du modèle probabiliste détermine la perte optimale.

---

<!-- _class: lead -->

# Classifieur naïf bayésien
## Approche générative pour la classification

---

<!-- footer: "📖 Chapitre 6 : Modèles probabilistes génératifs" -->

## Approches générative vs discriminative

![w:900](_static/generative_vs_discriminative.png)

| Approche | Modélise | Question posée |
|----------|----------|----------------|
| **Générative** | $p(\mathbf{x} \mid y)$ et $p(y)$ | À quoi ressemblent les données de chaque classe? |
| **Discriminative** | $p(y \mid \mathbf{x})$ | Quelle classe pour cette observation? |

---

## L'hypothèse d'indépendance conditionnelle

Le classifieur **naïf bayésien** suppose que les caractéristiques sont **conditionnellement indépendantes** étant donné la classe :

$$p(\mathbf{x} \mid y = c) = \prod_{d=1}^D p(x_d \mid y = c)$$

**Exemple** : Classification de courriels (pourriel/légitime)
- Sachant qu'un courriel est un pourriel, la présence de « gratuit » n'influence pas la probabilité de « urgent »

Cette hypothèse réduit drastiquement le nombre de paramètres :
- Sans indépendance : $O(K^D)$ paramètres (explose!)
- Avec indépendance : $O(KD)$ paramètres

---

## Le modèle complet

**A priori sur les classes** : $p(y = c) = \pi_c$ avec $\sum_c \pi_c = 1$

**Vraisemblance par caractéristique** : $p(x_d \mid y = c; \boldsymbol{\theta}_{dc})$

**Probabilité a posteriori** :

$$p(y = c \mid \mathbf{x}) = \frac{\pi_c \prod_{d=1}^D p(x_d \mid y = c)}{\sum_{c'} \pi_{c'} \prod_{d=1}^D p(x_d \mid y = c')}$$

**Classification** : choisir la classe qui maximise le numérateur.

---

## Forme logarithmique (stable)

Pour éviter les sous-dépassements numériques (*underflow*), on travaille en log :

$$\hat{y} = \arg\max_c \left[ \log \pi_c + \sum_{d=1}^D \log p(x_d \mid y = c) \right]$$

| Avantage | Explication |
|----------|-------------|
| Pas de sous-dépassement | Produit de petites probabilités → somme de logs |
| Plus rapide | Additions au lieu de multiplications |
| Numériquement stable | Pas de problème avec $10^{-300}$ |

C'est une application de l'astuce log-sum-exp!

---

## Estimation par maximum de vraisemblance

La log-vraisemblance se factorise, permettant une estimation séparée :

**A priori de classe** (fréquence empirique) :
$$\hat{\pi}_c = \frac{N_c}{N}$$

**Caractéristiques binaires** (présence/absence) :
$$\hat{\theta}_{dc} = \frac{N_{dc}}{N_c}$$

**Caractéristiques continues** (gaussiennes) :
$$\hat{\mu}_{dc} = \frac{1}{N_c} \sum_{n: y_n = c} x_{nd}, \quad \hat{\sigma}^2_{dc} = \frac{1}{N_c} \sum_{n: y_n = c} (x_{nd} - \hat{\mu}_{dc})^2$$

Pas d'optimisation itérative : formules fermées!

---

## Le problème des probabilités nulles

Si le mot « gratuit » n'apparaît dans aucun courriel légitime : $\hat{\theta} = 0$

$$p(\text{légitime} \mid \mathbf{x}) \propto \pi_{\text{lég}} \times \ldots \times \underbrace{p(\text{gratuit} \mid \text{lég})}_{= 0} \times \ldots = 0$$

Un seul mot peut dominer entièrement la décision!

**Solution : lissage de Laplace** (*add-one smoothing*)

$$\hat{\theta}_{dck} = \frac{N_{dck} + 1}{N_c + K}$$

C'est le MAP avec un a priori uniforme (Beta(1,1) ou Dirichlet).

---

## Effet du lissage

![w:700](_static/laplace_smoothing.png)

Le lissage ajoute des « pseudo-observations » : comme si nous avions vu chaque événement au moins une fois avant de commencer.

---

## Pourquoi ça fonctionne?

L'hypothèse d'indépendance est presque toujours **violée** en pratique.

Pourtant, Naive Bayes fonctionne souvent bien. Pourquoi?

| Observation | Explication |
|-------------|-------------|
| On classe, on n'estime pas | Seul l'ordre des probabilités compte |
| Erreurs qui s'annulent | Surestimation dans toutes les classes |
| Régularisation implicite | Modèle simple = moins de surapprentissage |

**Attention** : Les probabilités retournées sont souvent mal calibrées (trop proches de 0 ou 1). Pour des probabilités fiables, préférez la régression logistique.

---

<!-- _class: lead -->

# Modèles de mélange gaussien
## Partitionnement probabiliste et algorithme EM

---

<!-- footer: "📖 Chapitre 6 : Modèles probabilistes génératifs" -->

## Du supervisé au non supervisé

| Apprentissage | Étiquettes | Objectif |
|---------------|------------|----------|
| **Supervisé** | Connues | Prédire $y$ à partir de $\mathbf{x}$ |
| **Non supervisé** | **Inconnues** | Découvrir une structure dans $\mathbf{x}$ |

Le **partitionnement** (*clustering*) cherche à regrouper les données en groupes homogènes.

Les **modèles de mélange gaussien** (GMM) offrent une approche probabiliste qui généralise l'analyse discriminante au cas non supervisé.

---

## Formulation du GMM

Un GMM suppose que les données proviennent d'un mélange de $K$ gaussiennes :

$$p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

| Paramètre | Signification | Contrainte |
|-----------|---------------|------------|
| $\pi_k$ | Poids du composant $k$ | $\sum_k \pi_k = 1$, $\pi_k \geq 0$ |
| $\boldsymbol{\mu}_k$ | Moyenne du composant $k$ | $\in \mathbb{R}^D$ |
| $\boldsymbol{\Sigma}_k$ | Covariance du composant $k$ | Semi-définie positive |

---

## Variable latente et processus génératif

On peut interpréter le GMM avec une variable latente $z \in \{1, \ldots, K\}$ :

$$p(z = k) = \pi_k, \quad p(\mathbf{x} \mid z = k) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

**Processus de génération** :
1. Tirer un composant $z \sim \text{Catégorielle}(\boldsymbol{\pi})$
2. Tirer une observation $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_z, \boldsymbol{\Sigma}_z)$

Ce processus permet de **générer de nouvelles données** synthétiques.

---

## Responsabilités : partitionnement souple

La **responsabilité** du composant $k$ pour l'observation $\mathbf{x}_n$ :

$$r_{nk} = p(z_n = k \mid \mathbf{x}_n) = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

C'est un **partitionnement souple** : chaque point « appartient » partiellement à plusieurs composants ($r_{nk} \in [0, 1]$, $\sum_k r_{nk} = 1$).

---

## Partitionnement dur vs souple

![w:900](_static/gmm_clustering.png)

À gauche : chaque point assigné au composant le plus probable. À droite : la couleur reflète les responsabilités (mélange = incertitude).

---

## Lien avec k-moyennes

K-moyennes est un cas limite de GMM avec :
- Covariances sphériques identiques : $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$
- Variance tendant vers zéro : $\sigma^2 \to 0$

| Aspect | K-moyennes | GMM |
|--------|------------|-----|
| Assignation | **Dure** (0 ou 1) | **Souple** ($\in [0,1]$) |
| Forme des groupes | Sphères | Ellipses |
| Poids des groupes | Égaux | Variables |
| Incertitude | Non quantifiée | Quantifiée |

GMM généralise k-moyennes en capturant plus de structure.

---

## Le problème d'estimation

La log-vraisemblance du GMM est :

$$\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \sum_{n=1}^N \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)$$

Le **logarithme d'une somme** rend l'optimisation difficile. Pas de solution analytique!

**Le problème de la poule et l'œuf** :
- Si on connaissait les assignations $z_n$, on pourrait estimer les paramètres
- Si on connaissait les paramètres, on pourrait calculer les assignations

---

## L'algorithme Espérance-Maximisation (EM)

EM contourne le problème par **alternance** :

**Étape E (Espérance)** : Fixer les paramètres, calculer les responsabilités

$$r_{nk}^{(t)} = \frac{\pi_k^{(t)} \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j} \pi_j^{(t)} \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}$$

**Étape M (Maximisation)** : Fixer les responsabilités, mettre à jour les paramètres

C'est une forme de **descente de coordonnées**.

---

## Étape M : mise à jour des paramètres

Soit $N_k = \sum_{n=1}^N r_{nk}$ le « nombre effectif » de points dans le composant $k$.

**Poids du mélange** :
$$\pi_k^{(t+1)} = \frac{N_k^{(t)}}{N}$$

**Moyennes** :
$$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k^{(t)}} \sum_{n=1}^N r_{nk}^{(t)} \mathbf{x}_n$$

**Covariances** :
$$\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k^{(t)}} \sum_{n=1}^N r_{nk}^{(t)} (\mathbf{x}_n - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_n - \boldsymbol{\mu}_k^{(t+1)})^\top$$

---

## Convergence de l'algorithme EM

![w:800](_static/em_convergence.gif)

À partir d'une initialisation arbitraire, EM ajuste progressivement les composants. Les ellipses représentent les contours à 1 et 2 écarts-types.

---

## Pseudocode de l'algorithme EM

```
Entrée: Données X, nombre de composants K
1. Initialiser les paramètres θ = (π, μ, Σ)
2. Répéter jusqu'à convergence:
   a. Étape E: calculer les responsabilités r_nk
   b. Étape M: mettre à jour π, μ, Σ
   c. Calculer la log-vraisemblance
3. Vérifier la convergence (ΔLL < ε)
Sortie: Paramètres θ et responsabilités r
```

EM garantit que la log-vraisemblance **augmente** (ou reste stable) à chaque itération.

---

## Considérations pratiques

| Aspect | Problème | Solution |
|--------|----------|----------|
| **Initialisation** | Maximum local | Plusieurs essais, k-means++ |
| **Choix de $K$** | Hyperparamètre | BIC, AIC, validation croisée |
| **Singularités** | Covariance dégénérée | Régularisation $\boldsymbol{\Sigma}_k + \epsilon \mathbf{I}$ |
| **Convergence** | Lente parfois | Critère d'arrêt sur $\Delta$LL |

**Initialisation recommandée** : Exécuter k-moyennes d'abord, puis utiliser les centroïdes comme moyennes initiales.

---

<!-- footer: "" -->

## Résumé : Stabilité numérique

| Technique | Application |
|-----------|-------------|
| Soustraire le max | Softmax, responsabilités GMM |
| Travailler en log | Produits de probabilités |
| Log-sum-exp | Normalisation, log-vraisemblance |

$$\text{logsumexp}(\mathbf{a}) = a_{\max} + \log \sum_{c} e^{a_c - a_{\max}}$$

---

## Résumé : Théorie de l'information

| Concept | Formule | Interprétation |
|---------|---------|----------------|
| Entropie | $\mathbb{H}(p) = -\sum_y p(y) \log p(y)$ | Incertitude intrinsèque |
| Entropie croisée | $\mathbb{H}_{\text{ce}}(p, q) = -\sum_y p(y) \log q(y)$ | Surprise avec modèle $q$ |
| Divergence KL | $D_{\text{KL}}(p \| q) = \mathbb{H}_{\text{ce}}(p, q) - \mathbb{H}(p)$ | Distance aux données |

**EMV = minimiser la divergence KL** entre le modèle et la distribution empirique.

---

## Résumé : Modèles génératifs

| Modèle | Hypothèse clé | Usage |
|--------|---------------|-------|
| **Naive Bayes** | Indépendance conditionnelle | Classification (texte, spam) |
| **LDA/QDA** | Gaussien par classe | Classification supervisée |
| **GMM** | Mélange de gaussiennes | Partitionnement non supervisé |

L'algorithme **EM** estime les paramètres des GMM par alternance E/M.

---

<!-- _class: lead -->

# Questions?

**Exercices recommandés** :
- Exercice 1 (ch5) : Entropie et divergence KL
- Exercice 1 (ch6) : Naive Bayes sur données binaires
- Exercice 4 (ch6) : Responsabilités GMM
- Exercice 5 (ch6) : Étape M de l'algorithme EM
