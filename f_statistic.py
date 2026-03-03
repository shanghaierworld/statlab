import numpy as np

def f_statistic_regression(y, X):
    """
    Calcule la statistique F pour une régression linéaire multiple.
    X : matrice des variables independantes (n x k), chaque colonne représente une variable indépendante
    y : variable dépendante (n x 1)

    faire prealablement l'appel de la fonction:
        Normalisation des donnees X et y pour eviter les problèmes de grandeurs différentes entre les variables indépendantes et la variable dépendante, ce qui peut affecter la stabilité numérique du calcul des coefficients de régression et de la statistique F.
        Concatener horizontalement les k variables independantes dans une matrice X (n x k) et la variable dépendante dans un vecteur y (n x 1).
    """
    # nombre d'observations et nombre de variables indépendantes
    n = len(y)
    k = X.shape[1]

    # Ajoute une colonne de 1 pour l'intercept au debut de la matrice X, pou avoir un modèle avec intercept (β_0)
    X = np.column_stack((np.ones(n), X))

    # Estimation des coefficients β = (X'X)^(-1)X'y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    # Prédictions
    y_hat = X @ beta

    # Moyenne de y
    y_mean = np.mean(y)

    # Sommes des carrés
    SSR = np.sum((y_hat - y_mean) ** 2)   # expliquée
    SSE = np.sum((y - y_hat) ** 2)        # résiduelle

    # Statistique F
    F = (SSR / k) / (SSE / (n - k - 1))

    return F


"""""
#  Exemple
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

F_value = f_statistic_regression(y, X)
print("F-statistic:", round(F_value, 2)) # output: F-statistic: 4.5
"""""