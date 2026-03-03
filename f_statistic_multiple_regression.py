"""
Fonction pour calculer la F-statistic d'une régression linéaire multiple

parameters:
- y : variable dépendante (n x 1)
- *X_vars : variables indépendantes (k variables, chacune de taille n x 1)

outputs:  
- F : la F-statistic de la régression
- beta : les coefficients de régression (standardisés)    

Note : 
- Utilise la methode des moindres carres pour estimer les coefficients de la regression lineaire.
"""
def f_statistic_regression_multiple(y, *X_vars):

    import numpy as np
    
    y = np.asarray(y)
    X = np.column_stack(X_vars)

    n = len(y)
    k = X.shape[1]

    if n - k - 1 <= 0:
        raise ValueError("Pas assez d'observations.")

    # Ajout constante
    X = np.column_stack((np.ones(n), X))

    # Estimateur MCO robuste
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    y_hat = X @ beta
    y_mean = np.mean(y)

    SSR = np.sum((y_hat - y_mean)**2)
    SSE = np.sum((y - y_hat)**2)

    F = (SSR / k) / (SSE / (n - k - 1))

    return F, beta

"""
# Example

F_value, beta = f_statistic_regression_multiple(
    y,
    population,
    densite,
    entreprises
)

print("F-statistic :", F_value)
print("Coefficients (standardisés) :", beta)
"""