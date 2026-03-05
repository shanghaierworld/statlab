import numpy as np
# permet de calculer la probabilite d'obtenir une valeur de chi2 aussi extreme que celle observee, en utilisant la fonction de distribution complementaire de la distribution gamma
from scipy.special import gammaincc
import warnings

def chi2_ind(observed, alpha=0.05):
    """
    Calculate the chi-squared statistic for a set of observed values.
    
    Parameters:
    observed (array-like): An array of observed values.
    
    Output:
    float: The chi-squared statistic.
    int: The degrees of freedom.
    array: The expected values.
    """
    
    observed = np.array(observed).astype(float)

    #verification que l'input est une matrice de contingence (2D)
    if observed.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    
    if np.any(observed < 0):
        raise ValueError("Observed values must be non-negative.")
    

    alpha = float(alpha)
    # aggregate observed values by rows and columns to get the expected values
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    total = observed.sum()

    #calcul des degres de liberte (n-1)*(p-1) pour la matrice de contingence de taille n*p
    ddl = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    # expected values for the observed table
    expected = np.outer(row_sums, col_sums) / total

    if np.any(expected == 0):
        raise ValueError("Expected frequencies contain zeros, chi-squared test is invalid.")

    number_value_expected = np.sum(expected < 5)
    if number_value_expected > 0.2 * expected.size:
        warnings.warn("More than 20% of expected values are less than 5. Test may be unreliable.")
    
    # statistique du chi_squared
    chi_squared_statistic = np.sum((observed - expected)**2 / expected)

    # calcul de p-value avec la fonction de gamma incomplete 
    p_value = gammaincc(ddl / 2, chi_squared_statistic / 2)

    results = {
        "chi_squared_statistic": chi_squared_statistic,
        "p_value": p_value,
        "alpha": alpha,
        "degrees_of_freedom": ddl,
        "expected_values": expected,
        "decision": "reject H_0" if p_value < alpha else "fail to reject H_0"
    }    
    return results