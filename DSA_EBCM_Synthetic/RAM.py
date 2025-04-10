'''##########################################################################################################################################################'''
'''############################################################  RAM ########################################################################################'''
'''##########################################################################################################################################################'''
''' Functions for S update'''
import numpy as np
from numpy import linalg as LA

def cholupdate(L, x):
    """
    Updates a Cholesky decomposition of a matrix L by adding the rank-1 update `x * x^T`.

    Parameters:
    - L: The current Cholesky factor of a matrix (lower triangular).
    - x: The update vector.

    Returns:
    - L: The updated Cholesky factor.
    """
    n = len(x)
    for k in range(n):
        r = np.sqrt(L[k, k]**2 + x[k]**2)  # Compute the new diagonal element
        c = r / L[k, k]  # Cosine-like factor for updating
        s = x[k] / L[k, k]  # Sine-like factor for updating
        L[k, k] = r
        if k < n - 1:
            # Update the off-diagonal elements
            L[k + 1:, k] = (L[k + 1:, k] + s * x[k + 1:]) / c
            x[k + 1:] = c * x[k + 1:] - s * L[k + 1:, k]
    return L


def choldowndate(L, x):
    """
    Downdates a Cholesky decomposition of a matrix L by subtracting the rank-1 update `x * x^T`.

    Parameters:
    - L: The current Cholesky factor of a matrix (lower triangular).
    - x: The vector to subtract.

    Returns:
    - L: The updated Cholesky factor after downdate.
    """
    n = len(x)
    for k in range(n):
        r = np.sqrt(L[k, k]**2 - x[k]**2)  # Compute the new diagonal element
        c = r / L[k, k]  # Cosine-like factor for updating
        s = x[k] / L[k, k]  # Sine-like factor for updating
        L[k, k] = r
        if k < n - 1:
            # Update the off-diagonal elements
            L[k + 1:, k] = (L[k + 1:, k] - s * x[k + 1:]) / c
            x[k + 1:] = c * x[k + 1:] - s * L[k + 1:, k]
    return L


def adapt_S(S, u, alpha_n, i, gam, target_alpha=0.23):
    """
    Adaptively updates a matrix S based on the current value of the step size and target acceptance ratio.

    Parameters:
    - S: Current covariance matrix or Cholesky factor.
    - u: Vector of proposed parameter update.
    - alpha_n: Current acceptance ratio.
    - i: Iteration number.
    - gam: Scaling factor for the step size adjustment.
    - target_alpha: Desired target acceptance ratio (default = 0.23).

    Returns:
    - S: Updated covariance matrix or Cholesky factor.
    """
    d = len(u)  # Dimension of the vector u
    dif_alpha = alpha_n - target_alpha  # Difference between current and target acceptance ratio
    a = min(1, d * i**(-gam)) * abs(dif_alpha)  # Step size adjustment factor
    u_norm = LA.norm(u)  # Normalize the update vector u
    u_ = (np.dot(S, u / u_norm)) * np.sqrt(a)  # Scale the update by the factor a
    
    # Update or downdate the covariance matrix based on the direction of alpha
    if dif_alpha > 0:
        S = cholupdate(S, u_)  # Perform Cholesky update
    else:
        S = choldowndate(S, u_)  # Perform Cholesky downdate
    
    return S

    
def qlogis(x):
    """
    Computes the logit (inverse of the logistic function) of the input.

    Parameters:
    - x: A scalar or array of values in the range (0, 1).

    Returns:
    - result: Logit-transformed values.
    """
    if isinstance(x, list):
        result = np.array([np.log(ele / (1 - ele)) for ele in x])
    else:
        result = np.log(x / (1 - x))
    return result


def plogis(x):
    """
    Computes the logistic (sigmoid) function of the input.

    Parameters:
    - x: A scalar or array of values.

    Returns:
    - result: Logistic-transformed values.
    """
    if isinstance(x, list):
        result = np.array([np.exp(ele) / (1 + np.exp(ele)) for ele in x])
    else:
        result = np.exp(x) / (1 + np.exp(x))
    return result