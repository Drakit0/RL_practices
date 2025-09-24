import numpy as np


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Runs value iteration.

    Args:
            P, nS, nA, gamma: defined in the main file
            tol (float): Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol

    Returns:
            value_function (np.ndarray[nS]): value function resulting from value iteration
            policy (np.ndarray[nS]): policy resulting from value iteration
    """
    V_s = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ### START CODE HERE ###

    ### END CODE HERE ###
    return V_s, policy
