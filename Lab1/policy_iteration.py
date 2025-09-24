import numpy as np


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """
    Evaluate the value function from a given policy.

    Args:
            P, nS, nA, gamma: defined in the main file
            policy (np.array[nS]): The policy to evaluate. Maps states to actions.
            tol (float): Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol

    Returns:
            value_function (np.ndarray[nS]): The value function of the given policy,
            where value_function[s] is the value of state s.
    """

    V_s = np.zeros(nS)

    ### START CODE HERE ###

    ### END CODE HERE ###

    return V_s


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """
    Given the value function from policy improve the policy.

    Args:
            P, nS, nA, gamma: defined in the main file
            value_from_policy (np.ndarray): The value calculated from the policy
            policy (np.array): The previous policy

    Returns:
            new_policy (np.ndarray[nS]): An array of integers. Each integer is the optimal
            action to take in that state according to the environment dynamics and the
            given value function.
    """

    new_policy = np.zeros(nS, dtype="int")

    ### START CODE HERE ###

    ### END CODE HERE ###
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Runs policy iteration.

    Args:
            P, nS, nA, gamma: defined in the main file
            tol (float): tol parameter used in policy_evaluation()

    Returns:
            value_function (np.ndarray[nS]): value function resulting from policy iteration
            policy (np.ndarray[nS]): policy resulting from policy iteration

    Hint:
            You should call the policy_evaluation() and policy_improvement() methods to
            implement this method.
    """

    V_s = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ### START CODE HERE ###

    ### END CODE HERE ###
    return V_s, policy
