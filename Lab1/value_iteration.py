import numpy as np
import time


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
    diff = float("inf")
    
    while diff >= tol:
        diff = 0.0
        
        for state in range(nS):
            v = V_s[state]
            max_v = 0
            
            for action in range(nA):
                new_v = 0
                
                for probability, nextstate, reward, _ in P[state][action]:
                    new_v += probability*(reward + gamma*V_s[nextstate])
                    
                if new_v > max_v:
                    max_v = new_v
                
            V_s[state] = max_v
            
            state_diff = np.abs(v - V_s[state])
            diff = max(diff, state_diff)

    for state in range(nS):
        max_action = 0
        max_v = -float("inf")
        
        for action in range(nA):
            new_v = 0
            
            for probability, next_state, reward, _ in P[state][action]:
                new_v += probability*(reward + gamma*V_s[next_state])
                
            if new_v > max_v:
                max_v = new_v
                max_action = action
                
        policy[state] = max_action
                
    ### END CODE HERE ###
    return V_s, policy
