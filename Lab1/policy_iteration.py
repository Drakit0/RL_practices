import numpy as np
import matplotlib.pyplot as plt


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
    diff = float("inf")
    
    diffs = []
    
    while diff >= tol:
        diff = 0.0

        for state in range(nS):
            v = V_s[state]
            new_v = 0

            for probability, nextstate, reward, _ in P[state][policy[state]]:
                new_v += probability*(reward + gamma*V_s[nextstate])

            V_s[state] = new_v

            state_diff = np.abs(v - V_s[state])
            diff = max(diff, state_diff)
            diffs.append(diff)
    
    ### END CODE HERE ###

    return V_s, diffs


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
    
    for state in range(nS):
        new_action = 0
        max_v = -float("inf")
        
        for action in range(nA):
            new_v = 0
            
            for probability, next_state, reward, _ in P[state][action]:
                new_v += probability*(reward + gamma*value_from_policy[next_state])
                
            if new_v > max_v:
                max_v = new_v
                new_action = action
                
        new_policy[state] = new_action
        
    ### END CODE HERE ###        
    
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3, deterministic=True):
    """Runs policy iteration.

    Args:
        P, nS, nA, gamma: defined in the main file
        tol (float): tol parameter used in policy_evaluation()
        deterministic (bool): deterministic model or stochastic

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
    
    diffs = {}
    i = 1
    
    while True:
        new_V_s, diffs_i = policy_evaluation(P, nS, nA, policy, gamma, tol)
        
        diffs[i] = diffs_i
        i += 1
        
        if np.allclose(new_V_s, V_s, atol = 1e-3):
            break
        
        V_s = new_V_s.copy()
        
        policy = policy_improvement(P, nS, nA, V_s, policy, gamma)
        
    V_s = new_V_s.copy()
    unmodified_V_s = new_V_s.copy()


    # V_s update
    rows, cols = 2, 2
    plt.figure(figsize=(12, 10))
    
    for idx, (iteration, diffs_i) in enumerate(diffs.items()):
        plt.subplot(rows, cols, idx + 1)
        plt.plot(diffs_i, color=plt.cm.tab10(idx % 10))
        plt.xlabel("States")
        plt.ylabel("Max Difference")
        plt.title(f"Iter. {iteration}")
        plt.grid(True)
    
    plt.suptitle(f"V_s update in policy iteration ({"deterministic" if deterministic else "stochastic"})", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f"fonts/v_s_policy_{"deterministic" if deterministic else "stochastic"}.png")
    plt.close()
    
    
    # Policy suboptimality gap calculation (ChatGPT generated)
    gaps = {}
    i = 1

    # Recalculate to get gaps for each iteration
    V_s = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    while True:
        new_V_s, _ = policy_evaluation(P, nS, nA, policy, gamma, tol)
        
        # Calculate Q values and policy gap
        gap_values = []
        for state in range(nS):
            max_q = -float("inf")
            policy_q = 0
            
            # Calculate Q(s,a) for all actions
            for action in range(nA):
                q_value = 0
                for probability, next_state, reward, _ in P[state][action]:
                    q_value += probability * (reward + gamma * new_V_s[next_state])
                if action == policy[state]:
                    policy_q = q_value
                max_q = max(max_q, q_value)
            
            gap_values.append(max_q - policy_q)
        
        gaps[i] = max(gap_values)
        i += 1
        
        if np.allclose(new_V_s, V_s, atol=1e-3):
            break
        
        V_s = new_V_s.copy()
        policy = policy_improvement(P, nS, nA, V_s, policy, gamma)

    # Plot policy suboptimality gap
    plt.figure(figsize=(10, 6))
    iterations = list(gaps.keys())
    gap_values = list(gaps.values())
    plt.plot(iterations, gap_values, 'o-', color='red', linewidth=2, markersize=6)
    plt.xlabel("Iteration")
    plt.ylabel("Policy Suboptimality Gap")
    plt.title(f"Policy Suboptimality Gap ({"deterministic" if deterministic else "stochastic"})")
    plt.grid(True)
    plt.savefig(f"fonts/policy_gap_{"deterministic" if deterministic else "stochastic"}.png")
    plt.close()

    ### END CODE HERE ###
    return unmodified_V_s, policy
