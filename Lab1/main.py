###########################################
### PRÁCTICA 1 APRENDIZAJE POR REFUERZO ###
###########################################

## APARTADO C ##
"""
Enunciado del apartado C
"""
import time
import argparse
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from value_iteration import value_iteration
from policy_iteration import policy_iteration

register(
    id="JumpToTheGoalEnv-v0",
    entry_point="env:JumpToTheGoalEnv",
)

parser = argparse.ArgumentParser(description="Assignment 1 for the IMAT RL course.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--env_behavior",
    help="Define the environment as deterministic or stochastic.",
    choices=["deterministic", "stochastic"],
    default="deterministic",
)

parser.add_argument("--algorithm", help="The name of the algorithm to run.", choices=["both", "policy_iteration", "value_iteration"], default="both")

def render_single(env, policy):
    state, _ = env.reset()
    for a in policy:
        next_state, reward, done, _, _ = env.step(policy[state])
        state = next_state
        env.render()
        time.sleep(0.5)
        if done:
            break
    env.close()


# Edit below to run policy and value iteration on different environments and visualize the resulting policies.
# You may change the parameters in the functions below
if __name__ == "__main__":
    # read in script arguments
    args = parser.parse_args()

    # Make gym environment
    deterministic = True if args.env_behavior == "deterministic" else False
    actions_prob = 1.0 if args.env_behavior == "deterministic" else 0.8
    env = gym.make("JumpToTheGoalEnv-v0", render_mode="human", deterministic=deterministic, prob=actions_prob)

    # Hyperparameters
    gamma = 0.9
    convergence_tolerance = 1e-3

    """
    For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
    the parameters P, nS, nA, gamma are defined as follows:

        P (dict): From gym.core.Environment
            For each pair of states in [0, nS - 1] and actions in [0, nA - 1], P[state][action] is a
            list of tuples of the form [(probability, nextstate, reward, terminal),...] where
                - probability: float
                    the probability of transitioning from "state" to "nextstate" with "action"
                - nextstate: int
                    denotes the state we transition to (in range [0, nS - 1])
                - reward: int
                    the reward for transitioning from "state" to "nextstate" with "action"
                - terminal: bool
                True when "nextstate" is a terminal state (obstacle or goal), False otherwise
        nS (int): number of states in the environment
        nA (int): number of actions in the environment
        gamma (float): Discount factor. Number in range [0, 1)
    """

    if (args.algorithm == "both") | (args.algorithm == "policy_iteration"):
        print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

        start_time = time.time()
        V_pi, p_pi = policy_iteration(env.unwrapped.P, env.unwrapped.nS, env.unwrapped.nA, gamma=gamma, tol=convergence_tolerance)
        time_pi = time.time() - start_time
        print(f"Policy iteration completed after {time_pi:.6f} seconds. \n")
        print("V* values (Policy Iteration):")
        np.set_printoptions(precision=2, suppress=True)
        print(V_pi.reshape((7, 5)))
        print("\n")
        print("π* values (Policy Iteration):")
        print(p_pi.reshape((7, 5)))
        print("\n")
        render_single(env, p_pi)

    if (args.algorithm == "both") | (args.algorithm == "value_iteration"):

        print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

        start_time = time.time()
        V_vi, p_vi = value_iteration(env.unwrapped.P, env.unwrapped.nS, env.unwrapped.nA, gamma=gamma, tol=convergence_tolerance)
        time_vi = time.time() - start_time
        print(f"Value iteration completed after {time_vi:.6f} seconds. \n")
        print("V* values (Value Iteration):")
        np.set_printoptions(precision=2, suppress=True)
        print(V_vi.reshape((7, 5)))
        print("\n")
        print("π* values (Value Iteration):")
        print(p_vi.reshape((7, 5)))
        print("\n")
        render_single(env, p_vi)
