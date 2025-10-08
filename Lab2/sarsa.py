import numpy as np
import gymnasium as gym


def sarsa(env, num_episodes, alpha, gamma, epsilon):
    """SARSA algorithm implementation."""

    # START YOUR CODE HERE

    # END YOUR CODE HERE

    return Q


# Env registration
gym.envs.registration.register(
    id="JumpToTheGoalEnv-v0",
    entry_point="env:JumpToTheGoalEnv",
)
env = gym.make("JumpToTheGoalEnv-v0", render_mode="human", deterministic=True)

# Hyperparameters
num_episodes =
alpha =
gamma =
epsilon =

# Training 
Q_sarsa = sarsa(env, num_episodes, alpha, gamma, epsilon)


# Policy eval and visualization
def visualize_policy(Q):
    actions = ["UP_RIGHT", "DOWN_RIGHT"]
    # DEFINE HERE THE POLICY TO BE EVALUATED
    # START YOUR CODE HERE

    # END YOUR CODE HERE
    print("\nLearned Policy:")
    for row in policy:
        print([actions[a] for a in row])

    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        # DEFINE HERE THE ACTION SELECTION, AND THE UPDATE OF THE ENVIRONMENT STATE
        # START YOUR CODE HERE

        # END YOUR CODE HERE
        env.render()  
    env.close() 


visualize_policy(Q_sarsa)
