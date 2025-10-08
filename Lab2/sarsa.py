import numpy as np
import gymnasium as gym


def sarsa(env, num_episodes, alpha, gamma, epsilon):
    """SARSA algorithm implementation."""

    # START YOUR CODE HERE
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    
    for _ in range(num_episodes):
        s, _ = env.reset()
        
        if np.random.random() < epsilon:
            a = np.random.choice(nA)
        
        else:
            a = np.argmax(Q[s, :])     
            
        t = False
        
        while not t:
            
            s_p, r, t, _ , _ = env.step(a)
            
            if np.random.random() < epsilon:
                a_p = np.random.choice(nA)
            
            else:
                a_p = np.argmax(Q[s_p, :])
            
            Q[s, a] = Q[s, a] + alpha * (r + gamma * (Q[s_p, a_p] - Q[s, a]))
            
            s = s_p
            a = a_p            

    # END YOUR CODE HERE

    return Q


# Env registration
gym.envs.registration.register(
    id="JumpToTheGoalEnv-v0",
    entry_point="env:JumpToTheGoalEnv",
)
env = gym.make("JumpToTheGoalEnv-v0", render_mode="human", deterministic=True)

# Hyperparameters
num_episodes = 100
alpha = 0.9
gamma = 0.9
epsilon = 0.05

# Training 
Q_sarsa = sarsa(env, num_episodes, alpha, gamma, epsilon)


# Policy eval and visualization
def visualize_policy(Q):
    actions = ["UP_RIGHT", "DOWN_RIGHT"]
    # DEFINE HERE THE POLICY TO BE EVALUATED
    # START YOUR CODE HERE
    policy = np.zeros(env.observation_space.n, dtype = np.int64)
    
    for state in range(env.observation_space.n):
        policy[state] = np.argmax(Q[state, :])

    policy = policy.reshape((7, 5))
    
    # END YOUR CODE HERE
    print("\nLearned Policy:")
    for row in policy:
        print([actions[a] for a in row])

    state, _ = env.reset()
    env.render()
    done = False
    
    policy = policy.flatten()
    
    while not done:
        # DEFINE HERE THE ACTION SELECTION, AND THE UPDATE OF THE ENVIRONMENT STATE
        # START YOUR CODE HERE
        action = policy[state]

        state, _, done, _, _ = env.step(action)
        
        # END YOUR CODE HERE
        env.render()  
    env.close() 


visualize_policy(Q_sarsa)
