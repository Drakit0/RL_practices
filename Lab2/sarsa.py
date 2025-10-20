import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def sarsa(env, num_episodes, alpha, gamma, epsilon):
    """SARSA algorithm implementation."""

    # START YOUR CODE HERE
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    
    steps_episodes = {}
    rewards_episodes = {}
    Q_13_U = {}
    Q_13_D = {}
    Q_22_U = {}
    Q_22_D = {}
    
    for eps in range(num_episodes):
        s, _ = env.reset()
        
        steps = 0
        rewards = 0
        
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
            
            Q[s, a] = Q[s, a] + alpha * (r + gamma * Q[s_p, a_p] - Q[s, a])
            
            s = s_p
            a = a_p
            
            rewards += r
            steps += 1
        
        if steps in steps_episodes.keys():
            steps_episodes[steps] += 1
            
        else:
            steps_episodes[steps] = 1
        
        rewards_episodes[eps + 1] = rewards
        Q_13_U[eps + 1] = Q[13, 0]
        Q_13_D[eps + 1] = Q[13, 1]
        Q_22_U[eps + 1] = Q[22, 0]
        Q_22_D[eps + 1] = Q[22, 1]
        
    plt.figure(figsize=(12, 6))
    plt.bar(steps_episodes.keys(), steps_episodes.values())
    plt.xlabel('Steps')
    plt.ylabel('Episodes')
    plt.title('Q-learning steps per episode')
    plt.savefig('fonts/sarsa_steps_eps.png')
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_episodes.keys(), rewards_episodes.values())
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.title('Q-learning rewards per episode')
    plt.savefig('fonts/sarsa_rewards_eps.png')
    
    plt.figure(figsize=(12, 8))
    plt.plot(Q_13_U.keys(), Q_13_U.values(), label='Q(13, UP_RIGHT)')
    plt.plot(Q_13_D.keys(), Q_13_D.values(), label='Q(13, DOWN_RIGHT)')
    plt.plot(Q_22_U.keys(), Q_22_U.values(), label='Q(22, UP_RIGHT)')
    plt.plot(Q_22_D.keys(), Q_22_D.values(), label='Q(22, DOWN_RIGHT)')
    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.title('Q-learning Q-value convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fonts/sarsa_q_convergence.png')
    
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
