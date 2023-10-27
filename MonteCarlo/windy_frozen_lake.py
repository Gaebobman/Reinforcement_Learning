"""
For the lecture note MonteCarlo program example, implement your code.
Submit the report as a file that includes the code and result graphs.
Result graphs: as increasing the episodes, show the average success probability for every 500 episodes. 
Using different epsilon and different constant alpha values.
Instead of variable alpha (simple average), apply constant alpha.
"""

import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Dictonary to store the average success rate for every 500 episodes
success_per_episode_dict = {}


def cross_lake(alpha=0.1, epsilon=0.2):
    print(f'alpha: {alpha}, epsilon: {epsilon}')
    '''
    Wrappers will allow us to add functionality to environments, such as modifying
    observations and rewards to be fed to our agent.
    '''
    # deterministic state transition
    env = gym.make('FrozenLake-v0', is_slippery=False)  
    # Monitor can write information about your agent’s performance in a file
    env = wrappers.Monitor(env, './frozen_lake_results', force=True)

    # Action value initialization
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    # to compute the number of visits for (state, action) pairs
    n_s_a = np.zeros([env.observation_space.n, env.action_space.n])

    num_episodes = 100000
    # epsilon = 0.2
    # to record the return at the starting point for all episodes.  [0, 1] (fail, success)
    rList = []
    success_per_episode_list = [0]
    for i in range(num_episodes):
        state = env.reset()
        rAll = 0
        # done is True when the agent reaches to the hole or goal.
        done = False        
        # to record the (state, action) trace for the current episode.
        results_list = []
        result_sum = 0.0
        while not done:
            if np.random.rand() < epsilon:
                # Random action selection
                action = env.action_space.sample()
            else:
                # Greedy action selection
                action = np.argmax(Q[state, :])
            new_state, reward, done, _ = env.step(action)
            # Append (state, action) for tracing
            results_list.append((state, action))
            # Sum of rewards (What is the discount value?)
            result_sum += reward
            state = new_state
            rAll += reward
        rList.append(rAll)
        for (state, action) in results_list:
            # First visit MC on every-visit 
            n_s_a[state, action] += 1.0
            # variable or constant step size?
            # Uncomment under line for constant step size
            # alpha = 1.0 / n_s_a[state, action]
            Q[state, action] += alpha * (result_sum - Q[state, action])
        if i % 500 == 0 and i is not 0:
            # i: Variable alpha
            success_rate = sum(rList) / i
            success_per_episode_list.append(success_rate)
            # print(f"Success rate of {i}th episode: {success_rate:.3f}")
        
    avg_success_rate = sum(rList) / num_episodes
    # print(f"Average Success rate: {avg_success_rate:.3f}")
    success_per_episode_list.append(avg_success_rate)
    success_per_episode_dict[(alpha, epsilon)] = success_per_episode_list
    
    env.close()


alpha_list= [round(i, 7) for i in np.linspace(0.000001, 0.3, 10).tolist()]
eplison_list = [0.2, 0.3, 0.4, 0.5]
for alpha in alpha_list:
    for epsilon in eplison_list:
        cross_lake(alpha, epsilon)

# Show or save the graph 

x = [i for i in range(0, 100001, 500)]
iter_num = 0
for alpha in alpha_list:
    for epsilon in eplison_list:
        value = success_per_episode_dict[(alpha, epsilon)]
        # plt.plot(x, value, label=f'alpha: {alpha}, epsilon: {epsilon}')
        plt.plot(x, value, label=f'alpha: {alpha}, epsilon: {epsilon}, final value: {value[-1]}')
        plt.legend()
    
    # To plot the graph uncomment the below line 
    # plt.show()
    # To save the graph
    plt.savefig(f'./plot_res/result_{iter_num}.png')
    iter_num += 1
    plt.close()  
    
    
    
# Since the results were poor, I tried to use diffent epsilon values
alpha_list= [round(i, 7) for i in np.linspace(0.000001, 0.3, 10).tolist()]
eplison_list = [0.05, 0.1, 0.15, 0.2]
for alpha in alpha_list:
    for epsilon in eplison_list:
        cross_lake(alpha, epsilon)


plt.figure(figsize=(10, 8))
x = [i for i in range(0, 100001, 500)]
iter_num = 0
for alpha in alpha_list:
    for epsilon in eplison_list:
        value = success_per_episode_dict[(alpha, epsilon)]
        # plt.plot(x, value, label=f'alpha: {alpha}, epsilon: {epsilon}')
        plt.plot(x, value, label=f'alpha: {alpha}, epsilon: {epsilon}, final value: {value[-1]}')
        plt.legend()
    # To plot the graph uncomment the below line 
    # plt.show()
    # To save the graph
    plt.savefig(f'./plot_res0/result_{iter_num}.png')
    iter_num += 1
    plt.close()  
    

# Save the dictionary for future use
with open('success_per_episode_dict.pickle', 'wb') as f:
    pickle.dump(success_per_episode_dict, f, pickle.HIGHEST_PROTOCOL)
plt.figure(figsize=(10, 8))