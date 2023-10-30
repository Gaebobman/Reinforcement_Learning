import gym
import numpy as np
import matplotlib.pyplot as plt

SEED = 1314

def q_learning(LR, DSCT, RES_IDX):
    NUM_EP = 2000
    env = gym.make('FrozenLake-v0')
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    rList = []
    
    for i in range(NUM_EP):
        env.seed(SEED)
        state = env.reset()
        rAll = 0
        done = False

        while not done:
            # add decaying-random noise
            # standard normal random, (1 x num_of_actions) array
            action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
            new_state, reward, done, _ = env.step(action)
            Q[state, action] = (1-LR) * Q[state, action] + LR * (reward + DSCT * np.max(Q[new_state, :]))
            state = new_state
            rAll += reward
        rList.append(rAll)
    score = str(sum(rList)/NUM_EP)
    print(f"Score over time: {score}")
    print("Final Q-Table Values")
    print(Q)        
    plt.bar(range(len(rList)), rList, color='blue')
    plt.title(f'Learning Rate: {(LR):.3f}, Discount: {DSCT}, Score: {score}')
    plt.savefig(f'./plot_res/result_{RES_IDX}.png')
    plt.show()
    env.close()
    
learning_rate_list = [round(i, 3) for i in np.linspace(.8, .99, 5).tolist()]
dis_list = [round(i, 3) for i in np.linspace(.8, .99, 5).tolist()]

res_idx = 0
for lr in learning_rate_list:
    for dsct in dis_list:
        q_learning(lr, dsct, res_idx) 
        res_idx += 1