from envs.bandits import MultiBanditEnv
import numpy as np
import matplotlib.pyplot as plt
epsilons = [0.01, 0.1]
time_limit = 10000

mu_list = [0.2, -0.8, 1.7, 0.3, 1.5, -1.2, -0.1, -1.0, 1.2, -0.9]
sigma_list = [2 for _ in range(len(mu_list))]

env = MultiBanditEnv(mu_list,sigma_list)

for epsilon in epsilons:
    rewards = []
    avg_rewards = []
    q_array = np.zeros(shape=len(mu_list),dtype=np.float32)
    N_array = np.zeros(shape=len(mu_list),dtype=np.int32)
    q_tot_array = np.zeros(shape=len(mu_list),dtype=np.float32)

    for time in range(time_limit):
        if np.random.uniform(size=1)[0] > epsilon:
            action = np.argmax(q_array)
        else:
            action = np.random.randint(0,len(mu_list))
        reward = env.step(action)
        N_array[action] += 1
        q_tot_array[action] += reward
        q_array[action] = q_tot_array[action] / N_array[action]

        rewards.append(reward)
        avg_rewards.append(sum(rewards)/(time+1))

    fig = plt.plot(range(time_limit),avg_rewards,label=f"epsilon : {epsilon}")
    plt.legend()
plt.savefig("10_armed_bandit_test.png")








