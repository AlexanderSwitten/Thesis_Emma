import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats

# os.chdir("C:/Users/Jonas/OneDrive/Documents/Teaching Session")


#%%

# create a function for generating simulated data in a predefined
# environment, with a predefined temperature and learning rate:

def simulation(env, temp, learn, jitter_sd = 0.5):  #Alexander: jitter_sd = 0.5
    rule_list = []
    resp_list = []
    reward_list = []    
    
    # create a list containing the rewarded response in the above list in the stable environment:
    if env == 0:
        rev_list = []
        for trial_loop in range(100):
            rule_list.append(0)
            rev_list.append(trial_loop)
        rev_list = random.sample(rev_list, 20)
        for rev_loop in range(20):
            rule_list[rev_list[rev_loop]] = 1
    
    # create a list containing the rewarded response to each presented
    # stimulus in the above list in the volatile environment:
    if env == 1:
        trial = 0
        for block_loop in range(4):
            if block_loop == 0:
                switch = np.random.randint(22, 27, 1)[0] # list is [22, 23, 24, 25, 26]
                winner, loser = 0, 1
            if block_loop == 1:
                switch = np.random.randint(47, 52, 1)[0]
                winner, loser = 1, 0
            if block_loop == 2:
                switch = np.random.randint(72, 77, 1)[0]
                winner, loser = 0, 1
            if block_loop == 3:
                switch = 100
                winner, loser = 1, 0
            rev_list = []
            for trial_loop in range(trial, switch):
                rule_list.append(winner)
                rev_list.append(trial_loop)
            rev_list = random.sample(rev_list, round(0.1*len(rev_list)))
            for rev_loop in range(len(rev_list)):
                rule_list[rev_list[rev_loop]] = loser
            trial = switch
        
    # this is where the actual simulation begins (selecting responses
    # and updating weights after each trial)
    
    # create a value vector, containing only small random weights:
    weights = np.zeros(2)
    for trial_loop in range(100):
        noise_learn = np.clip(np.random.normal(learn, jitter_sd), 0, 1) #Alexander
        noise_temp = np.clip(np.random.normal(temp, jitter_sd), 0.01, None) #Alexander
        
        # select a response:
        prob_zero = np.exp(weights[0]/noise_temp)/(np.exp(weights[0]/noise_temp)+np.exp(weights[1]/noise_temp)) #Alexander: noise_temp ipv temp
        resp = int(random.random() > prob_zero)
        resp_list.append(resp)
        
        # check whether a reward was obtained:
        rule = rule_list[trial_loop]
        if resp == rule:
            reward = 1
        if resp != rule:
            reward = 0
        reward_list.append(reward)
        
        # update the relevant weight in the weight matrix:
        weights[resp] = weights[resp] + (noise_learn*(reward-weights[resp])) #Alexander: noise_learn ipv learn
    
    # save the data in a data frame:
    data = pd.DataFrame({"rule": rule_list, "resp": resp_list, "reward": reward_list})
    # return the simulated data:
    return data

#%%

# check whether the above function works as intended:

data = simulation(env = 0, temp = 0.1, learn = 0.5)
data.to_csv("Stable_Data_Simulation_Check.csv", index = False)

data = simulation(env = 1, temp = 0.1, learn = 0.5)
data.to_csv("Volatile_Data_Simulation_Check.csv", index = False)

#%%

# run some simulations to check reward rates for different temperatures and learning rates in both
# environments (here 2 environments * 9 learning rates * 9 temperatures * 100 simulations):
    
count = 1
env_list = []
learn_list = []
temp_list = []
reward_list = []
for env_loop in range(2):
    for temp_loop in range (1, 10):
        for learn_loop in range(1, 10):
            for sim_loop in range (100):
                env = env_loop
                env_list.append(env)
                temp = temp_loop/10*2
                temp_list.append(temp)
                learn = learn_loop/10
                learn_list.append(learn)
                # simulate data in the above defined environment, with
                # the above defined temperature and learning rate:
                data = simulation(env, temp, learn)
                # save the obtained reward rate:
                reward_list.append(np.mean(data["reward"]))
        print("{}% of simulations completed".format(np.round(count/(2*9)*100, 2)))
        count = count + 1

data = pd.DataFrame({"env": env_list, "temp": temp_list,
                     "learn": learn_list, "reward": reward_list})
data.to_csv("Reward_Rates.csv", index = False)

#%%

# visualise reward rates for different temperatures and learning rates in the stable environment:

stable = data.loc[data["env"] == 0]

plt.scatter(stable["learn"], stable["temp"], c = stable["reward"], cmap = "RdYlGn", lw = 3)
plt.xlabel("learning rates")
plt.ylabel("temperatures")
bar = plt.colorbar()
bar.set_label('reward rates')
plt.savefig("Stable_Reward_Rates_Scatter_Plot")

#%%

# visualise reward rates for different temperatures and learning rates in the volatile environment:

volatile = data.loc[data["env"] == 1]

plt.scatter(volatile["learn"], volatile["temp"], c = volatile["reward"], cmap = "RdYlGn", lw = 3)
plt.xlabel("learning rates")
plt.ylabel("temperatures")
bar = plt.colorbar()
bar.set_label('reward rates')
plt.savefig("Volatile_Reward_Rates_Scatter_Plot")

#%%

# visualise reward rates for temperture = 0.2 and different learning rates in more detail:

# data = pd.read_csv("Reward_Rates.csv")
# stabe = data.loc[data["env"] == 0]

stable = stable.loc[stable["temp"] == 0.2]

data_list = []
label_list = []
for learn_loop in range(1, 10):
    data_list.append(stable.loc[stable["learn"] == learn_loop/10]["reward"])
    label_list.append("0.{}".format(learn_loop))

plt.boxplot(data_list, labels = label_list)
plt.xlabel("learning rates")
plt.ylabel("reward rates")
plt.grid(axis = "y")
plt.savefig("Stable_Reward_Rates_Box_Plot")

#%%

# visualise reward rates for temperture = 0.2 and different learning rates in more detail:
    
# data = pd.read_csv("Reward_Rates.csv")
# volatile = data.loc[data["env"] == 1]

volatile = volatile.loc[volatile["temp"] == 0.2]

data_list = []
label_list = []
for learn_loop in range(1, 10):
    data_list.append(volatile.loc[volatile["learn"] == learn_loop/10]["reward"])
    label_list.append("0.{}".format(learn_loop))

plt.boxplot(data_list, labels = label_list)
plt.xlabel("learning rates")
plt.ylabel("reward rates")
plt.grid(axis = "y")
plt.savefig("Volatile_Reward_Rates_Box_Plot")

#%%

# run some simulations to check learning rate recovery rates for temperature = 0.2 and different
# learning rates in both environments (here 2 environments * 4 learning rates * 5 simulations):

def likelihood(learn, temp, data):
    log_lik = 0
    # create a weight matrix, containing only small random weights:
    weights = np.zeros(2)
    for trial_loop in range(100):
        resp = int(data.iloc[trial_loop]["resp"])
        prob_resp = np.exp(weights[resp]/temp)/(np.exp(weights[resp]/temp)+np.exp(weights[resp-1]/temp))
        # update the total log likelihood:
        log_lik = log_lik + np.log(prob_resp)
        # update the relevant weight in the weight matrix:
        weights[resp] = weights[resp]+(learn*(data.iloc[trial_loop]["reward"]-weights[resp]))
    # return the total negative log likelihood:
    return -log_lik

count = 1
env_list = []
true_learn_list = []
est_learn_list = []
log_lik_list = []
success_list = []
for env_loop in range(2):
    for learn_loop in range (1, 5):
        for sim_loop in range (5):
            env = env_loop
            env_list.append(env)
            temp = 0.2
            learn = learn_loop/10*2
            true_learn_list.append(learn)
            # simulate data in the above defined environment, with
            # the above defined temperature and learning rate:
            data = simulation(env, temp, learn)
            # estimate the temperature and learning rate from the simulated data:
            est = optimize.differential_evolution(likelihood, [[0, 1]], args = tuple([temp, data]))
            est_learn_list.append(est.x[0])
            log_lik_list.append(est.fun)
            success_list.append(est.success)
        print("{}% of simulations completed".format(np.round(count/(2*4)*100, 2)))
        count = count + 1
    
data = pd.DataFrame({"env": env_list, "true_learn": true_learn_list, "est_learn": est_learn_list,
                     "log_lik": log_lik_list, "success": success_list})
data.to_csv("Learning_Rate_Recovery_Rates.csv", index = False)

# here we only had 2 environments * 4 learning rates * 5 simulations = 40 simulations, which took only seconds
# but, in reality, you would need at least 2 environments * 9 temperatures * 9 learning rates * 100 simulations
# = 16200 simulations, which would take (many) hours

#%%

# visualise learning rate recovery rates for different temperatures and learning rates in the stable environment:
    
stable = data.loc[data["env"] == 0]

data_list = []
label_list = []
for learn_loop in range(1, 5):
    data_list.append(stable.loc[stable["true_learn"] == learn_loop/10*2]["est_learn"])
    label_list.append("{}".format(learn_loop/10*2))

plt.boxplot(data_list, labels = label_list)
plt.xlabel("true learning rates")
plt.ylabel("estimated learning rates")
plt.grid(axis = "y")
# plt.savefig("Stable_Learning_Rate_Recovery_Rates")

print(stats.pearsonr(stable["true_learn"], stable["est_learn"]))

#%%

# visualise learning rate recovery rates for different temperatures and learning rates in the volatile environment:

volatile = data.loc[data["env"] == 1]

data_list = []
label_list = []
for learn_loop in range(1, 5):
    data_list.append(volatile.loc[volatile["true_learn"] == learn_loop/10*2]["est_learn"])
    label_list.append("{}".format(learn_loop/10*2))

plt.boxplot(data_list, labels = label_list)
plt.xlabel("true learning rates")
plt.ylabel("estimated learning rates")
plt.grid(axis = "y")
# plt.savefig("Volatile_Learning_Rate_Recovery_Rates")

print(stats.pearsonr(volatile["true_learn"], volatile["est_learn"]))