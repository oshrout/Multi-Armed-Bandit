#\ import libraries

from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import math

# calculate code run time
import time
start_time = time.time()


# <editor-fold desc="Theoretical Part">
#\ Compute T3.2


# <editor-fold desc="Compute T3.2">
alpha_list = [1,2]
beta_list = [1,2]

x_line = np.linspace(beta.ppf(0.01, alpha_list[0], beta_list[0]),beta.ppf(0.99, alpha_list[0], beta_list[0]), 100)

prior_rv = [beta(alpha_list[0], beta_list[0]),
            beta(alpha_list[1], beta_list[1])]

D_1 = [1] * 4 + [0] * 1
D_2 = [1] * 40 + [0] * 10

def likelihood(t,x):
    n = len(x)
    return (t ** np.sum(x)) * ((1- t) ** (n - np.sum(x)))

def posterior_rv(a,b,x):
    n = len(x)
    return beta(n * (a - 1) + np.sum(x) + 1, n * b - np.sum(x) + 1)
# </editor-fold>


#\ plot T3.2 graphs


# <editor-fold desc="plot T3.2 graphs">
# Plot the prior, likelihood and posterior pdfs for alpha = beta = 1 and D^(1)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.plot(x_line, prior_rv[0].pdf(x_line), 'b', lw=2, label="Prior")
ax.plot(x_line, likelihood(x_line ,np.array(D_1)), 'r', lw=2, label="Likelihood")
ax.plot(x_line, posterior_rv(alpha_list[0],beta_list[0],np.array(D_1)).pdf(x_line), 'g', lw=2, label="Posterior")
ax.legend()
ax.grid()
ax.set_title(r"Prior,Likelihood and Posterior pdfs for $\alpha = 1 ~\beta = 1$ and $D^{(1)}$")
#plt.savefig('C:/Users\shrout\Desktop\לימודים\תואר שני\ניסוי ומיצוי בסוכנים טבעיים ומלאכותיים\Ex2\plots\T3_2 graph 1')
plt.show()

# Plot the prior, likelihood and posterior pdfs for alpha = beta = 1 and D^(2)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.plot(x_line, prior_rv[0].pdf(x_line), 'b', lw=2, label="Prior")
ax.plot(x_line, likelihood(x_line ,np.array(D_2)), 'r', lw=2, label="Likelihood")
ax.plot(x_line, posterior_rv(alpha_list[0],beta_list[0],np.array(D_2)).pdf(x_line), 'g', lw=2, label="Posterior")
ax.legend()
ax.grid()
ax.set_title(r"Prior,Likelihood and Posterior pdfs for $\alpha = 1 ~\beta = 1$ and $D^{(2)}$")
#plt.savefig('C:/Users\shrout\Desktop\לימודים\תואר שני\ניסוי ומיצוי בסוכנים טבעיים ומלאכותיים\Ex2\plots\T3_2 graph 2')
plt.show()

# Plot the prior, likelihood and posterior pdfs for alpha = beta = 2 and D^(1)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.plot(x_line, prior_rv[1].pdf(x_line), 'b', lw=2, label="Prior")
ax.plot(x_line, likelihood(x_line ,np.array(D_1)), 'r', lw=2, label="Likelihood")
ax.plot(x_line, posterior_rv(alpha_list[1],beta_list[1],np.array(D_1)).pdf(x_line), 'g', lw=2, label="Posterior")
ax.legend()
ax.grid()
ax.set_title(r"Prior,Likelihood and Posterior pdfs for $\alpha = 2 ~\beta = 2$ and $D^{(1)}$")
#plt.savefig('C:/Users\shrout\Desktop\לימודים\תואר שני\ניסוי ומיצוי בסוכנים טבעיים ומלאכותיים\Ex2\plots\T3_2 graph 3')
plt.show()

# Plot the prior, likelihood and posterior pdfs for alpha = beta = 2 and D^(2)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.plot(x_line, prior_rv[1].pdf(x_line), 'b', lw=2, label="Prior")
ax.plot(x_line, likelihood(x_line ,np.array(D_2)), 'r', lw=2, label="Likelihood")
ax.plot(x_line, posterior_rv(alpha_list[1],beta_list[1],np.array(D_2)).pdf(x_line), 'g', lw=2, label="Posterior")
ax.legend()
ax.grid()
ax.set_title(r"Prior,Likelihood and Posterior pdfs for $\alpha = 2 ~\beta = 2$ and $D^{(2)}$")
#plt.savefig('C:/Users\shrout\Desktop\לימודים\תואר שני\ניסוי ומיצוי בסוכנים טבעיים ומלאכותיים\Ex2\plots\T3_2 graph 4')
plt.show()
# </editor-fold>

# </editor-fold>

#\ ########### Empirical part ###########

# define a function for each algorithm

def Epsilon_greedy(K,T,c,delta):
    # initialization
    empirical_mean = np.zeros(K)  # the empirical mean of each arm
    d = (0.5 + delta) - 0.5    # smallest gap: E[r]=p, r ~ Bernoulli(p)
    r = np.zeros(K)  # accumulated reward for each arm
    R_T = np.zeros(T)  # Regret for each t
    exploration_index = np.zeros(T) # number of times the algorithm selected the non-greedy choice

    count = np.zeros(K)

    # run over time to T
    for t in range (T):
        epsilon_t = np.minimum(1, c * K / ((d ** 2) * (t + 1)))
        # flip a coin with probability: epsilon_t
        if np.random.binomial(size=1, n=1, p=epsilon_t):    # success
            a_t = np.random.randint(0,K)  # choose arm randomly
        else:
            if np.sum(empirical_mean) > 0:
                a_t = np.argmax(empirical_mean) # pick the arm with the highest empirical mean
            else:   # no empirical mean yet
                a_t = np.random.randint(0,K)  # choose arm randomly

        count[a_t] += 1
        # calculate the reward and regret based on the chosen arm
        if a_t==0:  # we assume that the optimal arm is the first one
            r[a_t] += np.random.binomial(size=1, n=1, p=0.5 + delta)
            R_T[t] = 0  # no regret if we pick the optimal arm
        else:       # sub-optimal arms
            r[a_t] += np.random.binomial(size=1, n=1, p=0.5)
            R_T[t] = (0.5 + delta) - 0.5    # E[r]=p, r ~ Bernoulli(p)

        # check if we pick the greedy choice or not
        if a_t == np.argmax(empirical_mean): # the greedy choice
            exploration_index[t] = 0  # we pick the greedy choice
        else:
            exploration_index[t] = 1  # we pick the non-greedy choice

        # compute the empirical mean
        empirical_mean[a_t] = r[a_t] / count[a_t]

    return np.cumsum(R_T),np.cumsum(exploration_index) # accumulated regret and exploration index

def UCB1(K,T,delta):
    # initialization
    n_t = np.zeros(K)   # number of time we pick arm for each arm
    rho_t = np.zeros(K)
    empirical_mean = np.zeros(K)    # the empirical mean of each arm
    r = np.zeros(K)     # accumulated reward for each arm
    UCB_t = np.zeros(K) # upper confidence bound for each arm
    R_T = np.zeros(T)   # Regret for each t
    exploration_index = np.zeros(T) # number of times the algorithm selected the non-greedy choice

    # Try each arm once
    for i in range (K):
        n_t[i] += 1
        rho_t[i] = np.sqrt(2*np.log(T) / n_t[i])
        if i==0:    # we will assume that the optimal arm is the first one
            r[i] += np.random.binomial(size=1, n=1, p=0.5 + delta)
        else:       # sub-optimal arms
            r[i] += np.random.binomial(size=1, n=1, p=0.5)
        empirical_mean[i] = r[i]
        UCB_t[i] = empirical_mean[i] + rho_t[i]

    # run over time to T
    for t in range (T):
        a_t = np.argmax(UCB_t)  # pick the arm with the max UCB
        n_t[a_t] += 1
        rho_t[a_t] = np.sqrt(2 * np.log(T) / n_t[a_t])
        if a_t==0:  # we assume that the optimal arm is the first one
            r[a_t] += np.random.binomial(size=1, n=1, p=0.5 + delta)
            R_T[t] = 0  # no regret if we pick the optimal arm
        else:       # sub-optimal arms
            r[a_t] += np.random.binomial(size=1, n=1, p=0.5)
            R_T[t] = (0.5 + delta) - 0.5    # E[r]=p, r ~ Bernoulli(p)

        # check if we pick the greedy choice or not
        if a_t == np.argmax(empirical_mean):  # the greedy choice
            exploration_index[t] = 0  # we pick the greedy choice
        else:
            exploration_index[t] = 1  # we pick the non-greedy choice

        # compute the empirical mean and upper bound
        empirical_mean[a_t] = r[a_t] / (t + 1)
        UCB_t[a_t] = empirical_mean[a_t] + rho_t[a_t]

    return np.cumsum(R_T),np.cumsum(exploration_index) # accumulated regret and exploration index

def Thompson_sampling(K,T,a,b,delta):
    # initialization
    S = np.zeros(K)
    F = np.zeros(K)
    r_acc = np.zeros(K)  # accumulated reward for each arm
    n_t = np.zeros(K)  # number of time we pick arm for each arm
    empirical_mean = np.zeros(K)
    R_T = np.zeros(T)  # Regret for each t
    exploration_index = np.zeros(T)  # number of times the algorithm selected the non-greedy choice

    for t in range (T):

        # compute theta for each arm
        theta = np.random.beta(S + a, F + b, size=K)  # sample from beta distribution

        # draw the arm with the maximal theta
        a_t = np.argmax(theta)
        n_t[a_t] += 1

        # calculate the reward and regret based on the pulled arm
        if a_t == 0:  # we assume that the optimal arm is the first one
            r = np.random.binomial(size=1, n=1, p=0.5 + delta)
            r_acc[a_t] =+ r
            R_T[t] = 0  # no regret if we pick the optimal arm
        else:  # sub-optimal arms
            r = np.random.binomial(size=1, n=1, p=0.5)
            r_acc[a_t] = + r
            R_T[t] = (0.5 + delta) - 0.5  # E[r]=p, r ~ Bernoulli(p)

        # check if we pick the greedy choice or not
        if a_t == np.argmax(empirical_mean):  # the greedy choice
            exploration_index[t] = 0  # we pick the greedy choice
        else:
            exploration_index[t] = 1  # we pick the non-greedy choice

        # update success and fail parameters
        if r == 1:
            S[a_t] += 1
        else:
            F[a_t] += 1

        # compute the empirical mean
        empirical_mean[a_t] = r_acc[a_t] / n_t[a_t]

    return np.cumsum(R_T), np.cumsum(exploration_index)  # accumulated regret and exploration index

def UCB_V(K,T,delta):
    # initialization
    n_t = np.zeros(K) # number of time we pick arm for each arm
    r = np.zeros(K) # accumulated reward for each arm
    r_last = np.zeros(K) # keep the last reward for each arm
    V_acc = np.zeros(K) # accumulated variance for each arm
    V = np.zeros(K) # empirical estimate of the expected variance
    X = np.zeros(K) # empirical estimate of the expected mean
    R_T = np.zeros(T)  # Regret for each t
    exploration_index = np.zeros(T)  # number of times the algorithm selected the non-greedy choice

    # because the algorithm chooses the arm with the maximal Bound, with the convention that 1/0 = +∞ we can instead
    # play each arm once and afterward go with the maximal Bound in order to speed the algorithm
    for a_t in range (K):
        n_t[a_t] += 1
        if a_t == 0:  # we assume that the optimal arm is the first one
            r_last[a_t] = np.random.binomial(size=1, n=1, p=0.5 + delta)
        else:  # sub-optimal arms
            r_last[a_t] = np.random.binomial(size=1, n=1, p=0.5)
        r[a_t] += r_last[a_t]
        # compute the empirical variance
        X[a_t] = r[a_t] / n_t[a_t]  # the empirical mean
        V_acc[a_t] += (r_last[a_t] - X[a_t]) ** 2
        V[a_t] = V_acc[a_t] / n_t[a_t]


    # run over time to T
    for t in range (T):
        # compute the Bound for each arms
        B = X + np.sqrt(2 * V * np.log(t+1) / n_t) + (3 * np.log(t+1) / n_t)

        # play the arm with the maximal Bound
        a_t = np.argmax(B)
        n_t[a_t] += 1   # update the number of time played that arm

        # get reward from the arm and update the variables
        if a_t == 0:  # we assume that the optimal arm is the first one
            r_last[a_t] = np.random.binomial(size=1, n=1, p=0.5 + delta)
            r[a_t] += r_last[a_t]
            R_T[t] = 0  # no regret if we pick the optimal arm
        else:  # sub-optimal arms
            r_last[a_t] = np.random.binomial(size=1, n=1, p=0.5)
            r[a_t] += r_last[a_t]
            R_T[t] = (0.5 + delta) - 0.5  # E[r]=p, r ~ Bernoulli(p)

        # check if we pick the greedy choice or not
        if a_t == np.argmax(X):  # the greedy choice
            exploration_index[t] = 0  # we pick the greedy choice
        else:
            exploration_index[t] = 1  # we pick the non-greedy choice

        # compute the empirical mean and variance
        X[a_t] = r[a_t] / n_t[a_t]  # the empirical mean
        V_acc[a_t] += (r_last[a_t] - X[a_t]) ** 2
        V[a_t] = V_acc[a_t] / n_t[a_t]

    return np.cumsum(R_T), np.cumsum(exploration_index)  # accumulated regret and exploration index

#\ define variables
T = 10 ** 6 #7
K_list = [2, 10, 100]
delta_list = [0.1, 0.01]
N = 10 # number of repetitions

alpha_list = [0.5,2,5]
beta_list = [0.5,2,5]

# create a dictionary for c based on pre computing reasonable value
c_dict = {
    #(K,delta)  : c
    (2, 0.1)    : 0.1,
    (2, 0.01)   : 0.01,
    (10, 0.1)   : 0.5,
    (10, 0.01)  : 0.05,
    (100, 0.1)  : 0.5,
    (100, 0.01) : 0.005
}

#\ define function to plot and save the results
def plot_results(e_greedy, ucb1, thompson_sampling, ucb_v, T, K, delta, plot_name):
    t_array = np.arange(0,T)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.plot(t_array, e_greedy, ':b', lw=2, label="epsilon-greedy")
    ax.plot(t_array, ucb1, '--r', lw=2, label="UCB1")
    ax.plot(t_array, thompson_sampling[:,0], '-.g', lw=2,
            label=r"Thompson sampling, $\alpha = %s, ~\beta = %s$"%(alpha_list[0], beta_list[0]))
    ax.plot(t_array, thompson_sampling[:,1], '-.m', lw=2,
            label=r"Thompson sampling, $\alpha = %s, ~\beta = %s$"%(alpha_list[1], beta_list[1]))
    ax.plot(t_array, thompson_sampling[:,2], '-.y', lw=2,
            label=r"Thompson sampling, $\alpha = %s, ~\beta = %s$"%(alpha_list[2], beta_list[2]))
    ax.plot(t_array, ucb_v, ':k', lw=2, label="UCB-V")
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlabel("t - Time")

    if plot_name == 'mean regret':
        ax.set_ylabel("Mean regret")
        ax.set_title("Mean regret vs time")
    elif plot_name == 'std regret':
        ax.set_ylabel("Std regret")
        ax.set_title("Std regret vs time")
    else:
        ax.set_ylabel("Exploration index")
        ax.set_title("Exploration index vs time")

    #plt.savefig('C:/Users\shrout\Desktop\לימודים\תואר שני\ניסוי ומיצוי בסוכנים טבעיים ומלאכותיים\Ex2\plots\K=%d, delta=%g, %s.png'%(K, delta, plot_name))
    plt.show()
    return

#\ compute each case and plot

tmp_time = time.time()
print("%s"%(tmp_time - start_time))

# loop over each case of K and delta
for K in K_list:
    for delta in delta_list:
        # initialization
        R_epsilon_greedy, exp_index_epsilon_greedy = np.zeros([T,N]), np.zeros([T,N])
        R_UCB1, exp_index_UCB1 = np.zeros([T,N]), np.zeros([T,N])
        R_Thompson_sampling, exp_index_Thompson_sampling = np.zeros([T,N,3]), np.zeros([T,N,3])
        R_UCB_V, exp_index_UCB_V = np.zeros([T,N]), np.zeros([T,N])
        c_epsilon_greedy = c_dict.get((K,delta))

        # repeat each case N times
        for rep in range(N):
            R_epsilon_greedy[:,rep], exp_index_epsilon_greedy[:,rep] = Epsilon_greedy(K,T,c_epsilon_greedy,delta)
            R_UCB1[:,rep], exp_index_UCB1[:,rep] = UCB1(K,T,delta)
            for i in range(3):
                a = alpha_list[i]
                b = beta_list[i]
                R_Thompson_sampling[:, rep, i], exp_index_Thompson_sampling[:, rep, i] = Thompson_sampling(K,T,a,b,delta)
            R_UCB_V[:, rep], exp_index_UCB_V[:, rep] = UCB_V(K,T,delta)

        # plot the mean and std of the regret and the mean of the exploration index
        # plot the mean regret
        plot_results(R_epsilon_greedy.mean(axis=1),
                     R_UCB1.mean(axis=1),
                     R_Thompson_sampling.mean(axis=1),
                     R_UCB_V.mean(axis=1),
                     T, K, delta, 'mean regret')
        # plot the std of the regret
        plot_results(R_epsilon_greedy.std(axis=1),
                     R_UCB1.std(axis=1),
                     R_Thompson_sampling.std(axis=1),
                     R_UCB_V.std(axis=1),
                     T, K, delta, 'std regret')
        # plot the mean exploration index
        plot_results(exp_index_epsilon_greedy.mean(axis=1),
                     exp_index_UCB1.mean(axis=1),
                     exp_index_Thompson_sampling.mean(axis=1),
                     exp_index_UCB_V.mean(axis=1),
                     T, K, delta, 'exploration index')
        print("---")
        print("K=%d, delta=%g : %s seconds"%(K,delta,time.time() - tmp_time))
        tmp_time = time.time()



#\ compare between UCB1 and UCB-V

# define the params
K = 2
sub_arm_reward = 0.48
opt_arm_reward = 0.51
T = 10 ** 5
N = 100

# initialization
R_UCB1, exp_index_UCB1 = np.zeros([T,N]), np.zeros([T,N])
R_UCB_V, exp_index_UCB_V = np.zeros([T,N]), np.zeros([T,N])

#\ compare between UCB1 and UCB-V - first comparison

# <editor-fold desc="comparison 1">
# repeat each case N times
for rep in range(N):

    # initialization
    n_t_UCB1 = np.zeros(K)   # number of time we pick arm for each arm for UCB1
    n_t_UCBV = np.zeros(K)  # number of time we pick arm for each arm for UCB-V

    r_UCB1 = np.zeros(K)     # accumulated reward for each arm for UCB1
    r_UCBV = np.zeros(K)  # accumulated reward for each arm for UCB-V
    r_last = np.zeros(K)  # keep the last reward for each arm for UCB-V

    rho_t = np.zeros(K) # confidence interval for UCB1
    empirical_mean = np.zeros(K)    # the empirical mean of each arm for UCB1
    V = np.zeros(K)  # empirical estimate of the expected variance for UCB-V
    V_acc = np.zeros(K)  # accumulated variance for each arm for UCB-V
    UCB_t = np.zeros(K) # upper confidence bound for each arm for UCB1
    X = np.zeros(K)  # empirical estimate of the expected mean for UCB-V

    R_T_UCB1 = np.zeros(T)   # Regret for each t for UCB1
    R_T_UCBV = np.zeros(T)  # Regret for each t for UCB-V

    exploration_index_UCB1 = np.zeros(T) # number of times the algorithm selected the non-greedy choice for UCB1
    exploration_index_UCBV = np.zeros(T) # number of times the algorithm selected the non-greedy choice for UCB-V

    # Try each arm once for both algorithms
    # for UCB1 algorithm
    for a_t in range (K):
        n_t_UCB1[a_t] += 1
        rho_t[a_t] = np.sqrt(2*np.log(T) / n_t_UCB1[a_t])
        if a_t==0:    # we will assume that the optimal arm is the first one
            r_UCB1[a_t] = np.random.binomial(size=1, n=1, p=opt_arm_reward)
        else:       # sub-optimal arms
            r_UCB1[a_t] = sub_arm_reward
        empirical_mean[a_t] = r_UCB1[a_t]
        UCB_t[a_t] = empirical_mean[a_t] + rho_t[a_t]

    # for UCB-V algorithm
        n_t_UCBV[a_t] += 1
        if a_t == 0:  # we assume that the optimal arm is the first one
            r_last[a_t] = np.random.binomial(size=1, n=1, p=opt_arm_reward)
        else:  # sub-optimal arms
            r_last[a_t] = sub_arm_reward
        r_UCBV[a_t] += r_last[a_t]
        # compute the empirical variance
        X[a_t] = r_UCBV[a_t] / n_t_UCBV[a_t]  # the empirical mean
        V_acc[a_t] += (r_last[a_t] - X[a_t]) ** 2
        V[a_t] = V_acc[a_t] / n_t_UCBV[a_t]


    # run over time to T
    for t in range (T):

        # for UCB1 algorithm
        a_t = np.argmax(UCB_t)  # pick the arm with the max UCB
        n_t_UCB1[a_t] += 1
        rho_t[a_t] = np.sqrt(2 * np.log(T) / n_t_UCB1[a_t])
        if a_t==0:  # we assume that the optimal arm is the first one
            r_UCB1[a_t] += np.random.binomial(size=1, n=1, p=opt_arm_reward)
            R_T_UCB1[t] = 0  # no regret if we pick the optimal arm
            exploration_index_UCB1[t] = 0  # we pick the greedy choice
        else:       # sub-optimal arms
            r_UCB1[a_t] += sub_arm_reward
            R_T_UCB1[t] = opt_arm_reward - sub_arm_reward    # E[r]=p, r ~ Bernoulli(p)
            exploration_index_UCB1[t] = 1  # we pick the non-greedy choice
        empirical_mean[a_t] = r_UCB1[a_t] / (t + 1)
        UCB_t[a_t] = empirical_mean[a_t] + rho_t[a_t]

        # for UCB-V algorithm
        B = X + np.sqrt(2 * V * np.log(t + 1) / n_t_UCBV) + (3 * np.log(t + 1) / n_t_UCBV) # compute the Bound for each arms
        a_t = np.argmax(B) # play the arm with the maximal Bound
        n_t_UCBV[a_t] += 1
        # get reward from the arm and update the variables
        if a_t == 0:  # we assume that the optimal arm is the first one
            r_last[a_t] = np.random.binomial(size=1, n=1, p=opt_arm_reward)
            r_UCBV[a_t] += r_last[a_t]
            R_T_UCBV[t] = 0  # no regret if we pick the optimal arm
            exploration_index_UCBV[t] = 0  # we pick the greedy choice
        else:  # sub-optimal arms
            r_last[a_t] = sub_arm_reward
            r_UCBV[a_t] += r_last[a_t]
            R_T_UCBV[t] = opt_arm_reward - sub_arm_reward  # E[r]=p, r ~ Bernoulli(p)
            exploration_index_UCBV[t] = 1  # we pick the non-greedy choice

        # compute the empirical variance
        X[a_t] = r_UCBV[a_t] / n_t_UCBV[a_t]  # the empirical mean
        V_acc[a_t] += (r_last[a_t] - X[a_t]) ** 2
        V[a_t] = V_acc[a_t] / n_t_UCBV[a_t]

    R_UCB1[:,rep], exp_index_UCB1[:,rep] = np.cumsum(R_T_UCB1), np.cumsum(exploration_index_UCB1)
    R_UCB_V[:, rep], exp_index_UCB_V[:, rep] = np.cumsum(R_T_UCBV), np.cumsum(exploration_index_UCBV)
# </editor-fold>


# <editor-fold desc="Plot comparison 1">
# plot the comparison
t_array = np.arange(0,T)

fig = plt.figure(figsize=(25,10))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(t_array, R_UCB1.mean(axis=1), '--b', lw=2, label="UCB1")
ax1.plot(t_array, R_UCB_V.mean(axis=1), '-.r', lw=2, label="UCB-V")
ax1.legend(loc='upper left')
ax1.grid()
ax1.set_xlabel("t - Time")
ax1.set_ylabel("Mean regret")
ax1.set_title("Mean regret vs time")

ax2 = fig.add_subplot(2,1,2)
ax2.plot(t_array, R_UCB1.std(axis=1), '--b', lw=2, label="UCB1")
ax2.plot(t_array, R_UCB_V.std(axis=1), '-.r', lw=2, label="UCB-V")
ax2.legend(loc='upper left')
ax2.grid()
ax2.set_xlabel("t - Time")
ax2.set_ylabel("Std regret")
ax2.set_title("Std of the regret vs time")
plt.show()
# </editor-fold>


#\ compare between UCB1 and UCB-V - second comparison

# <editor-fold desc="comparison 2">
# repeat each case N times
for rep in range(N):

    # initialization
    n_t_UCB1 = np.zeros(K)   # number of time we pick arm for each arm for UCB1
    n_t_UCBV = np.zeros(K)  # number of time we pick arm for each arm for UCB-V

    r_UCB1 = np.zeros(K)     # accumulated reward for each arm for UCB1
    r_UCBV = np.zeros(K)  # accumulated reward for each arm for UCB-V
    r_last = np.zeros(K)  # keep the last reward for each arm for UCB-V

    rho_t = np.zeros(K) # confidence interval for UCB1
    empirical_mean = np.zeros(K)    # the empirical mean of each arm for UCB1
    V = np.zeros(K)  # empirical estimate of the expected variance for UCB-V
    V_acc = np.zeros(K)  # accumulated variance for each arm for UCB-V
    UCB_t = np.zeros(K) # upper confidence bound for each arm for UCB1
    X = np.zeros(K)  # empirical estimate of the expected mean for UCB-V

    R_T_UCB1 = np.zeros(T)   # Regret for each t for UCB1
    R_T_UCBV = np.zeros(T)  # Regret for each t for UCB-V

    exploration_index_UCB1 = np.zeros(T) # number of times the algorithm selected the non-greedy choice for UCB1
    exploration_index_UCBV = np.zeros(T) # number of times the algorithm selected the non-greedy choice for UCB-V

    # Try each arm once for both algorithms
    # for UCB1 algorithm
    for a_t in range (K):
        n_t_UCB1[a_t] += 1
        rho_t[a_t] = np.sqrt(2*np.log(T) / n_t_UCB1[a_t])
        if a_t==0:    # we will assume that the optimal arm is the first one
            r_UCB1[a_t] = opt_arm_reward
        else:       # sub-optimal arms
            r_UCB1[a_t] = np.random.binomial(size=1, n=1, p=0.5)
        empirical_mean[a_t] = r_UCB1[a_t]
        UCB_t[a_t] = empirical_mean[a_t] + rho_t[a_t]

    # for UCB-V algorithm
        n_t_UCBV[a_t] += 1
        if a_t == 0:  # we assume that the optimal arm is the first one
            r_last[a_t] = opt_arm_reward
        else:  # sub-optimal arms
            r_last[a_t] = np.random.binomial(size=1, n=1, p=0.5)
        r_UCBV[a_t] += r_last[a_t]
        # compute the empirical variance
        X[a_t] = r_UCBV[a_t] / n_t_UCBV[a_t]  # the empirical mean
        V_acc[a_t] += (r_last[a_t] - X[a_t]) ** 2
        V[a_t] = V_acc[a_t] / n_t_UCBV[a_t]


    # run over time to T
    for t in range (T):

        # for UCB1 algorithm
        a_t = np.argmax(UCB_t)  # pick the arm with the max UCB
        n_t_UCB1[a_t] += 1
        rho_t[a_t] = np.sqrt(2 * np.log(T) / n_t_UCB1[a_t])
        if a_t==0:  # we assume that the optimal arm is the first one
            r_UCB1[a_t] += opt_arm_reward
            R_T_UCB1[t] = 0  # no regret if we pick the optimal arm
            exploration_index_UCB1[t] = 0  # we pick the greedy choice
        else:       # sub-optimal arms
            r_UCB1[a_t] += np.random.binomial(size=1, n=1, p=0.5)
            R_T_UCB1[t] = opt_arm_reward - 0.5    # E[r]=p, r ~ Bernoulli(p)
            exploration_index_UCB1[t] = 1  # we pick the non-greedy choice
        empirical_mean[a_t] = r_UCB1[a_t] / (t + 1)
        UCB_t[a_t] = empirical_mean[a_t] + rho_t[a_t]

        # for UCB-V algorithm
        B = X + np.sqrt(2 * V * np.log(t + 1) / n_t_UCBV) + (3 * np.log(t + 1) / n_t_UCBV) # compute the Bound for each arms
        a_t = np.argmax(B) # play the arm with the maximal Bound
        n_t_UCBV[a_t] += 1
        # get reward from the arm and update the variables
        if a_t == 0:  # we assume that the optimal arm is the first one
            r_last[a_t] = opt_arm_reward
            r_UCBV[a_t] += r_last[a_t]
            R_T_UCBV[t] = 0  # no regret if we pick the optimal arm
            exploration_index_UCBV[t] = 0  # we pick the greedy choice
        else:  # sub-optimal arms
            r_last[a_t] = np.random.binomial(size=1, n=1, p=0.5)
            r_UCBV[a_t] += r_last[a_t]
            R_T_UCBV[t] = opt_arm_reward - 0.5  # E[r]=p, r ~ Bernoulli(p)
            exploration_index_UCBV[t] = 1  # we pick the non-greedy choice

        # compute the empirical variance
        X[a_t] = r_UCBV[a_t] / n_t_UCBV[a_t]  # the empirical mean
        V_acc[a_t] += (r_last[a_t] - X[a_t]) ** 2
        V[a_t] = V_acc[a_t] / n_t_UCBV[a_t]

    R_UCB1[:,rep], exp_index_UCB1[:,rep] = np.cumsum(R_T_UCB1), np.cumsum(exploration_index_UCB1)
    R_UCB_V[:, rep], exp_index_UCB_V[:, rep] = np.cumsum(R_T_UCBV), np.cumsum(exploration_index_UCBV)
# </editor-fold>



# <editor-fold desc="Plot comparison 2">
# plot the comparison
t_array = np.arange(0,T)

fig = plt.figure(figsize=(25,10))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(t_array, R_UCB1.mean(axis=1), '--b', lw=2, label="UCB1")
ax1.plot(t_array, R_UCB_V.mean(axis=1), '-.r', lw=2, label="UCB-V")
ax1.legend(loc='upper left')
ax1.grid()
ax1.set_xlabel("t - Time")
ax1.set_ylabel("Mean regret")
ax1.set_title("Mean regret vs time")

ax2 = fig.add_subplot(2,1,2)
ax2.plot(t_array, R_UCB1.std(axis=1), '--b', lw=2, label="UCB1")
ax2.plot(t_array, R_UCB_V.std(axis=1), '-.r', lw=2, label="UCB-V")
ax2.legend(loc='upper left')
ax2.grid()
ax2.set_xlabel("t - Time")
ax2.set_ylabel("Std regret")
ax2.set_title("Std of the regret vs time")
plt.show()
# </editor-fold>


#\ print the run time
print("--- Finished in: %s seconds ---" % (time.time() - start_time))




