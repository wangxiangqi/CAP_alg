import numpy as np 
from scipy.optimize import linprog, OptimizeResult
import matplotlib.pyplot as plt

from datetime import datetime

import argparse

gen = np.random.default_rng()

def aname(m, n): 
    """
    Naming convention for parameter arrays.

    :param m: input stage 
    :param n: 0 or 1, where 0 indicates state and 1 indicates treatment
    """
    
    return "__{}_{}".format(m, n)
    
def dtr_solve(pr, n_inputs):
    """ 
    Receives probabilities of a dtr and its dimensionality; 
    returns an optimal deterministic policy.

    :param pr: dtr parameters 
    :param n_inputs: number of input states
    
    :return sol: dictionary of np.arrays providing optimal policy and optimal reward
    """
    
    v = pr["reward"]
    solution = {}
    for m in range(n_inputs - 1, -1, -1):
        # argmax over actions for last state whose value is not known
        actions = np.argmax(v, axis=-1)
        one_hot = np.zeros(v.shape)
        for index in np.ndindex(one_hot.shape):
            if index[-1] == actions[index[:-1]]:
                one_hot[index] = 1
        solution[aname(m, 1)] = one_hot
        # max over actions for last state whose value is not known
        values = np.amax(v, axis=-1)
        # multiply by probabilities and sum to get value of new stage of states 
        prb = pr[aname(m, 0)]
        v = np.sum(prb * values, axis=-1)
    solution["value"] = v

    return solution


def dtr_eval(pr_policy, pr_ground, n_inputs):
    """
    Evaluates a solution on a DTR.
    
    :param pr_policy: probability of actions under policy
    :param pr_ground: ground truth dtr parameters
    :param n_inputs: number of input states
    
    :return v: V^{\pi = pr_policy}(s_0)
    """
    
    v = pr_ground["reward"]
    for m in range(n_inputs - 1, -1, -1):
        # multiply probability of actions under policy w/ their value
        pol = pr_policy[aname(m, 1)]
        v = np.sum(pol * v, axis=-1)
        # multiply new value by probability of reaching that state
        prb = pr_ground[aname(m, 0)]
        v = np.sum(prb * v, axis=-1)
    
    return v 


def opt_solve(n, delta, t, n_inputs, n_vocab_words, ngram_size, log_frequency):
    """
    Implements UC-DTR algorithm for policy improvement via optimistic policy 
    selection within space of SCMs given causal bounds.  
    
    :param n: counts of observed state transitions
    :param delta: failure tolerance in (0, 1)
    :param t: number of observed episodes
    :param n_inputs: number of input states
    :param n_vocab_words: dimensionality of "treatment"
    :param ngram_size: size of n-gram used when representing treatment states

    :return opt: optimally-improved deterministic policy from experience so far
    """
    
    opt = {}

    # Compute estimates of conditional probabilities P_{x_k}(s_{k+1} | s_k) 
    # and E_{x_k}[Y | s_k] from counts in dictionary n.
    est = {}
    conditional, r_marginal = n["reward"], n[aname(n_inputs - 1, 1)]
    r_min_1_marginal = np.fmax(np.ones(r_marginal.shape), r_marginal)
    est["reward"] = conditional / r_min_1_marginal
    for m in range(n_inputs - 1, 0, -1):
        conditional, marginal = n[aname(m, 0)], n[aname(m - 1, 1)]
        min_1_marginal = np.expand_dims(np.fmax(np.ones(marginal.shape), marginal), axis=-1)
        est[aname(m, 0)] = conditional / min_1_marginal

    # Maximize E_{x_k}[Y | s_k]
    r_est = est["reward"]
    if t % log_frequency == 0:
        print("\t\t\t# of nonzero reward paths encountered:", np.count_nonzero(r_est))
    total_domain = n_vocab_words ** n_inputs * ngram_size ** n_inputs
    ub_r_ground = r_est + np.sqrt((2 * np.log((2 * n_inputs * total_domain * t) / delta)) / r_min_1_marginal)
    v = np.amax(ub_r_ground, axis=-1)
    actions = np.argmax(ub_r_ground, axis=-1)
    one_hot = np.zeros(ub_r_ground.shape)
    for index in np.ndindex(one_hot.shape):
        if index[-1] == actions[index[:-1]]:
            one_hot[index] = 1
    opt[aname(n_inputs - 1, 1)] = one_hot

    for m in range(n_inputs - 1, 0, -1):
        # Compute the maximum over P_{x_k}( \cdot \,|\, s_k) \in \mathcal{P}_k} \sum_{s_{k+1}
        # of V^\ast(s_{k+1}, x_{k})P_{x_k}( s_{k+1} \,|\, s_{k}), and store in array with index x_k. 
        values = np.zeros(n[aname(m - 1, 1)].shape)
        for index in np.ndindex(values.shape):
            # V^\ast values for all values of s_{k+1} concatenated w/ the dummy variables for enforcing constraint on L_1 norm.
            n_real_decision_vars = v[index].size
            # negate since we intend to maximize
            c = np.concatenate([-v[index], np.zeros(n_real_decision_vars)])
            total_decision_vars = c.size

            # Equality constraints
            A_eq = np.concatenate([np.zeros(n_real_decision_vars), np.ones(n_real_decision_vars)])[np.newaxis,:]

            # polytope bound
            count = n[aname(m - 1, 1)][index]
            min_1_count = max(1, count)
            # m is zero-indexed but the stages are 1-indexed
            total_domain = n_vocab_words ** m * ngram_size ** m
            polytope_bound = np.sqrt((6 * n_vocab_words ** (m + 1) * np.log((2 * n_inputs * total_domain * t) / delta)) / min_1_count)
            b_eq = np.array([polytope_bound])

            # Inequality constraints 
            A_ub = np.zeros((total_decision_vars, total_decision_vars))
            # ||...||_1 < polytope bound is encoded by writing 
            # -z_i <= p_i <= z_i where \sum_{i} z_i = polytope_bound
            # (see https://docs.mosek.com/modeling-cookbook/linear.html).
            # This loop encodes both halves of the inequality:
            #     -z_i - p_i <= 0
            # and 
            #      p_i - z_i <= 0.  
            for r in range(0, total_decision_vars):
                if r < n_real_decision_vars:
                    A_ub[r, r] = -1
                    A_ub[r, n_real_decision_vars + r] = -1
                else:
                    A_ub[r, r] = -1
                    A_ub[r, r - n_real_decision_vars] = +1

            b_ub = np.zeros((total_decision_vars,))

            # Upper and lower bounds (differences between probabilities)
            p_est_next = est[aname(m, 0)][index]
            lb = list(-p_est_next) + [None] * p_est_next.size
            ub = list(np.ones(p_est_next.size) - p_est_next) + [None] * p_est_next.size
            bounds = list(zip(lb, ub))
            
            # dict-like
            result : OptimizeResult = \
                linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            # undo previous negation
            values[index] = -result['fun'] 

        v = np.amax(values, axis=-1)

        # save deterministic policy
        actions = np.argmax(values, axis=-1)
        one_hot = np.zeros(values.shape)
        for index in np.ndindex(one_hot.shape):
            if index[-1] == actions[index[:-1]]:
                one_hot[index] = 1
        opt[aname(m - 1, 1)] = one_hot

    # save the final value
    p_0_est = n[aname(0, 0)] / np.sum(n[aname(0, 0)])
    opt["value"] = np.inner(v, p_0_est)

    return opt


def uc_dtr_simulation(n_inputs, ngram_size, n_vocab_words,
        log_frequency, moving_avg, n_experiments, n_episodes):
    """
    Imitates paper experiment on random DTR.  Defaults to parameters 
    that are manageable for local machines.  Logs to stdout and 
    saves a regret visualization in the current directory.

    :param n_inputs: number of nontrivial states
    :param ngram_size: dimensionality of treatment variables (n-grams)
    :param n_vocab_words: size of vocabulary for each element of n-gram
    :param log_frequency: number of time steps between log outputs
    :param moving_avg: window size for moving average regret
    :param n_experiments: number of policy learning attempts
    :param n_episodes: number of episodes between policy improvements
    """
    
    # regrets across experiments
    regrets_t  = np.zeros((n_experiments, n_episodes))
    regrets2_t = np.zeros((n_experiments, n_episodes))
    
    # sample the DTR
    pr, dims = {}, []
    for m in range(n_inputs):
        dims.append(n_vocab_words) # word dim
        pr[aname(m, 0)] = gen.random(tuple(dims)) * gen.binomial(1, 0.4, size=tuple(dims)) + 0.01 * np.ones(tuple(dims))
        pr[aname(m, 0)] /= np.sum(pr[aname(m, 0)], axis=-1, keepdims=True)
        dims.append(ngram_size)
    pr["reward"] = 10 * gen.random(tuple(dims)) # or * gen.binomial(1, 0.01, size=tuple(dims))
    print('\n\tunique paths', pr["reward"].size)

    print("\tsampled the DTR.")

    opt = dtr_solve(pr, n_inputs)
    print("\tsolved the DTR.")
    opt_val = opt["value"]
    print("\tvalue at initial state of optimal policy:", opt_val, end='\n\n')
    
    delta = 1 / n_episodes
    regrets = np.zeros((n_episodes,))

    for k in range(n_experiments):

        print("experiment #" + str(k + 1) + ":", end="\n\n")

        print("\t[Random exploration] learning attempt #" + str(k + 1) + ":", end='\n\n')
        
        last_n = []
        for t in range(n_episodes):
            if t == 0 or (t + 1) % log_frequency == 0:
                print("\t\tt:", t + 1)
            
            # generate random policy (for illustrative purposes) 
            pl_det, dims = {}, []
            for m in range(n_inputs):
                dims.append(n_vocab_words)
                dims.append(ngram_size)
                pl_det[aname(m, 1)] = np.zeros(tuple(dims))
                for index in np.ndindex(*dims[:-1]):
                    one_hot_ind = tuple(list(index) + list(gen.choice(dims[-1], size=1)))
                    pl_det[aname(m, 1)][one_hot_ind] = 1

            # sample from random policy (just for show)
            dims = []
            for m in range(n_inputs):
                ps = pr[aname(m, 0)]
                choices = ps[tuple(dims)] if dims else ps
                c = gen.choice(choices.size, size=1, p=choices)
                dims.append(c[0])
                pt = pl_det[aname(m, 1)]
                choices = pt[tuple(dims)]
                dims.append(np.argmax(choices))
            reward = pr["reward"][tuple(dims)]

            # evaluated randomly-sampled deterministic policy 
            pl_det_value = dtr_eval(pl_det, pr, n_inputs)
            difference = opt_val - pl_det_value
            last_n.append(difference)
            if len(last_n) > moving_avg:
                last_n = last_n[-moving_avg:]
            regrets[t] += sum(last_n) / len(last_n)
            if t == 0 or (t + 1) % log_frequency == 0:
                print("\t\t\tvalue of policy at state s_0:", pl_det_value)
                print("\t\t\tdifference from optimal value:", difference)
                print("\t\t\taverage regret over last %d differences:" % moving_avg, sum(last_n) / max(len(last_n), 1))
    
        regrets /= n_experiments 

        regrets2 = np.zeros((n_episodes,))

        print("\n\t[UC-DTR] learning attempt #" + str(k + 1) + ":", end='\n\n')
        n, dims = {}, []
        for m in range(n_inputs):
            dims.append(n_vocab_words)
            n[aname(m, 0)] = np.zeros(tuple(dims))
            dims.append(ngram_size)
            n[aname(m, 1)] = np.zeros(tuple(dims))
        n["reward"] = np.zeros(tuple(dims))
        
        last_n = []
        for t in range(n_episodes):
            if t == 0 or (t + 1) % log_frequency == 0:
                print("\t\tt:", t + 1)
            
            if t == 0:
                opt, dims = {}, []
                for m in range(n_inputs):
                    dims.append(n_vocab_words)
                    dims.append(ngram_size)
                    opt[aname(m, 1)] = np.zeros(tuple(dims))
                    for index in np.ndindex(*dims[:-1]):
                        one_hot_ind = tuple(list(index) + list(gen.choice(dims[-1], size=1)))
                        opt[aname(m, 1)][one_hot_ind] = 1
                opt["value"] = dtr_eval(opt, pr, n_inputs)
            else:
                opt = opt_solve(n, delta, t + 1, n_inputs, n_vocab_words, ngram_size, log_frequency)                

            dims = []
            for m in range(n_inputs):
                ps = pr[aname(m, 0)]
                choices = ps[tuple(dims)] if dims else ps 
                c = gen.choice(choices.size, size=1, p=choices)[0]
                dims.append(c)
                n[aname(m, 0)][tuple(dims)] += 1
                pt = opt[aname(m, 1)]
                choices = pt[tuple(dims)]
                dims.append(np.argmax(choices))
                n[aname(m, 1)][tuple(dims)] += 1
            reward = pr["reward"][tuple(dims)]
            n["reward"][tuple(dims)] += reward
            
            difference = opt_val - dtr_eval(opt, pr, n_inputs)
            last_n.append(difference)
            if len(last_n) > moving_avg:
                last_n = last_n[-moving_avg:]
            regrets2[t] += sum(last_n) / len(last_n)
            if t == 0 or (t + 1) % log_frequency == 0:
                print('\t\t\tvalue of policy at state s_0:', opt["value"])
                print("\t\t\tdifference from optimal value:", difference)
                print("\t\t\taverage regret over last %d differences:" % moving_avg, sum(last_n) / max(len(last_n), 1))

        regrets2 /= n_experiments 

        regrets_t[k,:] = regrets
        regrets2_t[k,:] = regrets2

        print()

        plt.plot(np.arange(n_episodes), regrets, regrets2)
        plt.show()
        now = datetime.now()
        plt.savefig(now.strftime("uc_dtr_simulation-%m-%d-%Y-%H-%M-%S.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run UC-DTR algorithm on randomly-sampled dynamic treatment regime (DTR).")
    parser.add_argument("--n-inputs", type=int, default=2, help="number of external inputs for which resulting policy yields sequential decisions (default: 2)")
    parser.add_argument("--ngram-size", type=int, default=3, help="size of n-gram resulting policy predicts per input (default: 3)")
    parser.add_argument("--n-vocab-words", type=int, default=20, help="vocabulary size for each token of n-grams (default: 20)")
    parser.add_argument("--log-frequency", type=int, default=25, help="(default: 25)")
    parser.add_argument("--moving-avg", type=int, default=10, help="size of moving average regret window (default: 10)")
    parser.add_argument("--n-experiments", type=int, default=1, help="number of UC-DTR simulations to run (default: 1)")
    parser.add_argument("--n-episodes", type=int, default=250, help="number of episodes per iteration of policy improvement (default: 250)")
    args = parser.parse_args()
    uc_dtr_simulation(n_inputs=args.n_inputs, ngram_size=args.ngram_size, 
                      n_vocab_words=args.n_vocab_words, log_frequency=args.log_frequency,
                      moving_avg=args.moving_avg, n_experiments=args.n_experiments,
                      n_episodes=args.n_episodes)