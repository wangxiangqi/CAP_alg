# Libraries to quote:
# Still to fill in
#
from contextualbandits.offpolicy import DoublyRobustEstimator
from contextualbandits.online import LinUCB
#contextual bandits算法并没有特别指定具体。
from contextualbandits.online import BootstrappedUCB, BootstrappedTS, LinUCB
from sklearn.linear_model import LogisticRegression
from contextualbandits.evaluation import evaluateRejectionSampling
import numpy as np
# define reward function
def reward_function(action, outcome):
    if action == outcome:
        return 1
    else:
        return 0
    
    
def observational_process(u,x,z,a,y,policy,dataset):
    #Use dataset to caculate p(u,x,o) and p_p(y,u,x,a,o)
    #p_p(y,u,x,a,o)=p_p(y|u,x,a,o)
    #to depict the probability of piob(a|u,x,o)
    # predict probability ratio for a given context
    # calculate count(u,x,o,a,y)
    context_list=[]
    for index,row in dataset.iterrows():
        context=np.append(row['x'],row['z'])
        #print(a)
        #print(policy.predict(context))
        if policy.predict(context)==y:
            context_list.append(context)
    # Problems here:
    # No exsiting API or algorithm input function to estimate a posterior probability
    to_count=np.append(x,z)
    context_list=np.array(context_list).flatten()
    #print(to_count)
    #print(context_list)
    print(np.count_nonzero(np.isin(context_list, to_count)))
    #print("pause1")
    prob_ratio=np.count_nonzero(np.isin(context_list, to_count))/(len(dataset['x'])*2)
    count_u_x_o_a_y = ((dataset['u'] == u) & (dataset['x'] == x) & (dataset['z'] == z) & (dataset['a'] == a) & (dataset['y'] == y)).sum()
    # calculate count(a,y)
    count_a_y = ((dataset['a'] == a) & (dataset['y'] == y)).sum()
    # calculate N
    N = len(dataset) 
    # calculate P(u,x,o,a,y)
    P_u_x_o_a_y = count_u_x_o_a_y / N
    # calculate P(a,y)
    P_a_y = count_a_y / N
    # calculate P(u,x,o)
    P_u_x_o = P_u_x_o_a_y / P_a_y
    print(P_u_x_o)
    # calculate count(y)
    count_y = (dataset['y'] == y).sum()
    # calculate count(u,x,a,o)
    count_u_x_a_o = ((dataset['u'] == u) & (dataset['x'] == x) & (dataset['a'] == a) & (dataset['z'] == z)).sum()
    # calculate P(u,x,a,o|y)
    P_u_x_a_o_y = count_u_x_o_a_y / count_y 
    # calculate P(y)
    P_y = count_y / N
    # calculate P(u,x,a,o)
    P_u_x_a_o = count_u_x_a_o / N
    # calculate P(y|u,x,a,o)
    P_y_u_x_a_o = P_u_x_a_o_y * P_y / P_u_x_a_o
    print(P_y_u_x_a_o)
    #print(prob_ratio)
    observational_prob=P_u_x_o*prob_ratio*P_y_u_x_a_o
    print("temp_pause")
    print(observational_prob)
    return observational_prob


def cal_expectation_ob_Z(dataset,Z,fronterior):
    policy = BootstrappedUCB(LogisticRegression(),10)
    #print(dataset)
    context_dataset = np.vstack((dataset['x'], dataset['z'])).T
    policy.fit(X=context_dataset,a=np.array(dataset['a']),r=np.array(dataset['y']))
    # Define the doubly robust estimator
    #在dataset中找到所有z=Z的部分。
    subdataset = dataset[dataset['z'] == Z]
    #Calculate the expectation of a factor given on its distribution using the doubly robust estimator
    expectation = 0
    #estimator, pre_out=evaluateRejectionSampling(policy,context_dataset,dataset['a'],dataset['r'])
    for index,row in dataset.iterrows():
        observational_prob = observational_process(row['u'], row['x'], row['z'], row['a'], row['y'], policy, dataset)
        if observational_prob != 0:
            expectation += fronterior * observational_prob
    return expectation / len(subdataset)

def solve_expectation_ob_A_x_Z(dataset,X,A,Z,fronterior):
    policy=BootstrappedUCB(LogisticRegression())
    policy.fit((dataset['x'],dataset['o']),dataset['a'],dataset['y'])
    # Define the doubly robust estimator
    doubly_robust_estimator = DoublyRobustEstimator()
    #在dataset中找到所有z=Z的部分。
    subdataset = dataset[(dataset['z'] == Z) & (dataset['x'] == X) & (dataset['a'] == A)]
    #Calculate the expectation of a factor given on its distribution using the doubly robust estimator
    expectation = 0
    expectation = 0
    for u in subdataset['u'].unique():
        for x in subdataset['x'].unique():
            for a in subdataset['a'].unique():
                for o in subdataset['o'].unique():
                    for y in subdataset['y'].unique():
                        observational_prob = observational_process(u, x, o, a, y, policy, doubly_robust_estimator,dataset)
                        if observational_prob != 0:
                            expectation += fronterior * observational_prob
    return expectation / len(subdataset)



def binary_search_fronterior(dataset, Z, target_expectation, start_fronterior=0, end_fronterior=1, tol=1e-1):
    # Define the policy
    fronterior = (start_fronterior + end_fronterior) / 2
    # Define the initial expectation
    expectation = cal_expectation_ob_Z(dataset, Z, fronterior)
    # Define the number of iterations
    num_iterations = 0
    # Loop until the expectation is within the tolerance or the maximum number of iterations is reached
    while abs(expectation - target_expectation) > tol and num_iterations < 100:
        # If the expectation is greater than the target, decrease the fronterior
        if expectation > target_expectation:
            end_fronterior = fronterior
        # If the expectation is less than the target, increase the fronterior
        else:
            start_fronterior = fronterior
        # Update the fronterior
        fronterior = (start_fronterior + end_fronterior) / 2
        # Update the expectation
        expectation = cal_expectation_ob_Z(dataset, Z, fronterior)
        # Increment the number of iterations
        num_iterations += 1
    # Return the final fronterior and expectation
    return fronterior, expectation

def binary_search_fronterior_A_x_z(dataset, A,X,Z, target_expectation, start_fronterior=0, end_fronterior=1, tol=1e-6):
    # Define the policy
    fronterior = (start_fronterior + end_fronterior) / 2
    # Define the initial expectation
    expectation = solve_expectation_ob_A_x_Z(dataset, Z, fronterior)
    # Define the number of iterations
    num_iterations = 0
    # Loop until the expectation is within the tolerance or the maximum number of iterations is reached
    while abs(expectation - target_expectation) > tol and num_iterations < 100:
        # If the expectation is greater than the target, decrease the fronterior
        if expectation > target_expectation:
            end_fronterior = fronterior
        # If the expectation is less than the target, increase the fronterior
        else:
            start_fronterior = fronterior
        # Update the fronterior
        fronterior = (start_fronterior + end_fronterior) / 2
        # Update the expectation
        expectation = solve_expectation_ob_A_x_Z(dataset, Z, fronterior)
        # Increment the number of iterations
        num_iterations += 1
    # Return the final fronterior and expectation
    return fronterior, expectation
