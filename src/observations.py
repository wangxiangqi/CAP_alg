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
    countb=0
    for index,row in dataset.iterrows():
        countb+=1
        if(countb<100):
            context=np.append(row['x'],row['z'])
            if policy.predict(context)==y:
                context_list.append(context)
        else:
            break
    # Problems here:
    # No exsiting API or algorithm input function to estimate a posterior probability
    to_count=np.append(x,z)
    context_list=np.array(context_list).flatten()
    #print(to_count)
    #print(context_list)
    #print(np.count_nonzero(np.isin(context_list, to_count)))
    #print("pause1")
    prob_ratio=np.count_nonzero(np.isin(context_list, to_count))/(100*2)
    count_u_x_o_a_y = 1
    countb=0
    for index,row in dataset.iterrows():
        countb+=1
        if (countb<100):
            if ((row['u'] == u) & (row['x'] == x) & (row['z'] == z) & (row['a'] == a) & (row['y'] == y)):
                count_u_x_o_a_y+=1
        else:
            break
    # calculate count(a,y)
    count_a_y = 1
    countb = 0
    for index,row in dataset.iterrows():
        countb+=1
        if (countb<100):
            if ((row['a'] == a) & (row['y'] == y)):
                count_a_y+=1
        else:
            break
    # calculate N
    N = len(dataset) 
    # calculate P(u,x,o,a,y)
    P_u_x_o_a_y = count_u_x_o_a_y / 100
    # calculate P(a,y)
    P_a_y = count_a_y / 100
    # calculate P(u,x,o)
    P_u_x_o = P_u_x_o_a_y / P_a_y
    #print(P_u_x_o)
    # calculate count(y)
    count_y = int((dataset['y'] == y).sum()*100/N)
    if count_y == 0:
        count_y = 1
    # calculate count(u,x,a,o)
    count_u_x_a_o = 1
    countb=0
    for index,row in dataset.iterrows():
        countb+=1
        if (countb<100):
            if ((row['u'] == u) & (row['x'] == x) & (row['a'] == a) & (row['y'] == y)):
                count_u_x_a_o+=1
        else:
            break
    # calculate P(u,x,a,o|y)
    P_u_x_a_o_y = count_u_x_o_a_y / count_y 
    # calculate P(y)
    P_y = count_y / 100
    # calculate P(u,x,a,o)
    P_u_x_a_o = count_u_x_a_o / 100
    # calculate P(y|u,x,a,o)
    P_y_u_x_a_o = P_u_x_a_o_y * P_y / P_u_x_a_o
    #print(P_y_u_x_a_o)
    #print(prob_ratio)
    observational_prob=P_u_x_o*prob_ratio*P_y_u_x_a_o
    #print("temp_pause")
    #print(observational_prob)
    return observational_prob


def cal_expectation_ob_Z(dataset,Z,fronterior,policy):
    # Define the doubly robust estimator
    #在dataset中找到所有z=Z的部分。
    subdataset = dataset[dataset['z'] == Z]
    #Calculate the expectation of a factor given on its distribution using the doubly robust estimator
    expectation = 0
    #estimator, pre_out=evaluateRejectionSampling(policy,context_dataset,dataset['a'],dataset['r'])
    countb=0
    for index,row in subdataset.iterrows():
        countb+=1
        if(countb<100):
            observational_prob = observational_process(row['u'], row['x'], row['z'], row['a'], row['y'], policy, dataset)
            if observational_prob != 0:
                expectation += fronterior * observational_prob
        else:
            break
    return expectation / 100

def solve_expectation_ob_A_x_Z(dataset,A,X,Z,fronterior,policy):
    # Define the doubly robust estimator
    #在dataset中找到所有z=Z的部分。
    subdataset = dataset[(dataset['z'] == Z) & (dataset['x'] == X) & (dataset['a'] == A)]
    #Calculate the expectation of a factor given on its distribution using the doubly robust estimator
    expectation = 0
    countb=0
    for index,row in subdataset.iterrows():
        countb+=1
        if(countb<100):
            observational_prob = observational_process(row['u'], row['x'], row['z'], row['a'], row['y'], policy,dataset)
            if observational_prob != 0:
                expectation += fronterior * observational_prob
        else:
            break
    return expectation / 100



def binary_search_fronterior(dataset, Z, policy, target_expectation, start_fronterior=0, end_fronterior=1, tol=1e-6):
    # Define the policy
    fronterior = (start_fronterior + end_fronterior) / 2
    # Define the initial expectation
    expectation = cal_expectation_ob_Z(dataset, Z, fronterior, policy)
    #print("expectation",expectation)
    # Define the number of iterations
    num_iterations = 0
    # Loop until the expectation is within the tolerance or the maximum number of iterations is reached
    while abs(expectation - target_expectation) > tol and num_iterations < 5:
        # If the expectation is greater than the target, decrease the fronterior
        #print("num_iterations",num_iterations)
        if expectation > target_expectation:
            end_fronterior = fronterior
        # If the expectation is less than the target, increase the fronterior
        else:
            start_fronterior = fronterior
        # Update the fronterior
        fronterior = (start_fronterior + end_fronterior) / 2
        # Update the expectation
        expectation = cal_expectation_ob_Z(dataset, Z, fronterior,policy)
        # Increment the number of iterations
        num_iterations += 1
    # Return the final fronterior and expectation
    return fronterior, expectation

def binary_search_fronterior_A_x_z(dataset, Z,X,A,policy, target_expectation, start_fronterior=0, end_fronterior=1, tol=1e-6):
    # Define the policy
    fronterior = (start_fronterior + end_fronterior) / 2
    # Define the initial expectation
    expectation = solve_expectation_ob_A_x_Z(dataset, A,X,Z, fronterior, policy)
    #print("expectation",expectation)
    # Define the number of iterations
    num_iterations = 0
    # Loop until the expectation is within the tolerance or the maximum number of iterations is reached
    while abs(expectation - target_expectation) > tol and num_iterations < 5:
        # If the expectation is greater than the target, decrease the fronterior
        #print("num_iterations",num_iterations)
        if expectation > target_expectation:
            end_fronterior = fronterior
        # If the expectation is less than the target, increase the fronterior
        else:
            start_fronterior = fronterior
        # Update the fronterior
        fronterior = (start_fronterior + end_fronterior) / 2
        # Update the expectation
        expectation = solve_expectation_ob_A_x_Z(dataset, A,X,Z, fronterior, policy)
        # Increment the number of iterations
        num_iterations += 1
    # Return the final fronterior and expectation
    return fronterior, expectation
