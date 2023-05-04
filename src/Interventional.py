from contextualbandits.offpolicy import DoublyRobustEstimator
from contextualbandits.online import LinUCB
from Contextualbandit import CCB_IV,CCB_PV
import numpy as np

def interventional_process(u,x,o,a,y,policy,dataset):
    # p_marginal[x] can be calculated as follows:
    # First, filter the dataset to only include rows where 'x' equals x
    # Then, calculate the proportion of rows where 'x' equals x out of the total number of rows in the dataset
    #    This can be done using the following code:
    p_marginal = len(dataset[dataset['x'] == x]) / len(dataset)
    #To generate p(u,o|x), we need to filter the dataset to only include rows where 'x' equals x and then calculate the proportion of rows where u and o appear out of the total number of rows in the filtered dataset. 
    #This can be done using the following code:
    # Filter the dataset to only include rows where 'x' equals x
    filtered_dataset = dataset[dataset['x'] == x]
    # Calculate the proportion of rows where u and o appear out of the total number of rows in the filtered dataset
    p_uo_given_x = len(filtered_dataset[(filtered_dataset['u'] == u) & (filtered_dataset['o'] == o)]) / len(filtered_dataset)
    prob_ratio=0
    context_list = []
    for index,row in dataset.iterrows():
        context=np.append(row['x'])
        #print(a)
        #print(policy.predict(context))
        if policy.predict(context)==y:
            context_list.append(context)
    prob_ratio=np.count_nonzero(np.isin(context_list, x))/(len(dataset['x']))
    # Filter the dataset to only include rows where x equals x, u equals u, a equals a, and o equals o
    filtered_dataset = dataset[(dataset['x'] == x) & (dataset['u'] == u) & (dataset['a'] == a) & (dataset['o'] == o)]
    # Use the DoublyRobustEstimator to estimate the probability of y given the filtered dataset and the specified policy
    p_y_given_u_x_a_o = len(filtered_dataset[filtered_dataset['y'] == y]) / len(filtered_dataset)
    p_all=p_marginal*p_uo_given_x*prob_ratio*p_y_given_u_x_a_o
    return p_all

def expectation_in_pi_Y(dataset,Y,policy):
    subdataset = dataset[dataset['y'] == Y]
    expectation=0
    for index,row in dataset.iterrows():
        prob=interventional_process(row['u'],row['x'],row['o'],row['a'],row['y'],policy,dataset)
        if prob != 0:
                expectation += Y * prob
    return expectation / len(subdataset)




