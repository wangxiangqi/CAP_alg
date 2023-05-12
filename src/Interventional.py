from contextualbandits.offpolicy import DoublyRobustEstimator
from contextualbandits.online import LinUCB
from Contextualbandit import CCB_IV,CCB_PV
import numpy as np

def interventional_process(u,x,z,a,y,policy,dataset):
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
    p_uo_given_x = len(filtered_dataset[(filtered_dataset['u'] == u) & (filtered_dataset['z'] == z)]) / len(filtered_dataset)
    prob_ratio=0
    context_list = []
    countb=0
    for index,row in dataset.iterrows():
        countb+=1
        if countb<100:
            context=[]
            context=np.append(context,row['x'])
            if policy.predict(context)==y:
                context_list.append(context)
        else:
            break
    prob_ratio=np.count_nonzero(np.isin(context_list, x))/(100)
    # Filter the dataset to only include rows where x equals x, u equals u, a equals a, and o equals o
    filtered_dataset = dataset[(dataset['x'] == x) & (dataset['u'] == u) & (dataset['a'] == a) & (dataset['z'] == z)]
    # Use the DoublyRobustEstimator to estimate the probability of y given the filtered dataset and the specified policy
    p_y_given_u_x_a_o = len(filtered_dataset[filtered_dataset['y'] == y]) / len(filtered_dataset)
    p_all=p_marginal*p_uo_given_x*prob_ratio*p_y_given_u_x_a_o
    return p_all

def modularize_array(arr):
    arra=[]
    for i in range(len(arr))-1:
        #if(arr[i]==8):
            #print(arr[i])
            #print(type(arr[i]))
        if not isinstance(arr[i], float):
            arra.append(float(arr[i]))
    return arra

def expectation_in_pi_Y(dataset,Y,policy):
    expectation=0
    countb=0
    for index,row in dataset.iterrows():
        countb+=1
        #print("countb.",countb)
        if countb<100:
            prob=interventional_process(row['u'],row['x'],row['z'],row['a'],row['y'],policy,dataset)
            if prob != 0:
                expectation += Y * prob
        else:
            break
    tar_num=1
    #arra=dataset['y']
    #arra=modularize_array(arra)
    """
    for i in range(len(arra)):
         #print(arra[i])
         #print(Y)
         #print(abs(arra[i]-Y))
         if abs(arra[i]-Y)<2:
              tar_num+=1 
    tar_num=int(tar_num*100/len(dataset['x']))
    """
    return expectation / 600



