# Problem of CCB is how to evaluate the average reward given an interventional policy
# how to efficiently find an interventional policy that maximizes the average reward. 
# quotation libraries
#... to be filled in later

# Procedures:
# Form the IES system to take identification of CATE
# formulate the IES as an unconditional moment minimax estimator and yield the confidence set from the estimator
# Such an uncertainty quantification makes it valid to construct a policy taking greedy pessimistic action

#这个是一个bandits的framework,需要具体选择一个bandits policy进行操作
import numpy as np
from contextualbandits.offpolicy import DoublyRobustEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from contextualbandits.online import LinTS
from Interventional import expectation_in_pi_Y
from observations import binary_search_fronterior_A_x_z
from Contextualbandit import CCB_IV
from Contextualbandit import CCB_PV
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import t
    # Use Q-ensemble method to create dataset
    #这是pseudo code
    # Use Q-ensemble method to create dataset
#这是pseudo code
def merge_confidence_sets(confidence_sets):
    """
    Merge a list of confidence sets into one big list.
    Args:
        confidence_sets: a list of lists, each containing a confidence set
    Returns:
        merged_confidence_set: a list containing all elements from all confidence sets
    """
    merged_confidence_set = []
    for confidence_set in confidence_sets:
        merged_confidence_set += confidence_set
    merged_confidence_set=np.unique(merged_confidence_set)
    return merged_confidence_set

def calculate_loss(clusters,hypothesis,policy,threshold, dataset):
    """
    Calculate the loss of each cluster and return the optimal hypothesis with the smallest confidence set.
    Args:
        clusters: a list of lists, each containing a cluster of hypotheses
        hypothesis: a list of all hypotheses
    Returns:
        loss: the loss of the optimal hypothesis
        confidence_set: the confidence set of the optimal hypothesis
    """
    loss_val = None
    confidence_set = []
    for cluster in clusters:
        # Calculate the average reward of each hypothesis in the cluster
        rewards = []
        for h in cluster:
            rewards.append(expectation_in_pi_Y(dataset, h, policy))
        # Find the hypothesis with the highest average reward
        max_reward=max(rewards)
        ci=t.interval(1-threshold,len(reward),loc=max_reward, scale=np.std(rewards)/np.sqrt(len(reward)))
        max_reward_hypothesis = []
        for i, reward in enumerate(rewards):
            if ci[0] < reward < ci[1]:
                max_reward_hypothesis.append(cluster[i])
        # Calculate the confidence set of the hypothesis
        # Calculate the loss of the hypothesis
        confidence_set.append(max_reward_hypothesis)
    labels=list(range(1,len(confidence_set)+1))
    loss_val=silhouette_score(confidence_set, labels)
    return loss_val, confidence_set


def build_confidence_set(hypothesis, threshold,N,policy, dataset):
    # Use Q-ensemble method to create dataset
    # Cluster hypothesis into N clusters
    kmeans = KMeans(n_clusters=N).fit(hypothesis)
    clusters = kmeans.labels_
    # Assign each hypothesis to its corresponding cluster
    hypothesis_clusters = [[] for _ in range(N)]
    for i, h in enumerate(hypothesis):
        hypothesis_clusters[clusters[i]].append(h)
    # Create confidence set for each cluster
    confidence_sets = []
    for cluster in hypothesis_clusters:
        confidence_sets.append(cluster)
    # Merge confidence sets
    # Calculate loss based on metric and ensembled g4
    loss=silhouette_score(hypothesis, clusters)
    countb=0
    while loss > threshold:
        countb+=1
        if(countb<10):
            kmeans = KMeans(n_clusters=N,random_state=42).fit(hypothesis)
            clusters = kmeans.labels_
            # Assign each hypothesis to its corresponding cluster
            hypothesis_clusters = [[] for _ in range(N)]
            for i, h in enumerate(hypothesis):
                hypothesis_clusters[clusters[i]].append(h)
            # Create confidence set for each cluster
            confidence_sets = []
            for cluster in hypothesis_clusters:
                confidence_sets.append(cluster) 
            #confidence_sets=np.array(confidence_sets).reshape(1,-1)
            loss=silhouette_score(hypothesis, clusters)
        else:
            break
    merged_confidence_set = merge_confidence_sets(confidence_sets)
    print("merged_confidence_set",merged_confidence_set)
    return merged_confidence_set


def minimax_estimator(policy,dataset,confidence_set, threshold):
    mini_v=10000
    # Traverse the g in confidence_set to make v minimal
    mini_g=None
    mini_v_list=[]
    mini_g_list=[]
    countb=0
    for g in confidence_set:
        #print("countb, conf_set",countb,len(confidence_set))
        v = expectation_in_pi_Y(dataset, g, policy)
        if abs(v-mini_v)<threshold or v<mini_v:
            mini_v=v
            mini_v_list.append(v)
            mini_g_list.append(g)
    #这里改变policy，得到max的policy
    #How to maimize the policy to make it to the maximize mini_v
    #这里要改变x和a的list
    #print("mini_v_list",mini_v_list)
    X_list=[]
    A_list=[]
    V_list=[]
    countb=0
    for index,row in dataset.iterrows():
        #Calulate g of according X and A
        #print("index",index)
        countb+=1
        #print(countb)
        if countb<20:
            temp=CCB_IV(row['x'],row['a'],row['x'],row['y'],dataset,policy)
            for i in range(len(mini_g_list)):
                #print("gap",abs(temp-mini_g_list[i]))
                if abs(temp-mini_g_list[i])<threshold:
                    X_list.append(row['x'])
                    X_list.append(row['z'])
                    A_list.append(row['a'])
                    V_list.append(mini_v_list[i])
        else:
            break
    X_list=np.array(X_list).reshape(-1,2)
    A_list=np.array(A_list).T
    V_list=np.array(V_list)
    return X_list,A_list,V_list
    

# Construct it as PPO algorithm does
def CAP_policy_learning_IV(dataset,threshold=1e-1):
    # Now is here to build confidence set
    policy = LinTS(10)
    #print(dataset)
    context_dataset = np.vstack((dataset['x'], dataset['z'])).T
    policy.fit(X=context_dataset,a=np.array(dataset['a']),r=np.array(dataset['y']))
    Set_g=[]
    countb=0
    for index, data in dataset.iterrows():
        #print(data)
        countb+=1
        #print("index", index)
        if (countb)<20:
            Temp=CCB_IV(data['z'],data['a'],data['x'],data['y'],dataset, policy)
            #print("Temp is",Temp)
            Set_g.append(Temp)
        else:
            break
    #print("Set_g is ",Set_g)
    print("successfully generated hypothesis dataset")
    # Define the doubly robust estimator
    #doubly_robust_estimator = DoublyRobustEstimator()
    Set_g=np.array(Set_g).reshape(1,-1).T
    Conf_g=build_confidence_set(Set_g,threshold,8,policy,dataset)
    print("successfully generated confidence set")
    #over here confidence set is secure
    X_list,V_list,A_list=minimax_estimator(policy,dataset,Conf_g, 4)
    print("successule minimax estimator")
    Total_X=np.concatenate((context_dataset,X_list))
    Total_A=np.concatenate((A_list,dataset['a']))
    Total_R=np.concatenate((V_list,dataset['y']))
    policy = LinTS(10)
    policy.fit(Total_X,Total_A,Total_R)
    return policy
    #整体的CAP algorithm的流程至此完毕

def CAP_policy_learning_PV(dataset,threshold=1e-1):
    # Now is here to build confidence set
    policy = LinTS(10)
    #print(dataset)
    context_dataset = np.vstack((dataset['x'], dataset['z'])).T
    policy.fit(X=context_dataset,a=np.array(dataset['a']),r=np.array(dataset['y']))
    Set_g=[]
    countb=0
    for index, data in dataset.iterrows():
        #print(data)
        countb+=1
        #print("index", index)
        if (countb)<20:
            Temp=CCB_PV(data['z'],data['a'],data['x'],data['y'],dataset, policy)
            #print("Temp is",Temp)
            Set_g.append(Temp)
        else:
            break
    print("Set_g is ",Set_g)
    # Define the doubly robust estimator
    #doubly_robust_estimator = DoublyRobustEstimator()
    Set_g=np.array(Set_g).reshape(1,-1).T
    Conf_g=build_confidence_set(Set_g,threshold,100,policy,dataset)
    #over here confidence set is secure
    X_list,V_list,A_list=minimax_estimator(policy,dataset,Conf_g, 4)
    print("successule minimax estimator")
    Total_X=np.concatenate((context_dataset,X_list))
    Total_A=np.concatenate((A_list,dataset['a']))
    Total_R=np.concatenate((V_list,dataset['y']))
    policy = LinTS(10)
    policy.fit(Total_X,Total_A,Total_R)
    return policy
    #整体的CAP algorithm的流程至此完毕

