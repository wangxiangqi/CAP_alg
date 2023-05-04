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
from contextualbandits.online import BootstrappedUCB
from Interventional import expectation_in_pi_Y
from observations import binary_search_fronterior_A_x_z
from Contextualbandit import CCB_IV
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
    return merged_confidence_set

def calculate_loss(clusters, hypothesis,policy, estimator, threshold, dataset):
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
            rewards.append(expectation_in_pi_Y(dataset, h, policy, estimator))
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


def build_confidence_set(hypothesis, threshold,N,policy,estimator):
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
    # Calculate loss based on metric and ensembled g
    loss,confidence_set = calculate_loss(confidence_sets, hypothesis)
    while loss > threshold:
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
        loss,confidence_set = calculate_loss(confidence_sets, hypothesis)

    merged_confidence_set = merge_confidence_sets(confidence_sets)
    return merged_confidence_set


def minimax_estimator(policy,dataset,confidence_set):
    mini_v=expectation_in_pi_Y(dataset,g,policy)
    # Traverse the g in confidence_set to make v minimal
    mini_g=None
    for g in confidence_set:
        v = expectation_in_pi_Y(dataset, g, policy)
        if v<mini_v:
            mini_v=v
            mini_g=g
    #这里改变policy，得到max的policy
    policy.fit(dataset['x'],dataset['a'],expectation_in_pi_Y(dataset, mini_g, policy, estimator))


    

# Construct it as PPO algorithm does
def CAP_policy_learning(dataset,threshold=1e-6):
    # Now is here to build confidence set
    policy = BootstrappedUCB(LogisticRegression(),10)
    #print(dataset)
    context_dataset = np.vstack((dataset['x'], dataset['z'])).T
    policy.fit(X=context_dataset,a=np.array(dataset['a']),r=np.array(dataset['y']))
    Set_g=[]
    for index, data in dataset.iterrows():
        #print(data)
        Set_g.append(CCB_IV(data['z'],data['a'],data['x'],data['y'],dataset, policy))
    policy=BootstrappedUCB(LogisticRegression())
    policy.fit(dataset['x'],dataset['a'],dataset['y'])
    # Define the doubly robust estimator
    #doubly_robust_estimator = DoublyRobustEstimator()
    Conf_g=build_confidence_set(Set_g,threshold,100,policy)
    #over here confidence set is secure
    minimax_estimator(policy,dataset,Conf_g)
    return policy
    #整体的CAP algorithm的流程至此完毕

