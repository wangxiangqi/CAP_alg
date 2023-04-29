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
from Interventional import expectation_in_pi_Y
from observations import binary_search_fronterior_A_x_z
from Contextualbandit import CCB_IV
from sklearn.metrics import mean_squared_error # 均方误差
def build_confidence_set(hypothesis, loss_metric, threshold,q_ensemble):
    # Use Q-ensemble method to create dataset
    #这是pseudo code
    Based on hypothesis space ensemble g to N cases: g1(x,a);...gN(x,a)
    #感觉下面的思路可以类似于KNN来做。
    calculate loss based on metric and ensembled g
    #下面假设ensemble出来的g1(x,a),...gN(x,a)，如何基于这些条件进行confidence set的构建。
    create_confidence_set(Hypothesis, ensembled G)
    #这边可以近似理解为，在hypothesis空间里面，距离ensembled G 的距离小于一定的d的集合构建为confidence seta
    while loss> threshold:
        move ensembled gN#这部分可以有点像KNN里面的移动中心点，基于loss进行操作
        create_confidence_set(Hypothesis, ensembled G)
        calculate loss based on metric and ensembled g

def minimax_estimator(policy,dataset,confidence_set):
    v=expectation_in_pi_Y(dataset,g,policy,estimator)

    

# Construct it as PPO algorithm does
def CAP_policy_learning(dataset,Loss_metric,threshold=1e-6):
    # Now is here to build confidence set
    Set_g=[]
    for data in dataset:
        Set_g.append(CCB_IV(data['z'],data['a'],data['x'],data['y'],dataset))
    Conf_g=build_confidence_set(Set_g,Loss_metric,threshold)
    #over here confidence set is secure
    policy=BootstrappedUCB(LogisticRegression())
    policy.fit(dataset['x'],,dataset['a'],dataset['y'])
    minimax_estimator(policy,dataset,Conf_g)
    return policy

def main():
    # Read dataset


    

