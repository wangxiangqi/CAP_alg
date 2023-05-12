# A Unified Framework of Policy Learning for Contextual Bandit with Confounding Bias and Missing Observations

This is a simulation for A Unified Framework of Policy Learning for Contextual Bandit with Confounding Bias and Missing Observations.

In data folder:

We implemented on FERTIL2 dataset and opted 'children' column as y, then syhthetically used the remaining columns as X, Z, A, U

In notebook folder:

We mainly stored our simulation of CAP_algorithm in there and implemented dataset preprocessing in the jupyter notebook

In src folder:

CAP_algorithm.py includes main function of CAP_policy_learning and confidence set construction(Q_ensemble method as example). And for the given loss function, to judge the ensemble performance, we used silhouette_score. For the contextual bandits algorithm, we used LinTS

For Interventional.py and observations.py:

We implemented the probability calculation and expectation calculating in interventional process and observational process.

For Contextualbandit.py:

We built in the CCB_IV part and CCB_PV method in and used the expectation search in Interventional and observations to calculate IES for estimated g(x,a)

## Parameters to tune:

1. Sampling numbers of calculating probility in observational process and interventional process
2. Hypothesis space volume
3.
