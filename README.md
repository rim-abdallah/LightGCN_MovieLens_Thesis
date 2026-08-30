Community-Aware LightGCN for Coded Caching

This repository contains code developed and adapted for a Master's thesis on community-aware cache placement with heterogeneous user demand.

The project uses LightGCN to learn user-item preferences from MovieLens interactions. The learned preference scores are then used to build user-specific request distributions, detect user communities, construct global and community-level popularity distributions, and prepare inputs for coded-caching experiments.

Project overview

The main workflow is:

Preprocess the MovieLens dataset.

Train LightGCN on user-movie interactions.

Export user-item scores and convert them into user preference distributions.

Build a user similarity graph.

Detect user communities using Louvain community detection.

Construct different cache-placement distributions:

individual user preference;

global popularity across users;

community-level popularity;

hybrid distributions controlled by an alpha parameter.

Prepare data for coded-caching experiments.

Analyze and visualize the resulting distributions and experiment outputs.

Dataset

This project uses MovieLens Latest Small (ml-latest-small), provided by the GroupLens Research group at the University of Minnesota.

The dataset contains:

610 users

9,742 movies

100,836 ratings

ratings collected between March 1996 and September 2018

MovieLens dataset page:

https://grouplens.org/datasets/movielens/latest/

Direct download:

https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Dataset documentation:

https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html

The MovieLens dataset is not included in this repository. It should be downloaded directly from GroupLens and used according to the dataset's terms of use.

MovieLens citation

If using the MovieLens dataset in academic work, please cite:

F. Maxwell Harper and Joseph A. Konstan. 2015.
The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), Article 19.
https://doi.org/10.1145/2827872

LightGCN

The recommendation component of this project is based on the original TensorFlow implementation of LightGCN.

Original repository:

https://github.com/kuandeng/LightGCN

LightGCN paper:

Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
Proceedings of SIGIR 2020.

Paper:

https://arxiv.org/abs/2002.02126

The original LightGCN implementation was adapted for the MovieLens-based workflow used in this thesis. Additional scripts were developed for preprocessing, preference extraction, community detection, caching-distribution construction, experiment preparation, and result analysis.

Repository structure

LightGCN_MovieLens_Thesis/
│
├── LightGCN.py
├── setup.py
├── sampling_utils.py
├── alpha_experiment.py
├── cachegain.py
├── dist.py
├── random_item.py
│
├── preprocessing/
│   ├── preprocess_movielens.py
│   └── create_random_movielens_subset.py
│
├── community/
│   ├── louvain.py
│   ├── top_k.py
│   └── change_user_id.py
│
├── community_setup/
│   ├── prep.py
│   └── similarity_com.py
│
├── zipf/
│   ├── com_zipf.py
│   ├── create_topk_probabilities.py
│   ├── make_zipf_probabilities.py
│   └── remove_zero.py
│
├── average/
│   ├── average_folder.py
│   └── calculate_averages.py
│
├── visualization/
│   ├── export_scores_matrix.py
│   ├── global_vs_user_dist.py
│   ├── heatmap.py
│   ├── plot_community_heatmap.py
│   ├── plot_random_users_distribution.py
│   ├── plot_training_results.py
│   ├── visualize_interest_graph.py
│   ├── visualize_lightgcn_bipartite.py
│   ├── visualize_louvain_communities.py
│   └── visualize_user_ranking.py
│
├── utility/
│
└── evaluator/

Data and experiment files

Large datasets, generated CSV files, experiment outputs, and plots are intentionally excluded from GitHub.

The repository is intended to store the source code, while generated data and simulation results remain local.

In particular, directories such as Data/, Shared/, and plots/, together with generated CSV and Excel files, are excluded through .gitignore.

Main components

MovieLens preprocessing

Scripts in preprocessing/ prepare MovieLens interactions for the LightGCN training pipeline.

User preference learning

LightGCN.py trains the recommendation model and produces user-item representations and recommendation scores.

These scores are used to derive a probability distribution over movies for each user.

Community detection

Scripts in community/ and community_setup/ construct user similarity information and apply Louvain community detection.

The detected communities are used to compare cache-placement strategies based on:

individual user preferences;

overall global popularity;

community-level popularity.

Caching experiments

The project contains scripts for preparing probability distributions and experiment configurations used by the coded-caching simulator.

Hybrid cache placement is also considered by combining individual preference and global or community popularity using an alpha parameter.

Visualization and analysis

Scripts in visualization/ and average/ are used to inspect learned user distributions, communities, experiment results, and averaged performance metrics.

Notes

Some scripts use local file paths for experiment inputs and outputs. These paths may need to be changed before running the project on another machine.

Generated datasets and experiment results are not version-controlled in this repository.

Acknowledgments

This work builds on the LightGCN implementation by Xiangnan He, Kuan Deng, Yingxin Wu, and the other contributors to the original LightGCN project.

MovieLens data are provided by the GroupLens Research group at the University of Minnesota.

References

X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang,
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, SIGIR 2020.

F. M. Harper and J. A. Konstan,
The MovieLens Datasets: History and Context, ACM Transactions on Interactive Intelligent Systems, 2015.