# Enhancing Value Function Estimation through First-Order State-Action Dynamics in Offline Reinforcement Learning

This repo is the official implementations of [Enhancing Value Function Estimation through First-Order State-Action Dynamics in Offline Reinforcement Learning](https://openreview.net/forum?id=nSGnx8lNJ6). International Conference on Machine Learning (ICML) 2024.

Authors: Yun-Hsuan Lien, Ping-Chun Hsieh, Tzu-Mao Li, Yu-Shuen Wang

## Abstract
In offline reinforcement learning (RL), updating the value function with the discrete-time Bellman Equation often encounters challenges due to the limited scope of available data. This limitation stems from the Bellman Equation, which cannot accurately predict the value of unvisited states.  To address this issue, we have introduced an innovative solution that bridges the continuous- and discrete-time RL methods, capitalizing on their advantages. Our method uses a discrete-time RL algorithm to derive the value function from a dataset while ensuring that the function's first derivative aligns with the local characteristics of states and actions, as defined by the Hamilton-Jacobi-Bellman equation in continuous RL. We provide practical algorithms for both deterministic policy gradient methods and stochastic policy gradient methods. Experiments on the D4RL dataset show that incorporating the first-order information significantly improves policy performance for offline RL problems.

