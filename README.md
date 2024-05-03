# Enhancing Value Function Estimation through First-Order State-Action Dynamics in Offline Reinforcement Learning

This repo is the official implementations of [Enhancing Value Function Estimation through First-Order State-Action Dynamics in Offline Reinforcement Learning](https://github.com/sophialien/DifferentiableOfflineRL/blob/main/%5Bpaper%5D%20Enhancing%20Value%20Function%20Estimation.pdf). International Conference on Machine Learning (ICML) 2024.

Authors: Yun-Hsuan Lien, Ping-Chun Hsieh, Tzu-Mao Li, Yu-Shuen Wang

## Abstract
In offline reinforcement learning (RL), updating the value function with the discrete-time Bellman Equation often encounters challenges due to the limited scope of available data. This limitation stems from the Bellman Equation, which cannot accurately predict the value of unvisited states.  To address this issue, we have introduced an innovative solution that bridges the continuous- and discrete-time RL methods, capitalizing on their advantages. Our method uses a discrete-time RL algorithm to derive the value function from a dataset while ensuring that the function's first derivative aligns with the local characteristics of states and actions, as defined by the Hamilton-Jacobi-Bellman equation in continuous RL. We provide practical algorithms for both deterministic policy gradient methods and stochastic policy gradient methods. Experiments on the D4RL dataset show that incorporating the first-order information significantly improves policy performance for offline RL problems.



<p align="center">
  <img src="https://github.com/sophialien/DifferentiableOfflineRL/blob/main/first_order/First_Order.png" width="1000" />
</p>

---
If you find this work useful in your research, please consider citing:
```
@inproceedings{lien2024firstorder,
 author={Yun-Hsuan Lien, Ping-Chun Hsieh, Tzu-Mao Li, Yu-Shuen Wang},
 booktitle={International Conference on Machine Learning (ICML)},
 year={2024}
 }
```

## Contact Information
If you have any questions, please contact Sophia Yun-Hsuan Lien: sophia.yh.lien@gmail.com
