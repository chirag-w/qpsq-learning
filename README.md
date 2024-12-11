# Learning Quantum Processes with Quantum Statistical Queries

This repository contains code for the paper [Learning Quantum Processes with Quantum Statistical Queries](https://arxiv.org/abs/2310.02075). We include an implementation of our learning algorithm for predicting arbitrary quantum processes using "QPSQs". We also provide an implementation of the similar ML algorithm using classical shadows from [Learning to predict arbitrary quantum processes](https://doi.org/10.1103/PRXQuantum.4.040337) by Huang, Chen and Preskill.

The `learn` method in `coeff.py` takes as input a unitary and an observable, and returns the learned coefficients as a dictionary. `learn` can be used for both HCP's classical shadow algorithm (`flag = 0`) and our QPSQ algorithm (`flag = 1`).

`predict.py` has a method `pred` that takes a dictionary of coefficients (as output by `learn`) and a set of states (density matrices), and returns the prediction for each state.

We generate synthetic QPSQ outputs for our simulations using the `q_statistical_query` method in `dataset.py`. We compare our synthetic data with a classical shadow method for estimating the expectation of an observable for a simple instance in `simulations\obs_error.py`.

We provide simulations of our learning algorithm in the `simulations` folder, where we simulate its performance on a range of unitaries and sets of target states. We include a similar simulation for HCP's algorithm in `main.py`.
