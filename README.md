# Learning Quantum Processes with Quantum Statistical Queries
Implementation of our learning algorithm for predicting arbitrary quantum processes using QPSQs
We also provide an implementation of the similar ML algorithm using classical shadows from "Learning to predict arbitrary quantum processes" by Hsin-Yuan Huang, Sitan Chen and John Preskill

The `learn` method in `coeff.py` takes as input a unitary and an observable, and returns the learned coefficients as a dictionary. `learn` can be used for both Huang's classical shadow algorithm (`flag = 0`) and our QPSQ algorithm (`flag` = 1)

`predict.py` has a method `pred` that takes a dictionary of coefficients (as output by `learn`) and a set of states (density matrices), and returns the prediction for each state

We provide an example of using our algorithms in `main.py`, where we simulate the performance of Huang's algorithm on a range of unitaries and sets of target states. We provide similar simulations for our algorithm in the `simulations` folder.

