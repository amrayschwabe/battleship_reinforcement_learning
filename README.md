# Coding Challenge
This repository contains my solution to a coding challenge for the deep reinforcement learning seminar at ETH. The goal is to write a deep reinforcement learning algorithm that maximizes the given reward for destroying the battleships as fast as possible. It contains two approaches, a dqn implementation which runs but does not learn and a policy gradient which works well.

## How to run
To train a model from scratch, run policyGradient.py with TRAIN = TRUE and LOAD = FALSE. To just evaluate a trained model, run the file with TRAIN = FALSE, LOAD = TRUE. To load a pretrained model and further train it, run it with TRAIN = TRUE, LOAD = TRUE. Make sure you add the right folder path to train the model. One pretrained model can be found in the models folder. 
