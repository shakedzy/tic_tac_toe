# Tic Tac Toe played by Double Deep Q-Networks
![tic_tac_toe](images/tic_tac_toe.png)
 
 This repository contains a (successful) attempt to train a Reinforcement Learning
 agent to play Tic-Tac-Toe. It learned to:
 
 * Distinguish valid from invalid moves
 * Comprehend how to win a game
 * Block the opponent when poses a threat
 
 The agent is based on a [Double Deep Q-Network model](https://arxiv.org/abs/1509.02971v5), 
 and is implemented with Python 3 and Tensorflow.

## DDQN key formulas:
The cost function used is:

![cost](images/ddqn_cost.png)

Where θ represents the trained Q-Network and ϑ represents the semi-static Q-Target
network.

The Q-Target update rule is:

![update_rule](images/ddqn_update.png)

for some 0 <= τ <= 1.


## Training and playing:
The `main.py` holds two functions:
* `train()` will initiate a training process, and save the model under
`models/q.ckpt`. Using the current settings, training took me less than 20 
minutes on a MacBook Pro (2018)
* `play()` allows a human player to play against a saved model

### out-of-the-box model:
`models/q.ckpt` is a model trained by me using the configurations
in the code. 

---------------------------

### Related blogposts:
Read about where I got stuck when developing this code on "[Lessons Learned from Tic-Tac-Toe: Practical Reinforcement Learning Tips](https://medium.com/@shakedzy/lessons-learned-from-tic-tac-toe-practical-reinforcement-learning-tips-5cac654a45a8)"
