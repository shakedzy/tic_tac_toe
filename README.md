# Tic Tac Toe played by Double Deep Q-Networks
![tic_tac_toe](images/tic_tac_toe.png)
 
 This repository contains a (successful) attempt to train a Reinforcement Learning
 agent to play Tic-Tac-Toe. It learned to:
 
 * Distinguish valid from invalid moves
 * Comprehend how to win a game
 * Block the opponent when poses a threat
 
 
## Training:
The ML code being used is imported from the [warehouse](https://github.com/shakedzy/warehouse) library I wrote. 
The agent is based on a [Double Deep Q-Network model](https://arxiv.org/abs/1509.02971v5). 

Two types of agents were trained: a regular DDQN agent, and another which learns using 
[maximum entropy](https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/). They are named _'Q'_ and _'E'_ 
respectively.  


## DDQN key formulas:
The cost function used is:

![cost](images/ddqn_cost.png)

Where θ represents the trained Q-Network and ϑ represents the semi-static Q-Target
network.

The Q-Target update rule is:

![update_rule](images/ddqn_update.png)

for some 0 <= τ <= 1.


## Do it yourself:
The `main.py` holds several useful functions. See doc-strings for more details:
* `train` will initiate a single training process. It will save the weights and plots graphs. 
Using the current settings, training took me around 70 minutes on a 2018 MacBook Pro
* `multi_train` will train several DDQN and DDQN-Max-Entropy models
* `play` allows a human player to play against a saved model
* `face_off` can be used to compare models by letting them play against each other

## Out-of-the-box models:
The `models/` directory holds several trained models. _Q_ files refer to DDQN models and _E_ files refer to 
DDQN-Max-Entropy models 

---------------------------

### Related blogposts:
Read about where I got stuck when developing this code on "[Lessons Learned from Tic-Tac-Toe: Practical Reinforcement Learning Tips](https://medium.com/@shakedzy/lessons-learned-from-tic-tac-toe-practical-reinforcement-learning-tips-5cac654a45a8)"
