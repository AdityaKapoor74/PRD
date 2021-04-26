## Table of contents
* [Partial Reward Decoupling](#general-info)
* [Architecture](#architecture)
* [Technologies](#technologies)
* [Environment](#environment)
* [Methodology](#methodology)


## Partial Reward Decoupling
Here we are tackling the credit assignment problem (internally) pertaining to multi-agent reinforcement learning. The idea is to learn to recognize the relevant set of agents, in a very large system, whose actions have a direct influence over their rewards. In short, the aim is to determine which actions taken by agents in a large group of cooperating agents helped the group achieve its goal and hence learn more rapidly.

## Architecture
We are using Advantage Actor Critic Algorithm to test out our hypothesis with a little tweaks while calculating the Values and Advantage function.
* Critic Network - GAT Network (Preprocess the Observations) + GAT Network (to calculate the weights that are used to calculate the z values) + FCN (to calculate the Values)
* Policy Network - FCN

<!-- ARCHITECTURE 1 -->
### Architecture 1

In this methodology we have 3 networks for the Critic Architecture that are/ stacked together:-
  1. Scalar Dot Product Attention to do the preprocessing of the observations of every agent with a message passing round (we use raw observations to calculate the weights) (SOFTMAX to calculate the weights)
  2. Scalar Dot Product Attention to calculate the z values (We use raw observations to calculate the weights; sometimes a small subset of the raw observations is also used) (SIGMOID to calculate the weights so that there is a pair-wise dependency)
  3. Fully Connected Network to calculate the Value estimates for an agent i not conditioned on agent j's actions; output is a NxN matrix where N is the number of agents

<!-- ARCHITECTURE 2 -->
### Architecture 2

In this methodology we have 3 networks for the Critic Architecture that are stacked together:-
  1. Scalar Dot Product Attention to calculate the z values (We use raw observations to calculate the weights) (SIGMOID to calculate the weights so that there is a pair-wise dependency)
  2. Scalar Dot Product Attention to do the message passing of the observations of every agent to every other agent; sharing of information (we use raw observations concatenated with z values to calculate the weights) (SOFTMAX to calculate the weights)
  3. Fully Connected Network to calculate the Value estimates for an agent i not conditioned on agent j's actions; output is a NxN matrix where N is the number of agents


<!-- ARCHITECTURE 3 -->
### Architecture 3

In this methodology we have 2 networks for the Critic Architecture that are stacked together:-
  1. Scalar Dot Product Attention to do the message passing of the observations of every agent to every other agent; sharing of information (we use raw observations concatenated with action values to calculate the weights) (SIGMOID/SOFTMAX to calculate the weights)
  2. We calculate attention values using {states,actions}. We further encode the source node's observations too
  3. Node features = {encoded state, aggregation(weight x attention values)}
  2. Fully Connected Network to calculate the Value estimates for an agent i not conditioned on agent j's actions; output is a NxN matrix where N is the number of agents


<!-- ARCHITECTURE 4 -->
### Architecture 4

In this methodology we have 2 networks for the Critic Architecture that are stacked together:-
  1. Scalar Dot Product Attention to do the message passing of the observations of every agent to every other agent; sharing of information (we use raw observations to calculate the weights) (SIGMOID/SOFTMAX to calculate the weights)
  2. Z = actions x weights + noise (Uniform/Gaussian)
  3. Node feature = {states, aggregation(states,z)}
  4. Fully Connected Network to calculate the Value estimates using node features for an agent i not conditioned on agent j's actions; output is a NxN matrix where N is the number of agents


<!-- ARCHITECTURE 5 -->
### Architecture 5

In this methodology we have 2 networks for the Critic Architecture that are stacked together:-
  1. Scalar Dot Product Attention to do the message passing of the observations of every agent to every other agent; sharing of information (we use raw observations to calculate the weights) (SOFTMAX to calculate the weights)
  2. We calculate attention values using {states,actions}.
  3. Node features = {state, aggregation((weight x attention values) + noise)}
  4. Fully Connected Network to calculate the Value estimates using node features for an agent i not conditioned on agent j's actions; output is a NxN matrix where N is the number of agents


NOTE: Decoupled files consist of 2 Critic Networks, 1 to get the Value Estimates and the other to calculate the Weights for Z(s). For the Critic, to calculate the Value Estimates, we use the Monte Carlo (discounted returns) target for calculating the loss. For the other Critic, we use TD-error to calculate the loss; in some experiments we also use to calculate the immediate reward.



## Environment
The environment comprises of "N" agents and "N" goal positions. Every agent is assigned a goal position and an agent that it is paired with. The reward function is such that every agent recieves a sum of the penalties (L2 distance from the goal position) of the agent it is paired with and itself. Along with this, there is a collision penalty as well which is set to -1 if the the agent or its paired partner collides with any other agent. The goal of the environment is that all "N" agents need to reach their goal positions.
* Reward Function = L2_dist(paired_agent_pose,paired_agent_goal_pose) + L2_dist(pose, goal_pose) + collision_penalty_paired_agent + collision
* Observation Space for the Critic Network = {Position, Velocity, Goal Position, Paired Agent's Goal Position} 
* Observation Space for the Actor Network = {Position, Velocity, Other Agent's Relative Position, Other Agent's Relative Velocity}

## Methodology
We construct a fully connected graph with self loops for the Critic Network
For every agent we calculate a matrix of values in the following manner:-
For every timestep for every agent, the Value Matrix will have a dimension of "NxN" where N is the number of agents.
We do this to calculate advantages in a particular way
V[i,j] = Estimated return of the rewards of agent i without conditioning on agent j's actions
A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i]) (This ensures that we do not condition the baseline (Advantage function) for agent i on agent i's actions)

Note: In this environment we do not use a binary variable to indicate pairing but the pairing is apparent as the observations of the paired agents have both their goal positions (itself/paired parnter)