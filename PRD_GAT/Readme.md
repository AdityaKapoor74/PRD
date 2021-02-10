## Table of contents
* [Partial Reward Decoupling](#general-info)
* [Architecture](#architecture)
* [Technologies](#technologies)
* [Environment](#environment)


## Partial Reward Decoupling
Here we are tackling the credit assignment problem (internally) pertaining to multi-agent reinforcement learning. The idea is to learn to recognize the relevant set of agents, in a very large system, whose actions have a direct influence over their rewards. In short, the aim is to determine which actions taken by agents in a large group of cooperating agents helped the group achieve its goal and hence learn more rapidly.

## Architecture
We are using Advantage Actor Critic Algorithm to test out our hypothesis with a little tweaks while calculating the Values and Advantage function.
* Critic Network - GAT Network (Preprocess the Observations) + GAT Network (to calculate the weights that are used to calculate the z values) + FCN (to calculate the Values)
* Policy Network - FCN

## Environment
The environment comprises of "N" agents and "N" goal positions. Every agent is assigned a goal position and an agent that it is paired with. The reward function is such that every agent recieves a penalty (L2 distance from the goal position) of the agent it is paired with. Along with this, there is a collision penalty as well which is set to -1 if the paired agent collides with any other agent. The goal of the environment is that all "N" agents need to reach their goal positions.
* Reward Function = L2_dist(paired_agent_pose,paired_agent_goal_pose) + collision_penalty_paired_agent
* Observation Space for the Critic Network = {Position, Velocity, Goal Position} 
* Observation Space for the Actor Network = {Position, Velocity, Other Agent's Relative Position, Other Agent's Relative Velocity}

## Methodology
We construct a fully connected graph with self loops for the Critic Network
For every agent we calculate a matrix of values in the following manner:-
For every timestep for every agent, the Value Matrix will have a dimension of "NxN" where N is the number of agents.
We do this to calculate advantages in a particular way
V[i,j] = Estimated return of the rewards of agent i without conditioning on agent j's actions
A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i]) (This ensures that we do not condition the baseline (Advantage function) for agent i on agent i's actions)

To go over the pipeline:
* GAT NETWORK 1 (Preprocess): We preprocess the observations using a GAT
* GAT NETWORK 2 (Calculate weights used for z calculations): We use either the preprocessed observations to calculate the weight values which are further used to calculate z values (z = weight x actions + weight x policies)
* FCN (Calculate V matrix): We finally concatenate the processed observations with the z values to calculate the Value Matrix for every agent for every timestep