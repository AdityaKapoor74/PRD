## Table of contents
* [Partial Reward Decoupling](#general-info)
* [Architecture](#architecture)
* [Technologies](#technologies)
* [Environment](#environment)
* [Methodology](#methodology)
* [Performance Measures](#peroformance)
* [Future Work](#futurework)


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

Note: In both the GATs we use a binary variable (1/-1) while calculating the attention (for GAT 1) and weight (for GAT 2) to specify the kind of relationship between agents (paired/unpaired)


## Performance Measures
* When we use the processed observations to calculate weight values, the contrast between paired and unpaired agent is not of significance. However when in the first GAT we replace the softmax with a sigmoid, the contrast seems a little apparent (Softmax gives a relative weight measure and it can so happen that the weight values are very low so the information transfer is less. On the other hand, if one uses sigmoid, the weight values are independent of other agent's features so the information transfer is much better. In short, the method of weightage assignment makes things better for sigmoid over softmax)
* The best performance is achieved when we use the raw observation data to calculate the weight values. The weights assigned to paired agents move to 1 and the unpaired agents move to 0. This is what we had hypothesized.
* The baseline was to use just a binary value that indicates if the agents are paired or unpaired to calculate the weight values for z calculations that gives a similar performance to the experiment when raw observations were used for such calculations.

## Future Work
We need to come up with an environmental setup such that we do not make the pairings as obvious (by using a binary indicator) and test our hypothesis.
