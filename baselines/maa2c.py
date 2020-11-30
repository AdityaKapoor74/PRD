import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from a2c_agent import A2CAgent


class MAA2C:

	def __init__(self,env,gif=True):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.env = env
		self.gif = gif
		self.num_agents = env.n
		self.num_actions = self.env.action_space[0].n
		self.one_hot_actions = torch.zeros(self.env.action_space[0].n,self.env.action_space[0].n)
		for i in range(self.env.action_space[0].n):
			self.one_hot_actions[i][i] = 1

		self.agents = A2CAgent(self.env, gif = self.gif)

		if not(self.gif):
			self.writer = SummaryWriter('../../runs/baselines/4_agents/value_2_layers_lr_2e-4_policy_lr_2e-4_entropy_0.008_policy')

	def get_actions(self,states):
		actions = []
		for i in range(self.num_agents):
			action = self.agents.get_action(states[i])
			actions.append(action)
		return actions

	def update(self,trajectory,episode):


		states_critic = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
		states_actor = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)
		actions = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
		rewards = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
		dones = torch.FloatTensor([sars[4] for sars in trajectory])

		value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy = self.agents.update(states_critic,states_actor,actions,rewards,dones)


		if not(self.gif):
			self.writer.add_scalar('Loss/Entropy loss',entropy,episode)
			self.writer.add_scalar('Loss/Value Loss',value_loss,episode)
			self.writer.add_scalar('Loss/Policy Loss',policy_loss,episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)



	def split_states(self,states):

		states_critic = []
		states_actor = []
		for i in range(self.num_agents):
			states_critic.append(states[i][0])
			states_actor.append(states[i][1])

		states_critic = np.asarray(states_critic)
		states_actor = np.asarray(states_actor)

		return states_critic,states_actor




	def run(self,max_episode,max_steps):  
		for episode in range(1,max_episode):
			states = self.env.reset()

			states_critic,states_actor = self.split_states(states)


			trajectory = []
			episode_reward = 0
			
			for step in range(max_steps):

				'''
				agents observation with actions concatenated
				Number of Agents  x Observation Space (includes actions for agents other than the agent in question)
				'''

				actions = self.get_actions(states_actor)
				policies = self.agents.policy_network.forward(torch.FloatTensor(states_actor).to(self.device)).detach().cpu().numpy()
				
				# other_actions = np.zeros((self.num_agents,self.num_actions))
				# for i,act in enumerate(actions):
				# 	other_actions[i][act] = 1
				# other_actions = other_actions+policies

				states_action_critic = np.zeros((self.num_agents,self.agents.value_input_dim))
				for i in range(self.num_agents):
					# other_actions_temp = np.delete(other_actions,i,axis=0)
					policies_temp = np.delete(policies,i,axis=0)
					tmp = np.copy(states_critic[i])
					for j in range(self.num_agents-1):
						if j == self.num_agents-2:
							# tmp = np.concatenate([tmp,other_actions_temp[j]])
							tmp = np.concatenate([tmp,policies_temp[j]])

							continue
						# tmp = np.concatenate([tmp[:-6*(self.num_agents-j-2)],other_actions_temp[j],tmp[-6*(self.num_agents-j-2):]])
						tmp = np.concatenate([tmp[:-6*(self.num_agents-j-2)],policies_temp[j],tmp[-6*(self.num_agents-j-2):]])
					states_action_critic[i] = tmp


				next_states,rewards,dones,info = self.env.step(actions)
				next_states_critic,next_states_actor = self.split_states(next_states)

				episode_reward += np.sum(rewards)


				if all(dones) or step == max_steps-1:

					dones = [1 for _ in range(self.num_agents)]
					trajectory.append([states_action_critic,states_actor,actions,rewards,dones])
					print("*"*100)
					print("EPISODE: {} | REWARD: {} \n".format(episode,np.round(episode_reward,decimals=4)))
					print("*"*100)

					if not(self.gif):
						self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
						self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)

					break
				else:
					dones = [0 for _ in range(self.num_agents)]
					trajectory.append([states_action_critic,states_actor,actions,rewards,dones])
					states_critic,states_actor = next_states_critic,next_states_actor
					states = next_states

			#       make a directory called models
			if not(episode%100) and episode!=0 and not(self.gif):
				torch.save(self.agents.value_network.state_dict(), "../../models/baselines/4_agents/value_net_2_layered_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_policy.pt")
				torch.save(self.agents.policy_network.state_dict(), "../../models/baselines/4_agents/policy_net_lr_2e-4_value_2_layered_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_policy.pt")  


			self.update(trajectory,episode) 

