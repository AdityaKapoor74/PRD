import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from coma_model import *
import torch.nn.functional as F

class COMAAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		self.env = env
		self.test_num = dictionary["test_num"]
		self.env_name = dictionary["env"]
		self.value_lr = dictionary["value_lr"]
		self.policy_lr = dictionary["policy_lr"]
		self.critic_entropy_pen = dictionary["critic_entropy_pen"]
		self.gamma = dictionary["gamma"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_rew = dictionary["norm_rew"]
		self.gif = dictionary["gif"]
		# TD lambda
		self.lambda_ = dictionary["lambda"]
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		
		self.num_agents = self.env.n_agents
		self.num_actions = self.env.action_space[0].n
		self.num_updates = dictionary["num_updates"]

		self.entropy_pen = dictionary["entropy_pen"]
		self.epsilon_start = dictionary["epsilon_start"]
		self.epsilon = self.epsilon_start
		self.epsilon_end = dictionary["epsilon_end"]
		self.epsilon_episode_steps = dictionary["epsilon_episode_steps"]
		self.episode = 0
		self.target_critic_update = dictionary["target_critic_update"]

		self.enable_grad_clip_actor = dictionary["enable_grad_clip_actor"]
		self.enable_grad_clip_critic = dictionary["enable_grad_clip_critic"]
		self.grad_clip_critic = dictionary["grad_clip_critic"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]
		self.actor_rnn_num_layers = dictionary["actor_rnn_num_layers"]
		self.critic_rnn_num_layers = dictionary["critic_rnn_num_layers"]

		# obs_dim = 2*3 + 1
		self.critic_observation = dictionary["global_observation"]
		self.critic_network = TransformerCritic(self.critic_observation, 64, self.critic_observation+self.num_actions, 64, 64+64, self.num_actions, self.num_agents, self.critic_rnn_num_layers, self.num_actions, self.device).to(self.device)
		self.target_critic_network = TransformerCritic(self.critic_observation, 64, self.critic_observation+self.num_actions, 64, 64+64, self.num_actions, self.num_agents, self.critic_rnn_num_layers, self.num_actions, self.device).to(self.device)
		
		self.target_critic_network.load_state_dict(self.critic_network.state_dict())

		# MLP POLICY
		self.seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(self.seeds[dictionary["iteration"]-1])
		# obs_input_dim = 2*3+1 + (self.num_agents-1)*(2*2+1)
		self.actor_observation = dictionary["local_observation"]
		self.policy_network = RNN_Policy(self.actor_observation, self.num_actions, self.actor_rnn_num_layers, self.num_agents, self.device).to(self.device)


		if dictionary["load_models"]:
			# Loading models
			if torch.cuda.is_available() is False:
				# For CPU
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"],map_location=torch.device('cpu')))
				self.target_critic_network.load_state_dict(torch.load(dictionary["model_path_value"],map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"],map_location=torch.device('cpu')))
			else:
				# For GPU
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.target_critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))

		
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml

	def get_critic_hidden(self, states, one_hot_actions):
		with torch.no_grad():
			states = torch.FloatTensor(states).unsqueeze(0)
			one_hot_actions = torch.FloatTensor(one_hot_actions).unsqueeze(0)
			_, _, rnn_hidden_state_critic = self.critic_network(states.to(self.device), one_hot_actions.to(self.device))
			return rnn_hidden_state_critic.cpu().numpy()

	def get_critic_output(self, states, one_hot_actions, critic_rnn_hidden_state):
		with torch.no_grad():
			states = torch.FloatTensor(states).unsqueeze(0).unsqueeze(0)
			one_hot_actions = torch.FloatTensor(one_hot_actions).unsqueeze(0).unsqueeze(0)
			critic_rnn_hidden_state = torch.FloatTensor(critic_rnn_hidden_state)
			Q_value, _, critic_rnn_hidden_state = self.target_critic_network(states.to(self.device), one_hot_actions.to(self.device), critic_rnn_hidden_state.to(self.device))

			return Q_value.squeeze(0).cpu().numpy(), critic_rnn_hidden_state.cpu().numpy()

	def get_actions(self, state, last_one_hot_actions, rnn_hidden_state, action_masks):
		with torch.no_grad():
			state = torch.FloatTensor(state)
			last_one_hot_actions = torch.FloatTensor(last_one_hot_actions)
			final_state = torch.cat([state, last_one_hot_actions], dim=-1).unsqueeze(0).unsqueeze(0).to(self.device)
			action_masks = torch.BoolTensor(action_masks).unsqueeze(0).unsqueeze(0).to(self.device)
			rnn_hidden_state = torch.FloatTensor(rnn_hidden_state).to(self.device)
			dists, rnn_hidden_state = self.policy_network(final_state, rnn_hidden_state, action_masks)
			action = [Categorical(dist).sample().cpu().detach().item() for dist in dists.squeeze(0).squeeze(0)]
			return dists.squeeze(0).squeeze(0).cpu().numpy(), action, rnn_hidden_state.cpu().numpy()

	def plot(self, episode):
		self.comet_ml.log_metric('Value_Loss',self.plotting_dict["value_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Value',self.plotting_dict["grad_norm_value"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"].item(),episode)

		entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"]* torch.log(torch.clamp(self.plotting_dict["weights_value"], 1e-10,1.0)), dim=2))
		self.comet_ml.log_metric('Critic_Weight_Entropy', entropy_weights.item(), episode)



	def calculate_policy_loss(self, probs, actions, advantage, masks):
		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach() * masks
		policy_loss = policy_loss.sum() / masks.sum()

		return policy_loss

	def update_parameters(self):
		self.episode += 1

		if self.epsilon>self.epsilon_end:
			self.epsilon = self.epsilon_start - self.episode*(self.epsilon_start-self.epsilon_end)/self.epsilon_episode_steps

		if self.episode%self.target_critic_update == 0:
			self.target_critic_network.load_state_dict(self.critic_network.state_dict())


	def build_td_lambda_targets(self, rewards, terminated, mask, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * (1-terminated[:, 1:])
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1,  -1):
			ret[:, t] = self.lambda_ * self.gamma * ret[:, t + 1] + mask[:, t] \
						* (rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t + 1] * (1 - terminated[:, t+1]))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret


	def update(self, buffer, episode):
		
		for i in range(self.num_updates):
			critic_states, critic_rnn_hidden_state, actor_states, actor_rnn_hidden_state, \
			actions, last_one_hot_actions, one_hot_actions, action_masks, masks, target_q_values, advantage = buffer.sample_recurrent_policy()

			probs, _ = self.policy_network(torch.cat([actor_states, last_one_hot_actions], dim=-1).to(self.device), actor_rnn_hidden_state.to(self.device), action_masks.to(self.device))
			
			q_values, weights_value, _ = self.critic_network(critic_states.to(self.device), one_hot_actions.to(self.device), critic_rnn_hidden_state.to(self.device))
			q_values = (q_values.reshape(*one_hot_actions.shape)*one_hot_actions.to(self.device)).sum(dim=-1)
			
			critic_loss = F.huber_loss(q_values*masks.to(self.device), target_q_values.to(self.device)*masks.to(self.device), reduction="sum", delta=10.0) / masks.sum()

			entropy = -torch.sum(probs*masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(probs*masks.unsqueeze(-1).to(self.device), 1e-10,1.0))) / masks.sum()
		
			policy_loss = self.calculate_policy_loss(probs, actions.to(self.device), advantage.to(self.device), masks.to(self.device)) - self.entropy_pen*entropy
			# # ***********************************************************************************
				
			
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			if self.enable_grad_clip_critic:
				grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
			else:
				grad_norm = 0
				for p in self.critic_network.parameters():
					param_norm = p.grad.detach().data.norm(2)
					grad_norm += param_norm.item() ** 2
				grad_norm_value = torch.tensor(grad_norm) ** 0.5
			self.critic_optimizer.step()

			self.target_critic_network.rnn_hidden_state = None
			self.critic_network.rnn_hidden_state = None


			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			if self.enable_grad_clip_actor:
				grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
			else:
				grad_norm = 0
				for p in self.policy_network.parameters():
					param_norm = p.grad.detach().data.norm(2)
					grad_norm += param_norm.item() ** 2
				grad_norm_policy = torch.tensor(grad_norm) ** 0.5
			self.policy_optimizer.step()
			self.policy_network.rnn_hidden_state = None

		self.update_parameters()

		self.plotting_dict = {
		"value_loss": critic_loss,
		"policy_loss": policy_loss,
		"entropy": entropy,
		"grad_norm_value":grad_norm_value,
		"grad_norm_policy": grad_norm_policy,
		"weights_value": weights_value,
		}

		if self.comet_ml is not None:
			self.plot(episode)
