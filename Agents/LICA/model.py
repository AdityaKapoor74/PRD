import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
	def __init__(self, obs_input_dim, rnn_hidden_dim, num_actions):
		super(RNNAgent, self).__init__()

		self.layer_1 = nn.Linear(obs_input_dim + num_actions, rnn_hidden_dim)
		self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
		self.layer_2 = nn.Linear(rnn_hidden_dim, num_actions)

		self.rnn_hidden_obs = None

	def init_hidden(self):
		gain = nn.init.calculate_gain('relu')
		nn.init.xavier_uniform_(self.layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.layer_2.weight)

	def forward(self, input_obs):
		x = F.relu(self.layer_1(input_obs))
		self.rnn_hidden_obs = self.rnn(x, self.rnn_hidden_obs)
		q = self.layer_2(self.rnn_hidden_obs)
		return q


class LICACritic(nn.Module):
	def __init__(self, obs_input_dim, mixing_embed_dim, num_actions, num_agents, num_hypernet_layers):
		super(LICACritic, self).__init__()

		self.num_actions = num_actions
		self.num_agents = num_agents

		self.state_dim = obs_input_dim * self.num_agents
		self.embed_dim = self.num_actions * self.num_agents * mixing_embed_dim
		self.hidden_dim = mixing_embed_dim

		if num_hypernet_layers == 1:
			self.hyper_w1 = nn.Linear(self.state_dim, self.embed_dim)
			self.hyper_w2 = nn.Linear(self.state_dim, self.embed_dim)
		elif num_hypernet_layers == 2:
			self.hyper_w1 = nn.Sequential(
				nn.Linear(self.state_dim, self.hidden_dim),
				nn.ReLU(),
				nn.Linear(self.hidden_dim, self.embed_dim))
			self.hyper_w2 = nn.Sequential(
				nn.Linear(self.state_dim, self.hidden_dim),
				nn.ReLU(),
				nn.Linear(self.hidden_dim, self.hidden_dim))

		self.hyper_b1 = nn.Linear(self.state_dim, self.hidden_dim)

		self.hyper_b2 = nn.Sequential(
			nn.Linear(self.state_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, 1))


	def forward(self, actions, states):
		bs = states.shape[0]
		states = states.reshape(-1, self.state_dim)
		action_probs = actions.reshape(-1, 1, self.num_agents*self.num_actions)

		w1 = self.hyper_w1(states)
		b1 = self.hyper_b1(states)
		w1 = w1.view(-1, self.num_agents * self.num_actions, self.hidden_dim)
		b1 = b1.view(-1, 1, self.hidden_dim)

		h = torch.relu(torch.bmm(action_probs, w1) + b1)

		w_final = self.hyper_w2(states)
		w_final = w_final.view(-1, self.hidden_dim, 1)

		h2 = torch.bmm(h, w_final)

		b2 = self.hyper_b2(states).view(-1, 1, 1)

		q = h2 + b2

		q = q.view(bs, -1, 1)

		return q



