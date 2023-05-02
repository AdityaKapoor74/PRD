import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize weights
def weights_init(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Conv2d):
		nn.init.orthogonal_(m.weight)


class RNNQNetwork(nn.Module):
	def __init__(self, height, width, in_channel, num_agents, num_actions, hidden_dim, device):
		super(RNNQNetwork, self).__init__()

		self.num_agents = num_agents
		self.rnn_hidden_state = None
		self.device = device
		self.obs_input_dim = height*width*in_channel
		scale = (height*width) ** -0.5
		self.positions = nn.Parameter(scale * torch.randn(self.num_agents, self.obs_input_dim)) # Num Patches, embedding size
		self.Conv = nn.Conv2d(in_channel, self.obs_input_dim, height)
		self.Layer1 = nn.Linear(self.obs_input_dim + num_actions, hidden_dim)
		self.RNN = nn.GRUCell(hidden_dim, hidden_dim)
		self.Layer2 = nn.Linear(hidden_dim, num_actions)

		self.apply(weights_init)

	def forward(self, states, actions):
		states = self.Conv(states).reshape(-1, self.obs_input_dim)+self.positions
		states_actions = torch.cat([states, actions], dim=-1).to(self.device)
		x = F.relu(self.Layer1(states_actions))
		self.rnn_hidden_state = self.RNN(x, self.rnn_hidden_state)
		Q_a_values = self.Layer2(self.rnn_hidden_state)
		return Q_a_values


class QMIXNetwork(nn.Module):
	def __init__(self, height, width, in_channel, num_agents, hidden_dim):
		super(QMIXNetwork, self).__init__()
		self.num_agents = num_agents
		self.hidden_dim = hidden_dim

		self.obs_input_dim = height*width*in_channel

		self.Conv_1 = nn.Conv2d(in_channel, self.obs_input_dim, height)
		self.hyper_w1 = nn.Sequential(
			nn.Linear(self.obs_input_dim*self.num_agents, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, num_agents * hidden_dim)
			)
		self.hyper_b1 = nn.Sequential(
			nn.Linear(self.obs_input_dim*self.num_agents, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, hidden_dim)
			)

		self.Conv_2 = nn.Conv2d(in_channel, self.obs_input_dim, height)
		self.hyper_w2 = nn.Sequential(
			nn.Linear(self.obs_input_dim*self.num_agents, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, hidden_dim)
			)
		self.hyper_b2 = nn.Sequential(
			nn.Linear(self.obs_input_dim*self.num_agents, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, 1)
			)

		# self.apply(weights_init)

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""

		# EMBEDDINGS
		nn.init.xavier_uniform_(self.hyper_w1[0].weight)
		nn.init.xavier_uniform_(self.hyper_w1[2].weight)

		nn.init.xavier_uniform_(self.hyper_b1[0].weight)
		nn.init.xavier_uniform_(self.hyper_b1[2].weight)

		nn.init.xavier_uniform_(self.hyper_w2[0].weight)
		nn.init.xavier_uniform_(self.hyper_w2[2].weight)

		nn.init.xavier_uniform_(self.hyper_b2[0].weight)
		nn.init.xavier_uniform_(self.hyper_b2[2].weight)

	def forward(self, q_values, images):
		total_obs = self.Conv_1(images).reshape(-1, self.obs_input_dim*self.num_agents)
		q_values = q_values.reshape(-1, 1, self.num_agents)
		w1 = torch.abs(self.hyper_w1(total_obs))
		b1 = self.hyper_b1(total_obs)
		w1 = w1.reshape(-1, self.num_agents, self.hidden_dim)
		b1 = b1.reshape(-1, 1, self.hidden_dim)

		x = F.elu(torch.bmm(q_values, w1) + b1)

		total_obs = self.Conv_2(images).reshape(-1, self.obs_input_dim*self.num_agents)
		w2 = torch.abs(self.hyper_w2(total_obs))
		b2 = self.hyper_b2(total_obs)
		w2 = w2.reshape(-1, self.hidden_dim, 1)
		b2 = b2.reshape(-1, 1, 1)

		x = torch.bmm(x, w2) + b2

		return x