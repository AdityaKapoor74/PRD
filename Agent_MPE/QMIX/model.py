import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize weights
def weights_init(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)


class RNNQNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_dim):
		super(RNNQNetwork, self).__init__()

		self.rnn_hidden_state = None
		self.Layer1 = nn.Linear(num_inputs + num_actions, hidden_dim)
		self.RNN = nn.GRUCell(hidden_dim, hidden_dim)
		self.Layer2 = nn.Linear(hidden_dim, num_actions)

		self.apply(weights_init)

	def forward(self, states_actions):

		x = F.relu(self.Layer1(states_actions))
		self.rnn_hidden_state = self.RNN(x, self.rnn_hidden_state)
		Q_a_values = self.Layer2(self.rnn_hidden_state)
		return Q_a_values


class QMIXNetwork(nn.Module):
	def __init__(self, num_agents, hidden_dim, total_obs_dim):
		super(QMIXNetwork, self).__init__()
		self.num_agents = num_agents
		self.hidden_dim = hidden_dim

		self.hyper_w1 = nn.Linear(total_obs_dim, num_agents * hidden_dim)
		self.hyper_b1 = nn.Linear(total_obs_dim, hidden_dim)
		self.hyper_w2 = nn.Linear(total_obs_dim, hidden_dim)
		self.hyper_b2_l1 = nn.Linear(total_obs_dim, hidden_dim)
		self.hyper_b2_l2 = nn.Linear(hidden_dim, 1)

		self.apply(weights_init)

	def forward(self, q_values, total_obs):
		q_values = q_values.reshape(-1, 1, self.num_agents)
		w1 = torch.abs(self.hyper_w1(total_obs))
		b1 = self.hyper_b1(total_obs)
		w1 = w1.reshape(-1, self.num_agents, self.hidden_dim)
		b1 = b1.reshape(-1, 1, self.hidden_dim)

		x = F.elu(torch.bmm(q_values, w1) + b1)

		w2 = torch.abs(self.hyper_w2(total_obs))
		b2 = self.hyper_b2_l2(F.relu(self.hyper_b2_l1(total_obs)))
		w2 = w2.reshape(-1, self.hidden_dim, 1)
		b2 = b2.reshape(-1, 1, 1)

		x = torch.bmm(x, w2) + b2

		return x