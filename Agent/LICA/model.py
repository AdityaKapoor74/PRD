import torch
import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
	weight_init(module.weight.data, gain=gain)
	if module.bias is not None:
		bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class RNNAgent(nn.Module):
	def __init__(self, obs_input_dim, rnn_num_layers, rnn_hidden_dim, num_actions):
		super(RNNAgent, self).__init__()

		# self.layer_1 = nn.Linear(obs_input_dim + num_actions, rnn_hidden_dim)
		# self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
		# self.layer_2 = nn.Linear(rnn_hidden_dim, num_actions)

		# self.rnn_hidden_obs = None

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)
		self.rnn_num_layers = rnn_num_layers

		self.Layer_1 = nn.Sequential(
			init_(nn.Linear(obs_input_dim, rnn_hidden_dim), activate=True),
			nn.GELU(),
			nn.LayerNorm(rnn_hidden_dim),
			)
		# self.RNN = nn.GRUCell(hidden_dim, hidden_dim)
		self.RNN = nn.GRU(input_size=rnn_hidden_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers, batch_first=True)
		# self.Layer2 = nn.Linear(hidden_dim, num_actions)
		self.Layer_2 = nn.Sequential(
			nn.LayerNorm(rnn_hidden_dim),
			init_(nn.Linear(rnn_hidden_dim, num_actions), gain=0.01)
			)


	# def init_hidden(self):
	# 	# gain = nn.init.calculate_gain('relu')
	# 	nn.init.xavier_uniform_(self.layer_1.weight)
	# 	nn.init.xavier_uniform_(self.layer_2.weight)

	# def forward(self, input_obs):
	# 	x = F.gelu(self.layer_1(input_obs))
	# 	self.rnn_hidden_obs = self.rnn(x, self.rnn_hidden_obs)
	# 	q = self.layer_2(self.rnn_hidden_obs)
	# 	return q

	def forward(self, states_actions, hidden_state, action_masks):
		batch, timesteps, num_agents, _ = states_actions.shape

		intermediate = self.Layer_1(states_actions)
		intermediate = intermediate.permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
		hidden_state = hidden_state.reshape(self.rnn_num_layers, batch*num_agents, -1)
		output, h = self.RNN(intermediate, hidden_state)
		output = output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3)
		logits = self.Layer_2(output)

		logits = torch.where(action_masks, logits, self.mask_value)
		
		return logits, h


class LICACritic(nn.Module):
	def __init__(self, obs_input_dim, mixing_embed_dim, num_actions, num_agents, num_hypernet_layers):
		super(LICACritic, self).__init__()

		self.num_actions = num_actions
		self.num_agents = num_agents

		self.state_dim = obs_input_dim
		self.embed_dim = self.num_actions * self.num_agents * mixing_embed_dim
		self.hidden_dim = mixing_embed_dim

		if num_hypernet_layers == 1:
			self.hyper_w1 = init_(nn.Linear(self.state_dim, self.embed_dim))
			self.hyper_w2 = init_(nn.Linear(self.state_dim, self.embed_dim))
		elif num_hypernet_layers == 2:
			self.hyper_w1 = nn.Sequential(
				init_(nn.Linear(self.state_dim, self.hidden_dim), activate=True),
				nn.ReLU(),
				init_(nn.Linear(self.hidden_dim, self.embed_dim), activate=True))
			self.hyper_w2 = nn.Sequential(
				init_(nn.Linear(self.state_dim, self.hidden_dim), activate=True),
				nn.ReLU(),
				init_(nn.Linear(self.hidden_dim, self.hidden_dim), activate=True))

		self.hyper_b1 = init_(nn.Linear(self.state_dim, self.hidden_dim))

		self.hyper_b2 = nn.Sequential(
			init_(nn.Linear(self.state_dim, self.hidden_dim), activate=True),
			nn.ReLU(),
			init_(nn.Linear(self.hidden_dim, 1), activate=True))


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



