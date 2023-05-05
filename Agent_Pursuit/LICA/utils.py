import math
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical

			
class GumbelSoftmax(OneHotCategorical):

	def __init__(self, logits, probs=None, temperature=1):
		super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
		self.eps = 1e-20
		self.temperature = temperature

	def sample_gumbel(self):
		U = self.logits.clone()
		U.uniform_(0, 1)
		return -torch.log( -torch.log( U + self.eps ) )

	def gumbel_softmax_sample(self):
		y = self.logits + self.sample_gumbel()
		return torch.softmax( y / self.temperature, dim=-1)

	def hard_gumbel_softmax_sample(self):
		y = self.gumbel_softmax_sample()
		return (torch.max(y, dim=-1, keepdim=True)[0] == y).float()

	def rsample(self):
		return self.gumbel_softmax_sample()

	def sample(self):
		return self.rsample().detach()

	def hard_sample(self):
		return self.hard_gumbel_softmax_sample()


def multinomial_entropy(logits):
	assert logits.size(-1) > 1
	return GumbelSoftmax(logits=logits).entropy()



def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)