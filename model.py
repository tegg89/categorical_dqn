import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalDQN(nn.Module):
	def __init__(self, num_inputs, num_actions, args):
		super(CategoricalDQN, self).__init__()
		self.num_inputs = num_inputs
		self.num_actions = num_actions
		self.num_atoms = args.atom
		self.vmax = args.vmax
		self.vmin = args.vmin

		self.linear1 = nn.Linear(num_inputs, args.hidden_size//4)
		self.linear2 = nn.Linear(args.hidden_size//4, args.hidden_size)
		self.linear3 = nn.Linear(args.hidden_size, num_actions * args.atom)

	def forward(self, input):
		x = F.relu(self.linear1(input))
		x = F.relu(self.linear2(x))
		x = F.relu(self.linear3(x))
		x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
		return x

	def act(self, state):
		with torch.no_grad():
			state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
		dist = self.forward(state).data.cpu() # [1, 2, 51]
		dist = dist * torch.linspace(self.vmin, self.vmax, self.num_atoms)
		action = dist.sum(2).max(1)[1].numpy()[0]
		return action
