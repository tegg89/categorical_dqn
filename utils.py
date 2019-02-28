import random
import numpy as np

def update_target(model, model_target):
	model_target.load_state_dict(model.state_dict())
	
class ReplayBuffer(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.next_idx = 0

	def push(self, state, action, reward, next_state, done):
		data = (state, action, reward, next_state, done)

		if self.next_idx >= len(self.buffer):
			self.buffer.append(data)
		else:
			self.buffer[self.next_idx] = data
		self.next_idx = (self.next_idx + 1) % self.capacity

	def __len__(self):
		return len(self.buffer)

	def encode_sample(self, idxes):
		obss, acts, rews, nobss, dones = [], [], [], [], []
		for i in idxes:
			data = self.buffer[i]
			obs, act, rew, nobs, done = data
			obss.append(np.array(obs, copy=False))
			acts.append(np.array(act, copy=False))
			rews.append(rew)
			nobss.append(np.array(nobs, copy=False))
			dones.append(done)
		return np.array(obss), np.array(acts), np.array(rews), np.array(nobss), np.array(dones)

	def sample(self, batch_size):
		idxes = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
		return self.encode_sample(idxes)