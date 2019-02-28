import numpy as np
import torch
import torch.optim as optim

from utils import ReplayBuffer, update_target
from model import CategoricalDQN

def train(args, env):
	model = CategoricalDQN(env.observation_space.shape[0], env.action_space.n, args)
	model_target = CategoricalDQN(env.observation_space.shape[0], env.action_space.n, args)
	update_target(model, model_target)

	replay_buffer = ReplayBuffer(args.memory_capacity)

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

	def project_dist(next_state, rewards, dones):
		delta_z = float(args.vmax - args.vmin) / (args.atom - 1)
		support = torch.linspace(args.vmin, args.vmax, args.atom)

		next_dist = model_target(next_state).data.cpu() * support
		next_action = next_dist.sum(2).max(1)[1]
		next_action = next_action.unsqueeze(1).unsqueeze(1).expand(args.batch_size, 1, args.atom)
		next_dist = next_dist.gather(1, next_action).squeeze(1)

		rewards = rewards.unsqueeze(1).expand_as(next_dist)
		dones = dones.unsqueeze(1).expand_as(next_dist)
		support = support.unsqueeze(0).expand_as(next_dist)

		Tz = rewards + (1 - dones) * args.discount * support
		Tz = Tz.clamp(min=args.vmin, max=args.vmax)
		b = (Tz - args.vmin) / delta_z
		l = b.floor().long()
		u = b.ceil().long()

		offset = torch.linspace(0, (args.batch_size - 1) * args.atom, args.batch_size).long()\
					.unsqueeze(1).expand(args.batch_size, args.atom)

		proj_dist = torch.zeros(next_dist.size())
		proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
		proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

		return proj_dist

	def compute_td_loss():
		s0, a, r, s1, done = replay_buffer.sample(args.batch_size)

		s0 = torch.FloatTensor(s0)
		a = torch.LongTensor(a)
		r = torch.FloatTensor(r)
		with torch.no_grad():
			s1 = torch.FloatTensor(s1)
		done = torch.FloatTensor(np.float32(done))

		proj_dist = project_dist(s1, r, done)

		dist = model(s0)
		action = a.unsqueeze(1).unsqueeze(1).expand(args.batch_size, 1, args.atom)
		dist = dist.gather(1, action).squeeze(1)
		dist.data.clamp_(0.01, 0.99)
		loss = -(proj_dist * dist.log()).sum(1).mean()
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		return loss.item()

	losses = []
	all_rewards = []
	episode_reward = 0

	state = env.reset()

	for i in range(args.max_episode_length):
		action = model.act(state)

		next_state, reward, done, _ = env.step(action)

		replay_buffer.push(state, action, reward, next_state, done)
		
		state = next_state
		episode_reward += reward

		if done:
			state = env.reset()
			all_rewards.append(episode_reward)
			episode_reward = 0

		if len(replay_buffer) > args.batch_size:
			loss = compute_td_loss()
			losses.append(loss)
		
		if i > 0 and i % args.learn_start == 0:
			print(np.mean(all_rewards[-10:]), losses[-1])
		
		if i % args.target_update == 0:
			update_target(model, model_target)

	

	
