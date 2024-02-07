import os
import time
import torch
import torch as T
import torch.nn.functional as F
import numpy as np
import math

from buffer import ReplayBuffer

from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], env=None, gamma=0.99, n_actions=2, max_size=1000000,
                 layer1_size=256, layer2_size=256, batch_size=64, entropy_scale=2, tau=0.005):
        # reward scale depends on the complexity of the environment.
        # tau controls how we modulate our target to value network
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                                  name = 'actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_1', fc1_dims=layer1_size, fc2_dims=layer2_size)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_2', fc1_dims=layer1_size, fc2_dims=layer2_size)
        self.value = ValueNetwork(beta, input_dims, name='value', fc1_dims=layer1_size, fc2_dims=layer2_size)

        self.target_value = ValueNetwork(beta, input_dims, name='target_value', fc1_dims=layer1_size, fc2_dims=layer2_size)
        self.scale = entropy_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        try:
            self.actor.load_checkpoint()
            self.value.load_checkpoint()
            self.target_value.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            print("Successfully loaded all models.")
            return True
        except:
            print("Failed to load models. Starting new training process.")
            return False

    def add_random_noise(self, tensor, percentage_change=0.1):
        noise = torch.rand_like(tensor, dtype=tensor.dtype) * 2 * percentage_change - percentage_change
        noisy_tensor = tensor + tensor * noise
        return noisy_tensor

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()

        value_target = critic_value - log_probs
        value_loss = F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # print("Critic Value: ", critic_value)
        # print("Log Probs: ", log_probs)
        actor_loss = (self.scale * log_probs - critic_value)
        # print("Actor Loss(pre noise): ", actor_loss)
        # print("Actor Loss(after noise): ", actor_loss)
        actor_loss = T.mean(actor_loss)


        # Zero the gradients
        self.actor.optimizer.zero_grad()

        # Backward pass
        actor_loss.backward()

        # Update the actor parameters
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = (reward + self.gamma * value_)
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()






