import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac_utils import *
from model import *
import sys
import time

class SAC(object):
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy, target_update_interval,
                 automatic_entropy_tuning, hidden_size, learning_rate, exploration_scaling_factor):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Initialize the predictive model
        self.predictive_model = PredictiveModel(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.predictive_model_optim = Adam(self.predictive_model.parameters(), lr=learning_rate)

        self.exploration_scaling_factor = exploration_scaling_factor

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        # else:
        #     self.alpha = 0
        #     self.automatic_entropy_tuning = False
        #     self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
        #     self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Predict the next state using the predictive model
        predicted_next_state = self.predictive_model(state_batch, action_batch)

        # Calculate prediction loss as an intrinsic reward
        prediction_error = F.mse_loss(predicted_next_state, next_state_batch)
        prediction_error_no_reduction = F.mse_loss(predicted_next_state, next_state_batch, reduce=False)
        # intrinsic_reward = prediction_error.unsqueeze(1)  # Make sure it has the correct shape
        
        # Add intrinsic reward to the reward batch (with a scaling factor if needed)
        # print("Predicted next state:")
        # print(predicted_next_state)
        # print("Next state batch:")
        # print(next_state_batch)

        scaled_intrinsic_reward = prediction_error_no_reduction.mean(dim=1)
        scaled_intrinsic_reward = self.exploration_scaling_factor * torch.reshape(scaled_intrinsic_reward, (batch_size, 1))


        # print("Scaled Intrinsic Reward")
        # print(scaled_intrinsic_reward)
        # print("Scaled Intrinsic Reward (New)")
        # print(f"Shape: {scaled_intrinsic_reward_new.shape}")
        # print(scaled_intrinsic_reward_new)


        # print(f"Scaled intrinsic reward: {scaled_intrinsic_reward}")
        # print(f"Reward batch before update: {reward_batch}")

        reward_batch = reward_batch + scaled_intrinsic_reward

        # print(f"Reward batch after update: {reward_batch}")

        # print(time.sleep(10))

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        # Update the critic network
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Update the predictive model
        self.predictive_model_optim.zero_grad()
        prediction_error.backward()
        self.predictive_model_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), prediction_error.item(), alpha_tlogs.item()


    # Save model parameters
    def save_checkpoint(self, env_name, suffix=""):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        print('Saving models')
        self.policy.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.predictive_model.save_checkpoint()

    # Load model parameters
    def load_checkpoint(self, evaluate=False):

        try:
            print('Loading models...')
            self.policy.load_checkpoint()
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
            self.predictive_model.load_checkpoint()
            print('Successfully loaded models')
        except:
            print("Unable to load models. Starting from scratch")

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()

    def set_mode(self, mode):
        if(mode == "train"):
            self.policy.train()
        elif(mode == "eval"):
            self.policy.eval()
        else:
            print("Invalid mode selected")
            raise Exception

