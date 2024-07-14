import numpy as np
import csv

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)


    def can_sample(self, batch_size):
        if self.mem_ctr > (batch_size * 5):
            return True
        else:
            return False

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


    def save_to_csv(self, filename='checkpoints/memory.npz'):
        np.savez(filename,
                 state=self.state_memory[:self.mem_ctr],
                 action=self.action_memory[:self.mem_ctr],
                 reward=self.reward_memory[:self.mem_ctr],
                 next_state=self.new_state_memory[:self.mem_ctr],
                 done=self.terminal_memory[:self.mem_ctr])
        print(f"Saved {filename}")

    def load_from_csv(self, filename='checkpoints/memory.npz'):
        try:
            data = np.load(filename)
            self.mem_ctr = len(data['state'])
            self.state_memory[:self.mem_ctr] = data['state']
            self.action_memory[:self.mem_ctr] = data['action']
            self.reward_memory[:self.mem_ctr] = data['reward']
            self.new_state_memory[:self.mem_ctr] = data['next_state']
            self.terminal_memory[:self.mem_ctr] = data['done']
            print(f"Successfully loaded {filename} into memory")
            print(f"{self.mem_ctr} memories loaded")
        except:
            print(f"Unable to load memory from ")



class CombinedReplayBuffer:

    def __init__(self, buffers, percentages):
        if len(buffers) != len(percentages):
            raise ValueError("Number of buffers must match number of percentages")

        if not np.isclose(sum(percentages), 1.0):
            raise ValueError("Percentages must sum to 1")

        self.buffers = buffers
        self.percentages = percentages
        
    def sample_buffer(self, batch_size):
        sizes = [int(batch_size * perc) for perc in self.percentages]
        
        if any(not buf.can_sample(size) for buf, size in zip(self.buffers, sizes)):
            raise ValueError("One of the buffers cannot currently sample the required batch size")
        
        sampled_data = [buf.sample_buffer(size) for buf, size in zip(self.buffers, sizes)]
        
        states, actions, rewards, states_, dones = zip(*sampled_data)
        
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        states_ = np.concatenate(states_, axis=0)
        dones = np.concatenate(dones, axis=0)
        
        return states, actions, rewards, states_, dones