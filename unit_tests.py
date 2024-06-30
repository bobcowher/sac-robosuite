from buffer import ReplayBuffer

# Start test to ensure new memories are getting rolled onto the stack. 
buffer_size = 10
loop_size = 20

memory = ReplayBuffer(buffer_size, input_shape=[1], n_actions=1)

for i in range(loop_size):
    memory.store_transition(i, i, i, i, i)

# print("MemorySize:", memory.mem_size)
# print("Memory:", memory.state_memory)

print("Testing to ensure the first memory state is correct.")
assert memory.state_memory[0] == 10
print("Test Successful\n")

print("Testing to ensure the last memory state is correct.")
assert memory.state_memory[-1] == 19
print("Test Successful\n")
# Complete test to ensure new memories are getting rolled onto the stack. 

