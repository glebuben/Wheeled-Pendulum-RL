import torch
from src.policy_network import PolicyNetwork
import matplotlib.pyplot as plt

loaded_checkpoint = torch.load("checkpoints/checkpoint_20260217_213154_a151c357.pt")

average_return_history = loaded_checkpoint["average_return_history"]
policy_state_dict = loaded_checkpoint["policy_state_dict"]

policy_net = PolicyNetwork(action_bound=1.0)
policy_net.load_state_dict(policy_state_dict)

for name, param in policy_net.named_parameters():
    print(f"Layer: {name} | Shape: {param.shape}")


plt.plot(average_return_history)
plt.xlabel("Update")
plt.ylabel("Average Return")
plt.title("Training Progress")
plt.show()
