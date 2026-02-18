import torch
from src.policy_network import PolicyNetwork
import matplotlib.pyplot as plt

ckpt_path = "checkpoints/checkpoint_20260218_063022_7f7c7c67.pt"

loaded_checkpoint = torch.load(ckpt_path)

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
