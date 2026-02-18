import torch
import argparse
import matplotlib.pyplot as plt
from src.policy_network import PolicyNetwork


# ---------- CLI arguments ----------
parser = argparse.ArgumentParser(description="Plot training progress from checkpoint")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to checkpoint file (.pt)"
)

args = parser.parse_args()
checkpoint_path = args.checkpoint


# ---------- Load checkpoint ----------
loaded_checkpoint = torch.load(checkpoint_path)

average_return_history = loaded_checkpoint["average_return_history"]
policy_state_dict = loaded_checkpoint["policy_state_dict"]

policy_net = PolicyNetwork(action_bound=1.0)
policy_net.load_state_dict(policy_state_dict)

# ---------- Print model structure ----------
for name, param in policy_net.named_parameters():
    print(f"Layer: {name} | Shape: {param.shape}")

# ---------- Plot ----------
plt.plot(average_return_history)
plt.xlabel("Update")
plt.ylabel("Average Return")
plt.title("Training Progress")
plt.show()
