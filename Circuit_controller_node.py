from typing import Any

import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchdiffeq import odeint
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

# Define constants for dimensions
dim_x = 2          # State dimension for [angle, angular velocity]
dim_xc = 0         # Auxiliary state dimension (choose as needed)
dim_u = 2          # Control input dimension (single torque input)
dim_ue = dim_xc + dim_u  # Extended control input dimension
dim_xe = dim_x + dim_xc  # Extended state dimension

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class LinearSystemModel(nn.Module):
    def __init__(self):
        super(LinearSystemModel, self).__init__()
        # Physical parameters for the linear system
        # self.A = torch.tensor([[-1, -1], [0, -1]], dtype=torch.float32) # State matrix, stable
        self.A = torch.tensor([[0, 1], [-1, 0.5]], dtype=torch.float32) # State matrix, unstable
        self.B = torch.eye(2, dtype=torch.float32) # Input matrix

    def forward(self, xe, ue):
        # Split xe into x and xc components
        x = xe[..., :dim_x]  # Extract state x (angle, angular velocity)

        Atensor = self.A.unsqueeze(0).repeat(xe.size(0), 1, 1).to(xe.device)
        Btensor = self.B.unsqueeze(0).repeat(xe.size(0), 1, 1).to(xe.device)

        fx = (Atensor @ x.unsqueeze(-1)).squeeze(-1)
        gx = Btensor

        # Calculate the batched vector field
        dxdt = fx + (Btensor @ ue.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, dim_xe)
        # Return the batched vector field
        return dxdt, fx, gx

class CircuitModel(nn.Module):
    def __init__(self, controller):
        super(CircuitModel, self).__init__()
        self.circuit = LinearSystemModel()
        self.controller = controller
        self.control_inputs = []

    def forward(self, t, state):
        # state = state.squeeze(0)
        if t == 0:
            self.control_inputs = []

        # For state feedback control
        u = self.controller(state).to(device)
        self.control_inputs.append(u)# Shape: (batch_size, 2)

        dxdt, _, _ = self.circuit(state, u)
        return dxdt

    def get_control_inputs(self):
        return torch.stack(self.control_inputs)


# State Feedback Controller
class StateFeedbackNN(nn.Module):
    def __init__(self):
        super(StateFeedbackNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, 8),
            nn.Tanh(),
            # # nn.Linear(64, 64),
            # # nn.Tanh(),
            nn.Linear(8, dim_u)
        )

    def forward(self, x):
        return self.net(x)


# Dataset for generating random initial states
class CircuitDataset(IterableDataset):
    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            # Random initial states with larger ranges
            x1 = (torch.rand(self.batch_size) * 2) - 1
            x2 = (torch.rand(self.batch_size) * 2) - 1

            initial_states = torch.stack([x1, x2], dim=-1)

            # Target non-zero
            target = 0
            target_states = torch.tensor([target, 0.0]).expand(self.batch_size, -1)

            # Ensure float32 type
            initial_states = initial_states.float()
            target_states = target_states.float()

            yield initial_states, target_states

    def __len__(self):
        return self.num_batches


# PyTorch Lightning Module for Training
class CircuitController(pl.LightningModule):
    def __init__(self):
        super(CircuitController, self).__init__()

        self.controller = StateFeedbackNN()

        self.model = CircuitModel(self.controller).to(device)
        # self.save_hyperparameters()

    def forward(self, x):
        x = x.to(device)
        return self.model(0, x)

    def training_step(self, batch, batch_idx):
        initial_state, target_state = batch

        # Simulate system
        timesteps = torch.linspace(0, 1, 100).to(device)
        trajectory = odeint(
            self.model,
            initial_state,
            timesteps,
            method='rk4'
        )

        control_inputs = self.model.get_control_inputs()

        # Compute loss  throughout trajectory

        state_error = trajectory - target_state.unsqueeze(0)
        loss = torch.mean(torch.sum(state_error ** 2, dim=-1))+torch.mean(torch.sum(control_inputs ** 2, dim=-1))

        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }

    def train_dataloader(self):
        return DataLoader(CircuitDataset(batch_size=64, num_batches=100), batch_size=None)

    def test_step(self, batch, batch_idx):
        x = batch

        # Define time points for integration
        timesteps = torch.linspace(0, 1, steps=100, device=x.device)

        trajectories = odeint(
            self.model, x, timesteps,
            method='rk4',
            options={'step_size': 0.01},
            rtol=1e-5,
            atol=1e-6
        )

        trajectories = trajectories.squeeze(1)
        control_inputs = []

        for state in trajectories:
            # For state feedback control
            u = self.controller(state.unsqueeze(0)).to(device)
            control_inputs.append(u.squeeze(0))  # Remove batch dimension
        control_inputs = torch.stack(control_inputs)
        # control_inputs = self.model.get_control_inputs()
        return trajectories, control_inputs


def train_and_save_controllers():
    # Train State Feedback Controller
    print("Training State Feedback Controller...")
    sf_controller = CircuitController()
    logger = TensorBoardLogger("tb_logs", name="circuit")
    trainer = pl.Trainer(max_epochs=20, logger=logger)  # Increased epochs
    trainer.fit(sf_controller)
    torch.save(sf_controller.state_dict(), "model/circuit_sf_controller_node_8.pth")



def visualize_controllers(sf_controller, n_samples=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    timesteps = torch.linspace(0, 1, 100).to(device)
    target =0

    # State Feedback Controller
    x1 = (torch.rand(n_samples) * 2) - 1
    x2 = (torch.rand(n_samples) * 2) - 1
    initial_state = torch.stack([x1, x2], dim=1).to(device)  # [n_samples, 2]

    trajectory = odeint(sf_controller.model,
            initial_state,
            timesteps,
            method='rk4',
            rtol=1e-5,
            atol=1e-6
        )
    trajectory = trajectory.detach().cpu().numpy()

    # Plot theta and theta_dot for each trajectory
    for i in range(n_samples):
        ax1.plot(timesteps.detach().cpu().numpy(), trajectory[:, i, 0], 'b-', alpha=0.3, label='x_1' if i==0 else None)  # x1
        ax2.plot(timesteps.detach().cpu().numpy(), trajectory[:, i, 1], 'r-', alpha=0.3, label='x_2' if i==0 else None)  # x2
    ax1.axhline(y=target, color='k', linestyle='--', alpha=0.5, label='Target x_1')
    ax2.axhline(y=target, color='k', linestyle='--', alpha=0.5, label='Target x_2')
    ax1.legend()
    ax2.legend()

    # Labels
    ax1.set_title('State Feedback Control')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('State')
    ax1.grid(True)

    ax2.set_title('State Feedback Control')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('State')
    ax2.grid(True)


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Train controllers
    train_and_save_controllers()

    # Load and visualize controllers
    sf_controller = CircuitController()
    sf_controller.load_state_dict(torch.load("model/circuit_sf_controller_node_8.pth"))

    initial_state = torch.tensor([
        torch.empty(1).uniform_(-1, 1),
        torch.empty(1).uniform_(-1, 1)
    ]).float()

    # Add batch dimension to the state and move to the model's device
    initial_state = initial_state.unsqueeze(0).to(device)
    traj, input = sf_controller.test_step(initial_state, 0)


    visualize_controllers(sf_controller)