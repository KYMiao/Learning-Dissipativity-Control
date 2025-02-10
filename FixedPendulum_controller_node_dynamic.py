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

# Constants for state dimensions
dim_x = 2  # State dimension [θ, θ_dot]
dim_u = 1  # Control input dimension (torque)
dim_xc = 1  # Integral error state dimension
dim_xe = dim_x + dim_xc  # Augmented state dimension
dim_ue = dim_u + dim_xc  # Augmented input dimension

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class FixedPendulum(nn.Module):
    def __init__(self):
        super(FixedPendulum, self).__init__()

        # Physical parameters for the inverted pendulum
        self.m = 0.1  # Mass of the pendulum (kg)
        self.l = 2  # Length of the pendulum (m)
        self.gr = 9.81  # Gravitational acceleration (m/s^2)
        self.I = (self.m * self.l ** 2) / 3  # Moment of inertia around the pivot
        self.I = 1 / 3

    def f(self, x):
        # Unforced dynamics (f(x)) of the fixed inverted pendulum system
        theta = x[..., 0]  # Pendulum angle
        theta_dot = x[..., 1]  # Angular velocity

        # Calculate angular acceleration (theta_ddot) due to gravity
        theta_ddot = (self.m * self.gr * self.l / (2 * self.I)) * torch.sin(theta)
        # Construct f(x)
        fx = torch.stack([theta_dot, theta_ddot], dim=-1).to(device)  # Shape: (batch_size, 2)

        return fx

    def g(self, x):
        # Control dynamics matrix (g(x)) of the fixed inverted pendulum system
        gx = torch.zeros(x.size(0), dim_x, dim_u, device=device)
        gx[:, 1, 0] = 1 / self.I  # Torque directly affects theta_dot_dot
        return gx

    def forward(self, xe, ue):
        # Split xe into x and xc components
        x = xe[..., :dim_x]  # Extract state x (angle, angular velocity)

        # Calculate f(x) and g(x)
        fx = self.f(x)  # Unforced dynamics of x
        gx = self.g(x)  # Control influence matrix g(x)

        # Construct the augmented matrix [0 gx; I 0] for the stacked system
        Gx = torch.zeros(x.size(0), dim_xe, dim_ue, device=device)
        Gx[:, :dim_x, dim_xc:dim_u + dim_xc] = gx
        Gx[:, dim_x:dim_x + dim_xc, :dim_xc] = torch.eye(dim_xc, device=device).expand(x.size(0), dim_xc, dim_xc)
        b = torch.zeros(x.size(0), dim_xc, device=device)

        # Stack f(x) and auxiliary dynamics
        Fx = torch.cat([fx, torch.zeros(x.size(0), dim_xc, device=device)], dim=-1)

        # Calculate the vector field dxe/dt = [f(x); 0] + Gx @ ue
        dxdt = Fx + (Gx @ ue.unsqueeze(-1)).squeeze(-1)
        return dxdt, Fx, Gx


# Utility function for angle wrapping
def wrap_angle(theta):
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi


class PendulumModel(nn.Module):
    def __init__(self, controller):
        super(PendulumModel, self).__init__()
        self.pendulum = FixedPendulum()
        self.controller = controller
        self.control_inputs = []

    def forward(self, t, state):
        # state = state.squeeze(0)
        if t == 0:
            self.control_inputs = []
        target = 0
        # Wrap the angle in augmented state
        wrapped_state = torch.cat([
            wrap_angle(state[..., 0:1]),  # Wrap theta
            state[..., 1:2],  # theta_dot unchanged
            state[..., 2:]  # integral term unchanged
        ], dim=-1)

        # ue_input = torch.cat([
        #     wrap_angle(state[..., 0:1] - target),  # Wrap theta
        #     state[..., 1:2],  # theta_dot unchanged
        #     state[..., 2:]  # integral term unchanged
        # ], dim=-1)

        # For PI control, use augmented dynamics
        ue = self.controller(wrapped_state).to(device)
        self.control_inputs.append(ue)# Get augmented control input [u, uc]
        dxdt, _, _ = self.pendulum(wrapped_state, ue)
        return dxdt


    def get_control_inputs(self):
        return torch.stack(self.control_inputs)


# PI Controller with Augmented State
class PIControllerNN(nn.Module):
    def __init__(self):
        super(PIControllerNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_xe, 8),
            nn.Tanh(),
            # nn.Linear(64, 64),
            # nn.Tanh(),
            nn.Linear(8, dim_ue)
        )

    def forward(self, xe):
        return self.net(xe)


# Dataset for generating random initial states
class PendulumDataset(IterableDataset):
    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            # Random initial states with larger ranges
            theta = (torch.rand(self.batch_size) * 2 - 1) * np.pi  # Initial angle
            theta_dot = (torch.rand(self.batch_size) * 4) - 2  # Initial angular velocity


            # Add integral error state for PI control
            xc = (torch.rand(self.batch_size) * 2) - 1
            initial_states = torch.cat([
                theta.unsqueeze(-1),
                theta_dot.unsqueeze(-1),
                xc.unsqueeze(-1)
            ], dim=-1)


            # Target non-zero angle (e.g., π/4)
            target_angle = 0
            target_states = torch.tensor([target_angle, 0.0, 0.0]).expand(self.batch_size, -1)

            # Ensure float32 type
            initial_states = initial_states.float()
            target_states = target_states.float()

            yield initial_states, target_states

    def __len__(self):
        return self.num_batches


# PyTorch Lightning Module for Training
class PendulumController(pl.LightningModule):
    def __init__(self):
        super(PendulumController, self).__init__()
        self.controller = PIControllerNN()


        self.model = PendulumModel(self.controller).to(device)
        # self.save_hyperparameters()

    def forward(self, x):
        x = x.to(device)
        return self.model(0, x)

    def training_step(self, batch, batch_idx):
        initial_state, target_state = batch

        # Simulate system
        timesteps = torch.linspace(0, 3, 301).to(device)
        trajectory = odeint(
            self.model,
            initial_state,
            timesteps,
            method='rk4'
        )

        control_inputs = self.model.get_control_inputs()

        # Compute loss based on wrapped angle error throughout trajectory
        wrapped_trajectory = torch.cat([
            wrap_angle(trajectory[..., 0:1]),  # Wrap theta
            trajectory[..., 1:2],  # theta_dot unchanged
            trajectory[..., 2:]  # integral term unchanged
        ], dim=-1)
        wrapped_target = torch.cat([
            wrap_angle(target_state[..., 0:1]),  # Wrap target theta
            target_state[..., 1:2],  # target theta_dot unchanged
            target_state[..., 2:]  # target integral term unchanged
        ], dim=-1)

        state_error = wrapped_trajectory - wrapped_target.unsqueeze(0)
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
        return DataLoader(PendulumDataset(batch_size=32, num_batches=100), batch_size=None)

    def test_step(self, batch, batch_idx):
        x = batch

        # Define time points for integration
        timesteps = torch.linspace(0, 10, steps=200, device=x.device)

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
            wrapped_state = torch.cat([
                wrap_angle(state[..., 0:1]),  # Wrap theta
                state[..., 1:2],
                state[..., 2:]  # theta_dot unchanged
            ], dim=-1)
            # For pi feedback control
            u = self.controller(wrapped_state.unsqueeze(0)).to(device)
            control_inputs.append(u.squeeze(0))  # Remove batch dimension
        control_inputs = torch.stack(control_inputs)
        # control_inputs = self.model.get_control_inputs()
        return trajectories, control_inputs


def train_and_save_controllers():
    # Train PI Controller
    print("Training PI Controller...")
    pi_controller = PendulumController()
    logger = TensorBoardLogger("tb_logs", name="pendulum")
    trainer = pl.Trainer(max_epochs=50, logger=logger) # Increased epochs
    trainer.fit(pi_controller)
    torch.save(pi_controller.state_dict(), "model/pendulum_pi_controller_node_8.pth")


def visualize_controllers(pi_controller, n_samples=10):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    timesteps = torch.linspace(0, 50, 3000).to(device)
    target =0

    # PI Controller
    theta = (torch.rand(n_samples) * 2 - 1) * np.pi
    theta_dot = (torch.rand(n_samples) * 4) - 2
    xc = (torch.rand(n_samples) * 2) - 1
    initial_state = torch.stack([theta, theta_dot, xc], dim=1)  # [n_samples, 3]

    trajectory = odeint(pi_controller.model,
            initial_state,
            timesteps,
            method='rk4',
            rtol=1e-5,
            atol=1e-6
        )
    trajectory = trajectory.detach().numpy()

    # Plot theta and theta_dot for each trajectory
    for i in range(n_samples):
        ax1.plot(timesteps, trajectory[:, i, 0], 'b-', alpha=0.3, label='θ' if i==0 else None)  # theta
        ax2.plot(timesteps, trajectory[:, i, 1], 'r-', alpha=0.3, label='θ_dot' if i==0 else None) # theta_dot
        ax3.plot(timesteps, trajectory[:, i, 2], 'g-', alpha=0.3, label='xc' if i==0 else None)
    ax1.axhline(y=target, color='k', linestyle='--', alpha=0.5, label='Target θ')
    ax2.axhline(y=target, color='k', linestyle='--', alpha=0.5, label='Target θ_dot')
    ax3.axhline(y=target, color='k', linestyle='--', alpha=0.5, label='Target xc')
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # Labels

    ax1.set_title('PI Control')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('State 1')
    ax1.grid(True)

    ax2.set_title('PI Control')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('State 2')
    ax2.grid(True)

    ax3.set_title('PI Control')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Xc')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Train controllers
    # train_and_save_controllers()
    #
    pi_controller = PendulumController()
    pi_controller.load_state_dict(torch.load("model/pendulum_pi_controller_node_8.pth"))

    initial_state = torch.tensor([
        torch.empty(1).uniform_(-1 * np.pi, 1 * np.pi),  # Initial angle theta
        torch.empty(1).uniform_(-2, 2),
        torch.empty(1).uniform_(-1,1)
        # Initial angular velocity theta_dot
    ]).float()

    # Add batch dimension to the state and move to the model's device
    initial_state = initial_state.unsqueeze(0).to(device)
    traj, input = pi_controller.test_step(initial_state, 0)
    #
    visualize_controllers(pi_controller)