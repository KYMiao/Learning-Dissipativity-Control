import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader
from matplotlib.patches import FancyArrowPatch
import copy
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib import cm
from dreal import *
# Random seed
torch.manual_seed(20)


# Check if 'cuda' device is available, otherwise fall back to 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants for dimensions
dim_x = 2          # State dimension for [angle, angular velocity]
dim_xc = 0         # Auxiliary state dimension (choose as needed)
dim_u = 2          # Control input dimension (single torque input)
dim_ue = dim_xc + dim_u  # Extended control input dimension
dim_xe = dim_x + dim_xc  # Extended state dimension

# Define Storage Function Neural Network Ve(xe)
class VNN(nn.Module):
    def __init__(self):
        super(VNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_xe, 8), # input dim : dim_xe
            nn.Tanh(),
            nn.Linear(8, dim_xe * dim_xe) # output dim : dim_xe*dim_xe
        )

    def forward(self, x):
        v_flat = self.net(x)
        v = v_flat.view(-1, dim_xe, dim_xe)  # output of the QNN is a matrix
        return v

# Define Q(xc) Neural Network
class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_xe, 8),
            nn.Tanh(),
            nn.Linear(8, dim_xe * dim_xe)
        )

    def forward(self, x):
        q_flat = self.net(x)
        q = q_flat.view(-1, dim_xe, dim_xe) # output of the QNN is a matrix
        return q

# Define S(xc) Neural Network
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_xe, 8),
            nn.Tanh(),
            nn.Linear(8, dim_ue)
        )

    def forward(self, x):
        s_flat = self.net(x)
        return s_flat


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


class RandomStateDataset(IterableDataset):
    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.data = []  # Stores both initial random samples and counterexamples
        for _ in range(self.num_batches):
            batch_states = torch.stack([
                torch.tensor([
                    torch.empty(1).uniform_(-1, 1),  # Angle theta (radians)
                    torch.empty(1).uniform_(-1, 1),  # Angular velocity theta_dot
                ])
                for _ in range(self.batch_size)
            ])
            self.data.extend(batch_states.tolist())  # Save generated samples to self.data
    def __iter__(self):
        # np.random.shuffle(self.data)
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i + self.batch_size]  # Allow incomplete batch
            yield torch.tensor(batch).float()

    def __len__(self):
        if not self.data:
            extended_num_batches = self.num_batches
        else:
            extended_num_batches = int(len(self.data) / self.batch_size)
        return extended_num_batches

    def add_counterexamples(self, counterexamples):
        """
        Add counterexamples to the dataset.
        Counterexamples are appended to self.data
        """
        # Ensure counterexamples are added as individual entries
        if isinstance(counterexamples[0], list):  # Multiple counterexamples
            self.data.extend(counterexamples)
        else:  # Single counterexample
            self.data.extend([counterexamples])  # Wrap single counterexample in a list

def compute_pendulum_dynamics(xe, model):
    """
    Compute symbolic closed-loop dynamics: Fx and Gx for the pendulum system.
    """

    # Compute fx (unforced dynamics)
    fx = [sum(model.pendulum_model.A[i, j] * xe[j] for j in range(len(xe))) for i in range(len(xe))]

    # Define gx as a matrix of shape (dim_x, dim_u)
    gx = model.pendulum_model.B

    return fx, gx


def symbolic_network(network, input_vars, output_dims):
    """
    Extracts weights, biases, and computes symbolic output for a neural network.

    Args:
        network: A PyTorch Sequential model.
        input_vars: List of symbolic variables representing the input.
        output_dims: Tuple indicating (rows, columns) of the output matrix.

    Returns:
        Symbolic representation of the network's output as a matrix.
    """
    # Extract weights and biases
    weights = []
    biases = []
    for layer in network:
        if isinstance(layer, nn.Linear):  # Check if the layer is Linear
            weights.append(layer.weight.detach().numpy())
            biases.append(layer.bias.detach().numpy())

    # Layer 1: Input to hidden layer
    hidden_layer = [Expression(0) for _ in range(len(weights[0]))]
    for i in range(len(weights[0])):
        for j in range(len(input_vars)):
            hidden_layer[i] += weights[0][i][j] * input_vars[j]
        hidden_layer[i] += biases[0][i]
        hidden_layer[i] = tanh(hidden_layer[i])

    # Layer 2: Hidden to output layer
    flat_output = [Expression(0) for _ in range(len(weights[1]))]
    for i in range(len(weights[1])):
        for j in range(len(hidden_layer)):
            flat_output[i] += weights[1][i][j] * hidden_layer[j]
        flat_output[i] += biases[1][i]

    # Reshape flat output to a matrix
    rows, cols = output_dims
    output_matrix = [[flat_output[i * cols + j] for j in range(cols)] for i in range(rows)]

    return output_matrix

def pgd_attack(model, local_bounds=(1, 1), num_samples=30, step_size=0.01):
    """
    Perform a PGD attack to generate counterexamples.
    """

    # Define input bounds for xe
    xe_lower_bound = torch.tensor([-local_bounds[i] for i in range(len(local_bounds))], dtype=torch.float32)
    xe_upper_bound = torch.tensor([local_bounds[i] for i in range(len(local_bounds))], dtype=torch.float32)

    # Generate random initial samples within bounds
    xe_samples = xe_lower_bound + (xe_upper_bound - xe_lower_bound) * torch.rand(num_samples*batch_size, len(local_bounds))
    xe_samples = xe_samples.detach().requires_grad_(True)

    # Condition 1: dissipation inequality
    xe_samples = xe_samples.requires_grad_(True)
    hxe = xe_samples

    Ve_xe = xe_samples.unsqueeze(-1).transpose(-1, -2) @ model.V(xe_samples) @ xe_samples.unsqueeze(-1)
    Ve_xe_sum = Ve_xe.sum()
    grad_Ve = torch.autograd.grad(Ve_xe_sum, xe_samples, create_graph=True)[0]

    # Compute closed-loop dynamics, Fx and Gx
    _, Fx, Gx = model.pendulum_model(xe_samples, torch.zeros(xe_samples.size(0), dim_ue, device=xe_samples.device))

    ue = compute_control(model, xe_samples)

    # Compute Ve and ∇Ve (gradient of Ve)
    Ve, Ve_derivative = compute_clf(model, xe_samples, ue)

    # Part 1 of the constraint: V_derivative PD
    fc1 = Ve_derivative
    fc1 = torch.relu(fc1).mean()

    # Part 3 of loss: storage PD
    tmp = -Ve
    # 0.1 x^2 <= V, V <= 3 * x^2
    residual = xe_samples.unsqueeze(-1).transpose(-1, -2) @ xe_samples.unsqueeze(-1)
    fc3 = tmp + 0.5 * residual
    fc3 = torch.relu(fc3).mean()
    fc3 = fc3 + torch.relu(- tmp - 3 * residual).mean()


    loss = fc1 + fc3
    loss.backward(torch.ones_like(loss))

    counterexamples = []

    with torch.no_grad():
        for i in range(num_samples*batch_size):
            xe_samples[i] += step_size * xe_samples.grad[i].sign()
            xe_samples[i] = torch.clamp(xe_samples[i], xe_lower_bound, xe_upper_bound)
            counterexamples.append(xe_samples[i].detach().cpu().tolist())


    if counterexamples:
        return counterexamples
    else:
        return None

def verify_conditions(model, local_bounds=(1, 1), epsilon=1e-2, num_samples=32*20):
    """
    Verify conditions using dReal for the four loss functions.
    """
    config = Config()
    config.precision = 1e-3
    config.use_polytope_in_forall = True
    config.use_local_optimization = True

    # Define symbolic variables
    xe = [Variable(f"xe_{i}") for i in range(dim_xe)]

    # Compute hxe and other terms

    # Compute xe bounds

    # xe_bounds = logical_and(*(xe_i ** 2 <= 1 for xe_i in xe))
    xe_bounds = logical_and(*(xe[i] ** 2 <= local_bounds[i] for i in range(len(xe))))

    # Compute dynamics
    Fx, Gx = compute_pendulum_dynamics(xe, model)

    # Compute control
    S_matrix = symbolic_network(model.S.net, xe, (1, dim_ue))
    ue_computed = S_matrix

    # Symbolically compute V(xe)
    V_matrix = symbolic_network(model.V.net, xe, (dim_xe, dim_xe))

    Ve = Expression(0)
    for i in range(len(xe)):
        for j in range(len(xe)):
            Ve += V_matrix[i][j] * xe[i] * xe[j]

    # Compute the gradient of Ve symbolically
    grad_Ve = [Ve.Differentiate(xe_i) for xe_i in xe]

    GeUe = [[sum(Gx[i][j] * ue_computed[0][j] for j in range(len(Gx[0])))] for i in range(len(Gx))]

    clf_derivative = sum(-grad_Ve[i] * (GeUe[i][0] + Fx[i]) - 0.1 * xe[i] * xe[i] for i in range(len(xe)))

    # Condition 1: clf_derivative_matrix >= -epsilon for sampled xe
    condition_1 = logical_imply(xe_bounds,
                                clf_derivative >= -epsilon
                                )

    # Revised condition 3: 0.5 xe^T xe - Ve <= epsilon for any sampled xe
    condition_3 = logical_imply(xe_bounds,
                                logical_and((sum(xe_i ** 2 for xe_i in xe) / 2 <= Ve + epsilon),
                                            (sum(xe_i ** 2 for xe_i in xe) * 3 >= Ve - epsilon))
                                )

    # Combine all conditions
    all_conditions = logical_and(condition_1, condition_3)
    # Check satisfiability
    result = CheckSatisfiability(logical_not(all_conditions), config)

    return result

def iterative_training(model, dataset, max_iterations, batch_size, n_epochs):
    """
    Iterative training process: Train -> Verify -> Augment dataset -> Retrain.
    """

    last_model_state = None
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}: Training model...")

        # Create dataloader for the augmented dataset
        dataloader = DataLoader(dataset)

        # Train model
        logger = TensorBoardLogger("tb_logs", name="pendulum")
        trainer = pl.Trainer(max_epochs=n_epochs, logger=logger)
        # #
        # # # Start training

        if iteration != 0 and last_model_state is not None:
            print("Warm starting the model using the state from the last iteration.")
            model.load_state_dict(last_model_state)  # Warm start with the last trained model state
        else:
            print("No warm start for the first iteration.")

        # Train the model using PyTorch Lightning's trainer
        trainer.fit(model, train_dataloaders=dataloader)

        # Test the model
        # plot_loss_distribution(model)

        # Save the current model state for warm starting the next iteration
        last_model_state = model.state_dict()
        # #

        counterexamples = pgd_attack(model)

        print(f"Found {len(counterexamples)} counterexamples. Adding to dataset...")
        dataset.add_counterexamples(counterexamples)

def train_and_save_model(model, dataset, max_iterations, batch_size, n_epochs):
    """
    Train the model iteratively and save the trained model.
    """
    print("Starting training...")
    iterative_training(model, dataset, max_iterations=max_iterations, batch_size=batch_size, n_epochs=n_epochs)

    print("Training complete, start verifying the model...")

    print(f"Saving trained model to model/model_storage_circuit_CLF.pth...")
    # # Save the model
    torch.save(model.state_dict(), "model/model_storage_circuit_CLF.pth")
    print("Model saved successfully.")

# Define the function to compute control based on model and state
# Define the function to compute control based on model and state
def compute_control(model, xe):
    u = model.S(xe)
    K = torch.tensor([[1.1338, -0.1547], [-0.1547, 1.4574]], dtype=torch.float32)

    # LQR gain
    u = - K.unsqueeze(0).repeat(xe.size(0), 1, 1).to(xe.device) @ xe.unsqueeze(-1)

    # Zero input
    # ue = - torch.zeros_like(ue)

    return u.squeeze(-1)


def compute_clf(model, xe, ue):
    xe.requires_grad_(True)

    # Compute Ve(xe)
    Ve_xe = xe.unsqueeze(-1).transpose(-1, -2) @ model.V(xe) @ xe.unsqueeze(-1)
    # Ve_xe = Ve_xe - 0.1 * xe.unsqueeze(-1).transpose(-1, -2) @ xe.unsqueeze(-1)

    # Compute the gradient of Ve(xe) with respect to xe
    Ve_xe_sum = Ve_xe.sum()
    grad_Ve = torch.autograd.grad(Ve_xe_sum, xe, create_graph=True)[0]

    # Compute the closed-loop dynamics: Ax + Bu
    _, Fx, Gx = model.pendulum_model(xe, ue)

    # Matrix form dissipation condition
    clf_derivative = ((grad_Ve * (Fx + (Gx @ ue.unsqueeze(-1)).squeeze(-1))).sum(dim=1).unsqueeze(-1).unsqueeze(-1) +
                      0.1 * xe.unsqueeze(-1).transpose(-1, -2) @ xe.unsqueeze(-1))

    return Ve_xe, clf_derivative

# 定义PyTorch Lightning模块
class Storage(pl.LightningModule):
    def __init__(self, pretrained_path=None):
        super(Storage, self).__init__()
        self.V = VNN()
        self.Q = QNN()
        self.S = SNN()
        self.R = nn.Parameter(0.1*torch.eye(dim_ue, dim_ue), requires_grad=True)
        self.pendulum_model = LinearSystemModel()

    def forward(self, xe):
        xe.requires_grad = True
        # xe = xe.to(device)
        # Extract x and xc from xe

        ue = compute_control(self, xe)

        # Compute Ve and ∇Ve (gradient of Ve)
        Ve, Ve_derivative = compute_clf(self, xe, ue)

        # Part 1 of the loss: V_derivative PD
        fc1 = Ve_derivative
        fc1 = torch.relu(fc1).mean()


        # Part 3 of loss: storage PD
        tmp = -Ve
        # 0.1 x^2 <= V, V <= 3 * x^2
        residual = xe.unsqueeze(-1).transpose(-1, -2) @ xe.unsqueeze(-1)
        fc3 = tmp + 0.5 * residual
        fc3 = torch.relu(fc3).mean()
        fc3 = fc3 + torch.relu(- tmp - 3 * residual).mean()

        return fc1, fc3

    def training_step(self, batch, batch_idx):
        xe = batch
        xe = xe.squeeze(0)
        loss1, loss3 = self(xe)
        self.log('storage_derivative', loss1, prog_bar=True, logger=True)
        self.log('storage_pd', loss3, prog_bar=True, logger=True)
        return  10 * loss1 + 10 * loss3

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        xe = batch

        # Define the vector field for dynamics
        def dynamics(t, xe):
            # Calculate control and dynamics
            ue = compute_control(self, xe)
            dxe_dt, *_ = self.pendulum_model(xe, ue)
            return dxe_dt
        # Define time points for integration
        timesteps = torch.linspace(0, 3, steps=300, device=xe.device)

        trajectories = odeint(
            dynamics, xe, timesteps,
            method='rk4',
            options={'step_size': 0.01},
            rtol=1e-5,
            atol=1e-6
        )

        trajectories = trajectories.squeeze(1)

        # Compute control inputs along the trajectory
        inputs = []
        for state in trajectories:
            ue = compute_control(self, state.unsqueeze(0))  # Add batch dimension
            inputs.append(ue.squeeze(0))  # Remove batch dimension
        inputs = torch.stack(inputs)

        return trajectories, inputs



# Test and visualization
def test_and_visualize(model_storage, n_samples=1, num_states=100):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

    # Visualization for Current and Voltage Evolution
    ax1.set_title('Evolution of Current and Voltage', fontsize=14)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('State Variables', fontsize=12)

    # Visualization for Control Inputs
    ax2.set_title('Control Input Evolution', fontsize=14)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Control Input', fontsize=12)

    for sample_index in range(n_samples):
        # Generate a random initial state for the linear system
        initial_state = torch.tensor([
            torch.empty(1).uniform_(-1, 1),  # x1
            torch.empty(1).uniform_(-1, 1),   # x2
        ]).float()

        # Add batch dimension to the state and move to the model's device
        initial_state = initial_state.unsqueeze(0).to(next(model_storage.parameters()).device)

        # Use the model.test_step method to generate the trajectory of the linear system
        trajectory, _ = model_storage.test_step(initial_state, 0)

        # Move trajectory to CPU and convert to numpy for visualization
        trajectory = trajectory.cpu().detach().numpy()

        # Pick exactly num_states uniformly from the trajectory
        selected_states = np.linspace(0, trajectory.shape[0] - 1, num_states, dtype=int)
        selected_trajectory = trajectory[selected_states]

        current = []
        voltage = []
        control_inputs = []

        for state in selected_trajectory:
            xe_tensor = torch.tensor(state, dtype=torch.float32, device=next(model_storage.parameters()).device).unsqueeze(0)
            control_input = compute_control(model_storage, xe_tensor).cpu().detach().numpy()

            # Append state variables and control inputs
            current.append(state[0])
            voltage.append(state[1])
            control_inputs.append(control_input[0])

        # Plot the current and voltage trajectories
        ax1.plot(range(len(current)), current, label=f'Sample {sample_index + 1} - Current', linestyle='-', linewidth=1)
        ax1.plot(range(len(voltage)), voltage, label=f'Sample {sample_index + 1} - Voltage', linestyle='--', linewidth=1)

        # Plot the control input trajectories
        ax2.plot(range(len(control_inputs)), control_inputs, label=f'Sample {sample_index + 1}', linewidth=1)

    # Add legends and grid to the subplots
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Optimize layout for clarity
    plt.tight_layout()

    # Save the figure for publication
    plt.savefig('circuit_evolution_icml.png', dpi=300)

    # Show the plot
    plt.show()


def plot_loss_distribution(model_storage):
    """
    Plot the distribution of fc1 loss for the training data.

    Args:
        model_storage: The trained model instance with methods `forward` and `train_dataloader`.
    """
    # Load training data
    # dataloader = model_storage.train_dataloader()

    # List to store computed losses
    loss_values1 = []
    loss_values2 = []

    batched_data = []
    for _ in range(n_batches):
        # Simulate batch states and inputs
        batch_states = torch.stack([
            torch.tensor([
                torch.empty(1).uniform_(-0.51568, -0.51562),  # Pendulum angle
                torch.empty(1).uniform_(0.61236, 0.61242),  # Angular velocity
            ])
            for _ in range(batch_size)
        ])
        batched_data.append(batch_states)

    # with torch.no_grad():
    # for batch in dataloader:
    for batch_idx, batch in enumerate(batched_data):
        xe = batch  # Assuming batch contains xe and ue
        xe = xe.squeeze(0)
        # xe = torch.cat((xe, torch.tensor([[ 1.3591, -0.1820,  0.0000]])), dim=0)
        # Compute the losses
        loss1, loss2 = model_storage.forward(xe)
        # Collect the fc1 loss: clf derivative
        loss_values1.append(loss1.item())


        loss_values2.append(loss2.item())


    # Flatten the loss values if needed
    loss_values1 = torch.tensor(loss_values1)
    loss_values2 = torch.tensor(loss_values2)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(loss_values1.numpy(), bins=30, edgecolor='black', alpha=0.7)
    plt.title("Distribution of fc1 on Training Data")
    plt.xlabel("Loss Value (fc1)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(loss_values2.numpy(), bins=30, edgecolor='black', alpha=0.7)
    plt.title("Distribution of fc2 on Training Data")
    plt.xlabel("Loss Value (fc2)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 运行测试和可视化

if __name__ == "__main__":
    train = False
    # Training parameter
    n_epochs = 1
    batch_size = 320
    n_batches = 100
    max_iterations = 10

    # Define model and dataset
    if train:
        model_storage = Storage()
        dataset = RandomStateDataset(batch_size=batch_size, num_batches=n_batches)
        #
        # # Train and save the model
        train_and_save_model(model_storage, dataset, max_iterations=max_iterations, batch_size=batch_size, n_epochs=n_epochs)

    # Test the model
    model_storage_test = Storage()
    model_storage_test.load_state_dict(torch.load("model/model_storage_circuit_CLF.pth"))
    model_storage_test.eval()

    if verify_conditions(model_storage_test):
        print(f"Verification result: {verify_conditions(model_storage_test)}")
    else:
        print("Verification result: PASS")

    # Test and visualize
    test_and_visualize(model_storage_test)

    # Test clf derivative on training data
    plot_loss_distribution(model_storage_test)

