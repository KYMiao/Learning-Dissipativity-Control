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
dim_xc = 1         # Auxiliary state dimension (choose as needed)
dim_u = 1          # Control input dimension (single torque input)
dim_ue = dim_xc + dim_u  # Extended control input dimension
dim_xe = dim_x + dim_xc  # Extended state dimension

# Define Storage Function Neural Network Ve(xe)
class VNN(nn.Module):
    def __init__(self):
        super(VNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_xe, 32), # input dim : dim_xe
            nn.Tanh(),
            nn.Linear(32, dim_xe * dim_xe) # output dim : dim_xe*dim_xe
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
            nn.Linear(dim_xe, 32),
            nn.Tanh(),
            nn.Linear(32, dim_xe * dim_xe)
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
            nn.Linear(dim_xe, 32),
            nn.Tanh(),
            nn.Linear(32, dim_xe * dim_ue)
        )

    def forward(self, x):
        s_flat = self.net(x)
        s = s_flat.view(-1, dim_xe, dim_ue) # output of the SNN is a matrix
        return s


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
        # Note the sign change to make theta = 0 the upright unstable equilibrium point

        theta_ddot = (self.m * self.gr * self.l / (2 * self.I)) * torch.sin(theta)
        # Construct f(x)
        fx = torch.stack([theta_dot, theta_ddot], dim=-1)  # Shape: (batch_size, 2)

        return fx

    def g(self, x):
        # Control dynamics matrix (g(x)) of the fixed inverted pendulum system
        # Since control input is a direct torque, it affects only angular acceleration
        gx = torch.zeros(x.size(0), dim_x, dim_u, device=x.device)
        gx[:, 1, 0] = 1 / self.I  # Torque directly affects theta_dot_dot
        return gx

    def forward(self, xe, ue):
        # Split xe into x and xc components
        x = xe[..., :dim_x]  # Extract state x (angle, angular velocity)

        # Calculate f(x) and g(x)
        fx = self.f(x)  # Unforced dynamics of x
        gx = self.g(x)  # Control influence matrix g(x)

        # Construct the augmented matrix [0 gx; I 0] for the stacked system
        Gx = torch.zeros(x.size(0), dim_xe, dim_ue, device=xe.device)  # Shape: (batch_size, dim_xe, dim_ue)
        Gx[:, :dim_x, dim_xc:dim_u + dim_xc] = gx
        Gx[:, dim_x:dim_x + dim_xc, :dim_xc] = torch.eye(dim_xc, device=xe.device).expand(x.size(0), dim_xc, dim_xc)

        # Stack f(x) and auxiliary dynamics to form the augmented system
        Fx = torch.cat([fx, torch.zeros(x.size(0), dim_xc, device=xe.device)],
                               dim=-1)  # Shape: (batch_size, dim_xe)

        # Calculate the batched vector field dxe/dt = [f(x); 0] + Gx @ ue
        if dim_xc == 0:
            dxdt = fx + (gx @ ue.unsqueeze(-1)).squeeze(-1)
            return dxdt, fx, gx
        else:
            dxdt = Fx + (Gx @ ue.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, dim_xe)
            return dxdt, Fx, Gx


class RandomStateDataset(IterableDataset):
    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.data = []  # Stores both initial random samples and counterexamples
        for _ in range(self.num_batches):
            batch_states = torch.stack([
                torch.tensor([
                    torch.empty(1).uniform_(-3.14, 3.14),  # Angle theta (radians)
                    torch.empty(1).uniform_(-2, 2),  # Angular velocity theta_dot
                    torch.empty(1).uniform_(-1, 1),
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
    theta = xe[0]  # Angle
    theta_dot = xe[1]  # Angular velocity

    # Compute fx (unforced dynamics)
    theta_ddot = (model.pendulum_model.m * model.pendulum_model.gr * model.pendulum_model.l / (2 * model.pendulum_model.I)) * sin(theta)
    fx = [theta_dot, theta_ddot]

    # Augmented dynamics
    Fx = fx + [Expression(0) for _ in range(len(xe) - len(fx))]  # Add zeros for auxiliary states

    # Define gx as a matrix of shape (dim_x, dim_u)
    gx = [[Expression(0) for _ in range(dim_u)] for _ in range(dim_x)]
    gx[1][0] = Expression(1) / model.pendulum_model.I  # Torque directly affects theta_dot_dot

    # Define Gx as a matrix of shape (dim_xe, dim_ue)
    Gx = [[Expression(0) for _ in range(dim_ue)] for _ in range(dim_xe)]

    # Populate Gx based on the torch logic
    for i in range(dim_x):
        for j in range(dim_u):
            Gx[i][j + dim_xc] = gx[i][j]  # Assign gx to the appropriate slice of Gx

    # Identity matrix for auxiliary state feedback in Gx
    for i in range(dim_xc):
        Gx[dim_x + i][i] = Expression(1)
    if dim_xc == 0:
        return fx, gx
    else:
        return Fx, Gx


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

def pgd_attack(model, local_bounds=(3.14, 2, 1), num_samples=30, step_size=0.01):
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

    Q_symmetric = 0.5 * (model.Q(xe_samples) + model.Q(xe_samples).transpose(-1, -2))
    R_symmetric = 0.5 * (model.R + model.R.transpose(-1, -2))

    ue = compute_control(model, xe_samples)

    # Compute Ve and ∇Ve (gradient of Ve)
    Ve, Ve_derivative = compute_clf(model, xe_samples, ue)

    # Part 1 of the constraint: V_derivative PD
    fc1 = Ve_derivative
    fc1 = torch.relu(fc1).mean()

    # Part 2 of the costraint: Delta PD
    inner_matrix = model.S(xe_samples) @ torch.inverse(R_symmetric) @ model.S(xe_samples).transpose(-1, -2) - Q_symmetric
    eigenvalues = torch.linalg.eigvals(-inner_matrix.cpu()).real  # Shape: (32, 3)
    max_eigenvalue = torch.max(eigenvalues, dim=1, keepdim=True)[0]  # Shape: (32, 1)
    fc2 = torch.relu(max_eigenvalue).mean()

    # Part 3 of loss: storage PD
    tmp = -Ve
    # 0.1 x^2 <= V, V <= 3 * x^2
    residual = xe_samples.unsqueeze(-1).transpose(-1, -2) @ xe_samples.unsqueeze(-1)
    fc3 = tmp + 0.5 * residual
    fc3 = torch.relu(fc3).mean()
    fc3 = fc3 + torch.relu(- tmp - 3 * residual).mean()


    loss = fc1 + fc2 + fc3
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

def verify_conditions(model, local_bounds=(0.01, 0.01, 0.01), epsilon=1e-2, num_samples=32*20):
    """
    Verify conditions using dReal for the four loss functions.
    """
    config = Config()
    config.precision = 1e-2
    config.use_polytope_in_forall = True
    config.use_local_optimization = True

    # Define symbolic variables
    xe = [Variable(f"xe_{i}") for i in range(dim_xe)]
    x = xe[:dim_x]  # Extract x from xe
    xc = xe[dim_x:]  # Extract xc from xe

    # Compute hxe and other terms
    hxe = xe

    # Compute xe bounds

    # xe_bounds = logical_and(
    #     logical_and(*(xe[i] >= 0.1 for i in range(len(xe)))),
    #     logical_and(*(xe[i]  <= 1 for i in range(len(xe))))
    # )


    # xe_bounds = logical_and(
    #     logical_and(*(xe_i ** 2 >= 0.01 for xe_i in xe)),
    #     logical_and(*(xe_i ** 2 <= 1 for xe_i in xe))
    # )

    # xe_bounds = logical_and(*(xe_i ** 2 <= 1 for xe_i in xe))
    xe_bounds = logical_and(*(xe[i] ** 2 <= local_bounds[i] for i in range(len(xe))))


    # Compute dynamics
    Fx, Gx = compute_pendulum_dynamics(xe, model)

    # Symbolically compute V(xe)
    V_matrix = symbolic_network(model.V.net, xe, (dim_xe, dim_xe))

    # Symbolically compute Q(xc)
    Q_matrix = symbolic_network(model.Q.net, xe, (dim_xe, dim_xe))
    # Compute symbolic transpose of Q_matrix
    Q_matrix_transpose = [[Q_matrix[j][i] for j in range(len(Q_matrix))] for i in range(len(Q_matrix[0]))]
    # Compute symmetric Q_matrix
    Q_symmetric = [[0.5 * (Q_matrix[i][j] + Q_matrix_transpose[i][j])
                    for j in range(len(Q_matrix[0]))]
                   for i in range(len(Q_matrix))]

    # Symbolically compute S(xc)
    S_matrix = symbolic_network(model.S.net, xe, (dim_xe, dim_ue))

    R_symmetric = 0.5 * (model.R + model.R.transpose(-1, -2))
    R_inverse = torch.inverse(R_symmetric)
    Ve = Expression(0)
    for i in range(len(xe)):
        for j in range(len(xe)):
            Ve += V_matrix[i][j] * xe[i] * xe[j]

    # Compute the gradient of Ve symbolically
    grad_Ve = [Ve.Differentiate(xe_i) for xe_i in xe]

    # Define blocks symbolically
    blk11 = R_inverse

    S_transpose_hxe = [sum(S_matrix[j][i] * hxe[j] for j in range(len(hxe))) for i in range(len(S_matrix[0]))]
    Gx_transpose_gradVe = [sum(Gx[j][i] * grad_Ve[j] for j in range(len(grad_Ve))) for i in range(len(Gx[0]))]
    # blk12 = S_matrix^T @ hxe - 0.5 * Gx^T @ grad_Ve
    blk12 = [S_transpose_hxe[i] - 0.5 * Gx_transpose_gradVe[i] for i in range(len(S_matrix[0]))]

    blk21 = blk12 # well, they are both ``vectors", with the same structure as in lists
    blk22 = -sum(grad_Ve[i] * Fx[i] for i in range(len(Fx))) + \
            sum(Q_symmetric[i][j] * hxe[i] * hxe[j] for i in range(len(hxe)) for j in range(len(hxe))) - \
            0.1 * sum(xe[i] ** 2 for i in range(len(xe)))

    # blk21_blk11_inv = blk21 @ blk11_inv
    blk21_blk11_inv = [
        sum(blk21[k] * blk11[k][j] for k in range(len(blk11)))
        for j in range(len(blk11[0]))
    ]

    # blk21_blk11_inv_blk12 = blk21_blk11_inv @ blk12
    blk21_blk11_inv_blk12 = sum(
        blk21_blk11_inv[j] * blk12[j] for j in range(len(blk12))
    )

    clf_derivative_matrix = blk22 - blk21_blk11_inv_blk12

    # Condition 1: clf_derivative_matrix >= -epsilon for sampled xe
    condition_1 = logical_imply(xe_bounds,
        clf_derivative_matrix >= -epsilon
    )

    # Revised condition 2: supply rate <= epsilon for sampled xe and computed feedback ue
    # Compute S^T * hxe
    ue_computed = [-sum(R_inverse[i][j] * S_transpose_hxe[j] for j in range(len(S_transpose_hxe))) for i in range(len(R_inverse))]

    supply_rate = (sum(hxe[i] * sum(Q_symmetric[i][j] * hxe[j] for j in range(len(hxe))) for i in range(len(hxe)))
         + 2 * sum(hxe[i] * sum(S_matrix[i][j] * ue_computed[j] for j in range(len(ue_computed))) for i in
                   range(len(hxe)))
         + sum(ue_computed[i] * sum(R_symmetric[i][j] * ue_computed[j] for j in range(len(ue_computed))) for i in
               range(len(ue_computed))))

    condition_2 = logical_imply(xe_bounds, supply_rate <= epsilon)

    # Revised condition 3: 0.5 xe^T xe - Ve <= epsilon for any sampled xe
    condition_3 = logical_imply(xe_bounds,
                                logical_and((sum(xe_i ** 2 for xe_i in xe) / 2 <= Ve + epsilon),
                                            (sum(xe_i ** 2 for xe_i in xe) * 3 >= Ve - epsilon))
                                )

    # Combine all conditions
    all_conditions = logical_and(condition_1, condition_2, condition_3)
    # Check satisfiability
    result = CheckSatisfiability(logical_not(all_conditions), config)

    # counterexamples = []
    # print(f"counter examples {result}")
    # if result:
    #     for _ in range(num_samples):
    #         counterexample = []
    #         for xe_var in xe:
    #             interval = result[xe_var]  # dReal returns intervals
    #             sample = np.random.uniform(interval.lb(), interval.ub())  # Sample from the interval
    #             counterexample.append(sample)
    #         counterexamples.append(counterexample)  # Append each vector as a row

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

    print(f"Saving trained model to model/model_storage_pendulum_dynamic_shaping.pth...")
    # # Save the model
    torch.save(model.state_dict(), "model/model_storage_pendulum_dynamic_shaping.pth")
    print("Model saved successfully.")

# Define the function to compute control based on model and state
# Define the function to compute control based on model and state
def compute_control(model, xe):

    # Feedback control ue = K(xc) * hxe
    # ue = model.K(xc) @ hxe
    x = xe[..., :dim_x]
    # QSR Control ue = -R^(-1) * S' * xe + \bar u_e
    hxe = xe.unsqueeze(-1)
    R_symmetric = 0.5 * (model.R + model.R.transpose(-1, -2))
    ue = - torch.inverse(R_symmetric) @ model.S(xe).transpose(-1, -2) @ hxe

    # Zero input
    # ue = - torch.zeros_like(ue)

    return ue.squeeze(-1)


def compute_clf(model, xe, ue):
    xe.requires_grad_(True)
    hxe = xe.unsqueeze(-1)

    # Compute Ve(xe)
    Ve_xe = xe.unsqueeze(-1).transpose(-1, -2) @ model.V(xe) @ xe.unsqueeze(-1)
    # Ve_xe = Ve_xe - 0.1 * xe.unsqueeze(-1).transpose(-1, -2) @ xe.unsqueeze(-1)

    # Compute the gradient of Ve(xe) with respect to xe
    Ve_xe_sum = Ve_xe.sum()
    grad_Ve = torch.autograd.grad(Ve_xe_sum, xe, create_graph=True)[0]

    # Compute the closed-loop dynamics: Ax + Bu
    _, Fx, Gx = model.pendulum_model(xe, ue)

    # Make Q and R symmetric
    Q_symmetric = 0.5 * (model.Q(xe) + model.Q(xe).transpose(-1, -2))
    R_symmetric = 0.5 * (model.R + model.R.transpose(-1, -2))

    # Matrix form dissipation condition
    blk11 = R_symmetric
    blk11 = blk11.unsqueeze(0).repeat(xe.size(0), 1, 1)  # Shape: (batch_size, row, column)
    blk12 = model.S(xe).transpose(-1, -2) @ hxe - 0.5 * Gx.transpose(-1, -2) @ grad_Ve.unsqueeze(-1)
    blk21 = blk12.transpose(-1, -2)
    blk22 = -(grad_Ve * Fx).sum(dim=1).unsqueeze(-1).unsqueeze(-1) + \
                hxe.transpose(-1, -2) @ Q_symmetric @ hxe - \
            0.1 * xe.unsqueeze(-1).transpose(-1, -2) @ xe.unsqueeze(-1)
    top_row = torch.cat((blk11, blk12), dim=2)
    bottom_row = torch.cat((blk21, blk22), dim=2)

    block_matrix = torch.cat((top_row, bottom_row), dim=1)
    eigenvalues = torch.linalg.eigvals(-block_matrix.cpu()).real  # Shape: (batch_size, 3)
    max_eigenvalue = torch.max(eigenvalues, dim=1, keepdim=True)[0]  # Shape: (batch_size, 1)
    clf_derivative = max_eigenvalue.to(xe.device)

    return Ve_xe, clf_derivative


class Storage(pl.LightningModule):
    def __init__(self, pretrained_path=None):
        super(Storage, self).__init__()
        self.V = VNN()
        self.Q = QNN()
        self.S = SNN()
        self.R = nn.Parameter(0.1*torch.eye(dim_ue, dim_ue), requires_grad=True)
        self.pendulum_model = FixedPendulum()

    def forward(self, xe):
        xe.requires_grad = True
        # xe = xe.to(device)
        # Extract x and xc from xe

        # Make Q and R symmetric
        Q_symmetric = 0.5 * (self.Q(xe) + self.Q(xe).transpose(-1, -2))
        R_symmetric = 0.5 * (self.R + self.R.transpose(-1, -2))

        ue = compute_control(self, xe)

        # Compute Ve and ∇Ve (gradient of Ve)
        Ve, Ve_derivative = compute_clf(self, xe, ue)

        # Part 1 of the loss: V_derivative PD
        fc1 = Ve_derivative
        fc1 = torch.relu(fc1).mean()

        # Part 2 of the los: Delta PD
        inner_matrix = self.S(xe) @ torch.inverse(R_symmetric) @ self.S(xe).transpose(-1, -2) - Q_symmetric
        eigenvalues = torch.linalg.eigvals(-inner_matrix.cpu()).real  # Shape: (32, 3)
        max_eigenvalue = torch.max(eigenvalues, dim=1, keepdim=True)[0].to(device)  # Shape: (32, 1)

        fc2 = torch.relu(max_eigenvalue).mean()

        # Part 3 of loss: storage PD
        tmp = -Ve
        # 0.1 x^2 <= V, V <= 3 * x^2
        residual = xe.unsqueeze(-1).transpose(-1, -2) @ xe.unsqueeze(-1)
        fc3 = tmp + 0.5 * residual
        fc3 = torch.relu(fc3).mean()
        fc3 = fc3 + torch.relu(- tmp - 3 * residual).mean()

        # R PD
        max_eigenvalue = torch.max(torch.linalg.eigvals(-R_symmetric.cpu()).real)
        fc4 = torch.relu(max_eigenvalue).mean()

        # loss function learning
        _, fx, gx = self.pendulum_model(xe, ue)
        grad_Ve = torch.autograd.grad(Ve.sum(), xe, create_graph=True)[0]
        mse_loss = nn.MSELoss()
        fc5 = mse_loss(grad_Ve.unsqueeze(-1).transpose(-1, -2) @ gx, 2 * xe.unsqueeze(-1).transpose(-1, -2) @ self.S(xe))
        term1 = xe.unsqueeze(-1).transpose(-1, -2) @ Q_symmetric @ xe.unsqueeze(-1)
        term1 = term1 + xe.unsqueeze(-1).transpose(-1, -2) @ inner_matrix @ xe.unsqueeze(-1)
        term1 = term1 - grad_Ve.unsqueeze(-1).transpose(-1, -2) @ fx.unsqueeze(-1)
        fc5 = fc5 + mse_loss(term1, xe.unsqueeze(-1).transpose(-1, -2) @ xe.unsqueeze(-1))  # loss x^T x
        fc5 = fc5 + mse_loss(R_symmetric, torch.eye(dim_u, dim_u, device=xe.device))

        return fc1, fc2, fc3, fc4, fc5

    def training_step(self, batch, batch_idx):
        xe = batch
        xe = xe.squeeze(0)
        loss1, loss2, loss3, loss4, loss5 = self(xe)
        self.log('storage_derivative', loss1, prog_bar=True, logger=True)
        self.log('delta', loss2, prog_bar=True, logger=True)
        self.log('storage_pd', loss3, prog_bar=True, logger=True)
        # self.log('storage at zero', loss5, prog_bar=True, logger=True)
        self.log('R_pd', loss4, prog_bar=True, logger=True)
        return  10 * loss1 + 10 * loss2 + 10 * loss3 + 10 * loss4 + 0.0001 * loss5

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        xe = batch

        # Define the vector field for dynamics
        def dynamics(t, xe):
            # Extract theta
            theta = xe[..., 0]
            # Wrap theta to be within [-pi, pi]
            theta_wrapped = (theta + torch.pi) % (2 * torch.pi) - torch.pi

            # Update xe with the wrapped theta
            xe[..., 0] = theta_wrapped  # Update theta in xe with the wrapped value

            # Calculate control and dynamics
            ue = compute_control(self, xe)
            dxe_dt, *_ = self.pendulum_model(xe, ue)
            return dxe_dt
        # Define time points for integration
        timesteps = torch.linspace(0, 10, steps=200, device=xe.device)

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
def test_and_visualize(model_storage, n_samples=1, num_states=20):
    # Define the pendulum length for visualization
    pendulum_length = 0.5  # Length of the pendulum (same as model parameter l)

    for sample_index in range(n_samples):
        # Generate a random initial state for the pendulum (angle and angular velocity)
        initial_state = torch.tensor([
            torch.empty(1).uniform_(-1 * np.pi, 1 * np.pi),  # Initial angle theta
            torch.empty(1).uniform_(-2, 2),           # Initial angular velocity theta_dot
            torch.empty(1).uniform_(-1, 1),
        ]).float()

        # Add btch dimension to the state and move to the model's device
        initial_state = initial_state.unsqueeze(0).to(next(model_storage.parameters()).device)

        # Use the model.test_step method to generate the trajectory of the pendulum system
        trajectory, _ = model_storage.test_step(initial_state, 0)

        # Move trajectory to CPU and convert to numpy for visualization
        trajectory = trajectory.cpu().detach().numpy()

        # Pick exactly ten states uniformly from the trajectory
        selected_states = np.linspace(0, trajectory.shape[0] - 1, num_states, dtype=int)
        selected_trajectory = trajectory[selected_states]

        # Prepare a high-resolution color gradient (viridis) for the evolution of the pendulum
        colors = cm.viridis(np.linspace(0, 1, num_states))

        # Plot the pendulum evolution over time
        plt.figure(figsize=(8, 6))
        for i, state in enumerate(selected_trajectory):
            state[0] = (state[0] + torch.pi) % (2 * torch.pi) - torch.pi
            theta = state[0]  # Pendulum angle
            theta_dot = state[1] # Angular velocity

            # Calculate the pendulum end position based on theta
            pendulum_pos = [pendulum_length * np.sin(theta), pendulum_length * np.cos(theta)]

            # Dynamically compute the control input for this state
            # Convert `state` back to a tensor for compatibility with the model's control function
            xe_tensor = torch.tensor(state, dtype=torch.float32, device=next(model_storage.parameters()).device).unsqueeze(0)
            control_input = compute_control(model_storage, xe_tensor)

            # Print the value of CLF and CLF derivative
            # Ve, drv_Ve = compute_clf(model_storage, xe_tensor, control_input)
            # print(f"State {i + 1}: CLF, CLF derivative {[Ve.item(), drv_Ve.item()]}")

            # Debug: Print pendulum angle for each state
            print(f"State {i + 1}: Pendulum angle, velocity, xc {[theta, theta_dot]}")

            # Print the control input for the current state
            # control_input = control_input.cpu().detach().numpy()
            # print(f"State {i + 1}: Control = {control_input[0, 1]:.4f} ")


            # Draw the pendulum as a line from the pivot (0,0) to the pendulum end
            plt.plot([0, pendulum_pos[0]], [0, pendulum_pos[1]], color=colors[i], linewidth=2)

            # Draw the pendulum bob as a circle at the end position
            plt.scatter(pendulum_pos[0], pendulum_pos[1], color=colors[i], s=30)

        # Customize plot appearance
        plt.xlabel('Horizontal Position')
        plt.ylabel('Vertical Position')
        plt.title(f'Fixed Inverted Pendulum Trajectory - Sample {sample_index + 1}')
        plt.xlim(-pendulum_length - 0.2, pendulum_length + 0.2)
        plt.ylim(-pendulum_length - 0.2, pendulum_length + 0.2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    train = False
    # Training parameter
    n_epochs = 1
    batch_size = 320
    n_batches = 100
    max_iterations = 100

    # Define model and dataset
    if train:
        model_storage = Storage()
        dataset = RandomStateDataset(batch_size=batch_size, num_batches=n_batches)
        #
        # # Train and save the model
        train_and_save_model(model_storage, dataset, max_iterations=max_iterations, batch_size=batch_size, n_epochs=n_epochs)

    # Test the model
    model_storage_test = Storage()
    model_storage_test.load_state_dict(torch.load("model/model_storage_pendulum_dynamic_shaping.pth"))
    model_storage_test.eval()

    if verify_conditions(model_storage_test):
        print(f"Verification result: {verify_conditions(model_storage_test)}")
    else:
        print("Verification result: PASS")

    # Test and visualize
    test_and_visualize(model_storage_test)
    #

