"""
PyTorch wrappers for CasADi-based system dynamics and costs.

This module provides PyTorch-compatible versions of the dynamics and cost
functions defined in systems.py, enabling automatic differentiation for
reinforcement learning training.
"""

import torch
import numpy as np
import casadi as cs
from .systems import get_system
from .config import STATE_SAMPLING_CONFIG


class CasADiTorchFunction(torch.autograd.Function):
    """
    Wraps a CasADi function to make it compatible with PyTorch autograd.
    Uses CasADi's automatic differentiation for the backward pass.
    """
    @staticmethod
    def forward(ctx, casadi_func, x, u):
        """
        Forward pass: evaluate CasADi function.

        Args:
            casadi_func: CasADi Function with signature f(x, u) -> output
            x: PyTorch tensor (batch_size, nx)
            u: PyTorch tensor (batch_size, nu)

        Returns:
            PyTorch tensor with function outputs
        """
        # Convert to numpy and evaluate
        x_np = x.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()

        # Evaluate for each batch element
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            out = casadi_func(x_np[i], u_np[i])
            outputs.append(np.array(out).flatten())

        output_np = np.array(outputs)

        # Handle scalar outputs (like cost functions)
        if output_np.ndim == 1:
            output_np = output_np.reshape(-1, 1)

        output = torch.tensor(output_np, dtype=x.dtype, device=x.device)

        # Save for backward
        ctx.casadi_func = casadi_func
        ctx.save_for_backward(x, u)
        ctx.output_shape = output.shape

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients using CasADi's AD.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradients w.r.t. inputs (None, grad_x, grad_u)
        """
        x, u = ctx.saved_tensors
        casadi_func = ctx.casadi_func

        # Create symbolic variables
        x_sym = cs.SX.sym('x', x.shape[1])
        u_sym = cs.SX.sym('u', u.shape[1])
        f_sym = casadi_func(x_sym, u_sym)

        # Compute Jacobians
        jac_x = cs.Function('jac_x', [x_sym, u_sym], [cs.jacobian(f_sym, x_sym)])
        jac_u = cs.Function('jac_u', [x_sym, u_sym], [cs.jacobian(f_sym, u_sym)])

        # Evaluate Jacobians for each batch element
        x_np = x.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        grad_x_list = []
        grad_u_list = []

        for i in range(x.shape[0]):
            jx = np.array(jac_x(x_np[i], u_np[i]))
            ju = np.array(jac_u(x_np[i], u_np[i]))

            # Reshape Jacobians properly
            if jx.ndim == 1:
                jx = jx.reshape(1, -1)
            if ju.ndim == 1:
                ju = ju.reshape(1, -1)

            # Chain rule: grad_input = grad_output @ jacobian
            grad_x_list.append(grad_out_np[i].reshape(1, -1) @ jx)
            grad_u_list.append(grad_out_np[i].reshape(1, -1) @ ju)

        grad_x = torch.tensor(np.array(grad_x_list).squeeze(1), dtype=x.dtype, device=x.device)
        grad_u = torch.tensor(np.array(grad_u_list).squeeze(1), dtype=u.dtype, device=u.device)

        return None, grad_x, grad_u


def create_torch_dynamics_and_cost(system):
    """
    Create PyTorch-compatible dynamics and cost functions from a system.

    This function wraps the CasADi symbolic expressions from systems.py
    to create differentiable PyTorch functions that can be used in RL training.

    Args:
        system: System object from systems.py (SingleIntegrator, DoublePendulum, etc.)

    Returns:
        tuple: (dynamics_torch, cost_torch)
            - dynamics_torch: Function (x, u) -> x_next
            - cost_torch: Function (x, u) -> cost
    """
    # Create CasADi functions from symbolic expressions
    x_sym = cs.SX.sym('x', system.nx)
    u_sym = cs.SX.sym('u', system.nu)

    dynamics_expr = system.dynamics_sym(x_sym, u_sym)
    cost_expr = system.running_cost_sym(x_sym, u_sym)

    dynamics_casadi = cs.Function('dynamics', [x_sym, u_sym], [dynamics_expr])
    cost_casadi = cs.Function('cost', [x_sym, u_sym], [cost_expr])

    def dynamics_torch(x, u):
        """PyTorch dynamics: x_next = f(x, u)"""
        return CasADiTorchFunction.apply(dynamics_casadi, x, u)

    def cost_torch(x, u):
        """PyTorch running cost: l(x, u)"""
        return CasADiTorchFunction.apply(cost_casadi, x, u)

    return dynamics_torch, cost_torch


def sample_states_for_system(system, batch_size, device):
    """
    Sample random states appropriate for the system type.
    Uses the same ranges as defined in STATE_SAMPLING_CONFIG to ensure
    consistency between critic dataset generation and RL training.

    Args:
        system: System object from systems.py
        batch_size: Number of states to sample
        device: PyTorch device (cpu or cuda)

    Returns:
        PyTorch tensor (batch_size, nx) with random states
    """
    system_name = system.__name__()

    # Get config for this system
    if system_name not in STATE_SAMPLING_CONFIG:
        raise ValueError(f"Sampling config not defined for system: {system_name}")
    
    cfg = STATE_SAMPLING_CONFIG[system_name]

    if system_name == "single_integrator":
        x_min, x_max = cfg["x_min"], cfg["x_max"]
        x = torch.rand(batch_size, 1, device=device) * (x_max - x_min) + x_min

    elif system_name == "double_integrator":
        p_min, p_max = cfg["p_min"], cfg["p_max"]
        v_min, v_max = cfg["v_min"], cfg["v_max"]
        x1 = torch.rand(batch_size, 1, device=device) * (p_max - p_min) + p_min
        x2 = torch.rand(batch_size, 1, device=device) * (v_max - v_min) + v_min
        x = torch.cat([x1, x2], dim=1)

    elif system_name == "single_pendulum":
        theta_min, theta_max = cfg["theta_min"], cfg["theta_max"]
        omega_min, omega_max = cfg["omega_min"], cfg["omega_max"]
        x1 = torch.rand(batch_size, 1, device=device) * (theta_max - theta_min) + theta_min
        x2 = torch.rand(batch_size, 1, device=device) * (omega_max - omega_min) + omega_min
        x = torch.cat([x1, x2], dim=1)

    else:  # double_pendulum
        theta_min, theta_max = cfg["theta_min"], cfg["theta_max"]
        omega_min, omega_max = cfg["omega_min"], cfg["omega_max"]
        theta = torch.rand(batch_size, 2, device=device) * (theta_max - theta_min) + theta_min
        omega = torch.rand(batch_size, 2, device=device) * (omega_max - omega_min) + omega_min
        x = torch.cat([theta, omega], dim=1)

    return x
