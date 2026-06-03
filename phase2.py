import torch
import math
from phase1 import U1LatticeField

class U1PhysicsEngine:
    """
    Phase 2: The Energy Functional and Curvature.
    
    Evaluates the Action of the geometric neural network.
    """
    def __init__(self, lattice: U1LatticeField):
        self.lattice = lattice
        
    def compute_plaquettes(self) -> torch.Tensor:
        """
        Calculates the phase angle around every elementary square.
        """
        t = self.lattice.theta
        
        # 1. XY Plane (mu=0, nu=1)
        t_y_shift_x = torch.roll(t[1], shifts=-1, dims=0)
        t_x_shift_y = torch.roll(t[0], shifts=-1, dims=1)
        P_xy = t[0] + t_y_shift_x - t_x_shift_y - t[1]
        
        # 2. YZ Plane (mu=1, nu=2)
        t_z_shift_y = torch.roll(t[2], shifts=-1, dims=1)
        t_y_shift_z = torch.roll(t[1], shifts=-1, dims=2)
        P_yz = t[1] + t_z_shift_y - t_y_shift_z - t[2]
        
        # 3. ZX Plane (mu=2, nu=0)
        t_x_shift_z = torch.roll(t[0], shifts=-1, dims=2)
        t_z_shift_x = torch.roll(t[2], shifts=-1, dims=0)
        P_zx = t[2] + t_x_shift_z - t_z_shift_x - t[0]
        
        P = torch.stack([P_xy, P_yz, P_zx], dim=0)
        # Wrap phases strictly to [-pi, pi)
        P = torch.remainder(P + math.pi, 2 * math.pi) - math.pi
        return P

    def compute_action(self) -> torch.Tensor:
        P = self.compute_plaquettes()
        action_density = 1.0 - torch.cos(P)
        return torch.sum(action_density)


# ==========================================
# UNIT TESTING & VERIFICATION (Phase 2)
# ==========================================
if __name__ == "__main__":
    print("--- Running Gauge Invariance Test (Constraint C2) ---")
    
    L = 32
    lattice = U1LatticeField(L=L, device='cpu')
    
    # FIX: For mathematical theorem verification, we temporarily promote 
    # the tensor to float64 (Double Precision) to eliminate summation rounding errors.
    lattice.theta = lattice.theta.to(torch.float64)
    
    engine = U1PhysicsEngine(lattice)
    
    # 1. Initial State
    initial_plaquettes = engine.compute_plaquettes()
    initial_action = engine.compute_action().item()
    print(f"[*] Initial Total Action S: {initial_action:.8f}")
    
    # 2. Apply Gauge Transformation
    print("[*] Applying continuous gauge transformation (A_mu -> A_mu + d_mu lambda)...")
    lambda_field = torch.empty((L, L, L), dtype=torch.float64).uniform_(-math.pi, math.pi)
    
    with torch.no_grad():
        lattice.theta[0] += torch.roll(lambda_field, shifts=-1, dims=0) - lambda_field
        lattice.theta[1] += torch.roll(lambda_field, shifts=-1, dims=1) - lambda_field
        lattice.theta[2] += torch.roll(lambda_field, shifts=-1, dims=2) - lambda_field
        
        lattice.theta.copy_(torch.remainder(lattice.theta + math.pi, 2 * math.pi) - math.pi)
        
    # 3. Post-Transformation State
    transformed_plaquettes = engine.compute_plaquettes()
    gauge_transformed_action = engine.compute_action().item()
    print(f"[*] Post-Transformation Action S: {gauge_transformed_action:.8f}")
    
    # 4. Rigorous Mathematical Assertions
    abs_error = abs(initial_action - gauge_transformed_action)
    rel_error = abs_error / initial_action
    
    # Element-wise check of the field strength tensor F_mu_nu
    # We use a trick to account for phase wrapping differences at the pi / -pi boundary
    phase_diff = torch.remainder(initial_plaquettes - transformed_plaquettes + math.pi, 2*math.pi) - math.pi
    max_element_error = torch.max(torch.abs(phase_diff)).item()

    print(f"\n[*] Global Relative Error: {rel_error:.2e}")
    print(f"[*] Max Element-wise F_mu_nu Error: {max_element_error:.2e}")
    
    if rel_error < 1e-10 and max_element_error < 1e-10:
        print("\nSUCCESS: Lagrangian is rigorously Gauge Invariant (Passes Constraint C2).")
        print("Symmetry holds exactly up to IEEE 754 Double Precision limits.")
    else:
        print("\nFAIL: Symmetry broken.")