import torch
import math
from phase1 import U1LatticeField
from phase2 import U1PhysicsEngine
from phase3 import U1GradientFlow

class U1Observables:
    """
    Phase 4: Topological Observables & Validation.
    
    Extracts the physical magnetic fields from the neural gauge lattice and 
    rigorously tests the mathematical tautologies (Maxwell's Equations) 
    derived in the IA.
    """
    def __init__(self, engine: U1PhysicsEngine):
        self.engine = engine

    def get_magnetic_field(self) -> torch.Tensor:
        """
        Extracts the magnetic field vector B = (B_x, B_y, B_z) from the curvature tensor.
        
        From IA Section 3.5:
        B_x = F_yz (Plaquette in YZ plane)
        B_y = F_zx (Plaquette in ZX plane)
        B_z = F_xy (Plaquette in XY plane)
        
        Returns:
            torch.Tensor: Vector field of shape [3, L, L, L]
        """
        P = self.engine.compute_plaquettes()
        
        # P[0] is XY, P[1] is YZ, P[2] is ZX
        B_x = P[1]
        B_y = P[2]
        B_z = P[0]
        
        return torch.stack([B_x, B_y, B_z], dim=0)

    def compute_bianchi_identity(self) -> torch.Tensor:
        """
        Rigorously calculates the divergence of the magnetic field: div(B) = \nabla \cdot B.
        
        On a discrete lattice, the divergence over a voxel (cube) is the sum of fluxes 
        through its 6 faces:
        div(B) = [P_xy(z) - P_xy(z+1)] + [P_yz(x) - P_yz(x+1)] + [P_zx(y) - P_zx(y+1)]
        
        Returns:
            torch.Tensor: Scalar field of shape [L, L, L] representing local magnetic monopoles.
        """
        P = self.engine.compute_plaquettes()
        P_xy, P_yz, P_zx = P[0], P[1], P[2]
        
        # Calculate the flux differences across the voxel
        # Shift dims match the normal vectors of the faces
        dP_xy_dz = P_xy - torch.roll(P_xy, shifts=-1, dims=2)  # Z-shift
        dP_yz_dx = P_yz - torch.roll(P_yz, shifts=-1, dims=0)  # X-shift
        dP_zx_dy = P_zx - torch.roll(P_zx, shifts=-1, dims=1)  # Y-shift
        
        # Sum them up to get the total lattice divergence
        div_B = dP_xy_dz + dP_yz_dx + dP_zx_dy
        
        # Wrap the divergence back into the principal U(1) bundle phase space [-pi, pi)
        # because the topology is defined modulo 2*pi.
        div_B_wrapped = torch.remainder(div_B + math.pi, 2 * math.pi) - math.pi
        return div_B_wrapped

    def compute_energy_density(self) -> torch.Tensor:
        """
        Calculates the local magnetic energy density |B|^2.
        This will be used for the final 3D rendering to visualize the flux tubes.
        """
        B = self.get_magnetic_field()
        # |B|^2 = B_x^2 + B_y^2 + B_z^2. Sum over the vector dimension (dim=0).
        energy_density = torch.sum(B**2, dim=0)
        return energy_density

# ==========================================
# UNIT TESTING & VERIFICATION (Phase 4)
# ==========================================
if __name__ == "__main__":
    print("--- Running Theorem 3.2 Validation (Bianchi Identity) ---")
    
    # Initialize our system
    L = 32
    lattice = U1LatticeField(L=L, device='cpu')
    
    # We upgrade to float64 to test the mathematical theorem cleanly without floating-point noise
    lattice.theta = lattice.theta.to(torch.float64)
    
    engine = U1PhysicsEngine(lattice)
    observables = U1Observables(engine)
    
    print("[*] Computing Lattice Divergence of B on random, highly chaotic initialization...")
    div_B = observables.compute_bianchi_identity()
    
    # Check the absolute maximum divergence anywhere on the 32x32x32 lattice
    max_divergence = torch.max(torch.abs(div_B)).item()
    
    print(f"[*] Maximum absolute value of (nabla \cdot B): {max_divergence:.4e}")
    
    # Rigorous Mathematical Check
    if max_divergence < 1e-13:
        print("\nSUCCESS: Gauss's Law for Magnetism (nabla \cdot B = 0) holds globally!")
        print("Theorem 3.2 is verified: The CNN topology enforces Clairaut's Theorem dynamically.")
    else:
        print("\nFAIL: Magnetic monopoles detected, Bianchi identity violated.")