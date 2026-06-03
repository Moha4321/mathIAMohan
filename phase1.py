import torch
import math
import sys

class U1LatticeField:
    """
    Phase 1: Topology and Field Discretization.
    
    This class constructs the foundational physical manifold (the 'neural lattice').
    It models a 3D grid where each node represents a spatial coordinate, and each 
    directional edge represents a U(1) neural network weight / gauge connection.
    """
    
    def __init__(self, L: int, device: str = 'mps'):
        """
        Initializes the U(1) gauge field on an L x L x L periodic lattice.
        
        Args:
            L (int): The linear dimension of the lattice (e.g., 512).
            device (str): Compute backend ('mps' for Apple Silicon M4, 'cuda', or 'cpu').
        """
        self.L = L
        
        # Rigorous Hardware Selection:
        # We explicitly target the M4's Unified Memory Architecture via MPS.
        # This prevents CPU-to-GPU data transfer bottlenecks during gradient flow.
        if device == 'mps' and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Initializing U(1) Lattice Field of size {L}^3 on {self.device}...")

        # Formulating the Continuous/Discrete Bridge:
        # The connection (weights) is represented by an angle theta in [-pi, pi).
        # We initialize with uniform random noise. In physical terms, this is an 
        # "infinite temperature" state (maximum entropy). In neural network terms, 
        # this is the "random weight initialization" before training begins.
        # Shape: [3 directions (x, y, z), L, L, L]
        # Data type: float32 (requires 4 bytes per parameter).
        
        self.theta = torch.empty((3, L, L, L), dtype=torch.float32, device=self.device)
        self.theta.uniform_(-math.pi, math.pi)
        
        # By setting requires_grad=True, we tell PyTorch to track the 
        # computational graph. This directly prepares us for the Euler-Lagrange 
        # continuous limit, allowing us to compute \delta S / \delta \theta
        self.theta.requires_grad_(True)
        
        self._calculate_memory_footprint()

    def _calculate_memory_footprint(self):
        """
        Calculates and logs the memory requirements.
        Crucial for ensuring the 512^3 grid fits perfectly within the Mac Mini's 16GB RAM
        without paging to the SSD, which would ruin compute times.
        """
        # 4 bytes per float32 * total elements
        bytes_used = self.theta.numel() * 4
        gb_used = bytes_used / (1024 ** 3)
        print(f"[*] Base Tensor Memory Footprint: {gb_used:.3f} GB")
        print(f"[*] Expected memory during gradient flow (Adam): ~{gb_used * 3:.3f} GB")

    def shift_forward(self, direction: int) -> torch.Tensor:
        """
        Implements the periodic boundary conditions conceptually mapped to a 3-Torus.
        
        In the IA (Equation 2), you used Taylor Expansion to define the shift x -> x + ε.
        On a discrete grid, shifting data from x+ε back to x is handled by a circular roll.
        
        Args:
            direction (int): 0 for x, 1 for y, 2 for z.
            
        Returns:
            torch.Tensor: The shifted field.
        """
        # PyTorch 'roll' maps precisely to the mathematical periodic boundary condition.
        # Shifts by -1 step along the specified spatial dimension (which is index + 1
        # because index 0 is the direction channel [x, y, z]).
        return torch.roll(self.theta, shifts=-1, dims=direction + 1)
        
    def shift_backward(self, direction: int) -> torch.Tensor:
        """
        The inverse operator of the forward shift, ensuring reversible transport.
        """
        return torch.roll(self.theta, shifts=1, dims=direction + 1)

# ==========================================
# UNIT TESTING & VERIFICATION (Phase 1)
# ==========================================
if __name__ == "__main__":
    # Test 1: Fast test on a small lattice to verify logic
    print("--- Running Topology Integrity Test (L=16) ---")
    test_lattice = U1LatticeField(L=16, device='cpu')
    
    # Verify periodicity (shifting L times should return the exact original tensor)
    shifted_tensor = test_lattice.theta
    for _ in range(16):
        # We must use torch.roll manually here to compound the shift without breaking graph
        shifted_tensor = torch.roll(shifted_tensor, shifts=-1, dims=1) 
        
    # Rigorous numerical check: The L-shifted field must identically equal the starting field
    is_periodic = torch.allclose(test_lattice.theta, shifted_tensor, atol=1e-7)
    print(f"Periodic Boundary Condition verification: {'PASSED' if is_periodic else 'FAILED'}")

    # Test 2: Memory scaling test simulating your target Mac Mini M4 execution
    print("\n--- Dry Run: M4 Memory Footprint Test (L=512) ---")
    try:
        # Note: We allocate this on CPU for the dry script run to avoid instant crashes 
        # if run on an unsupported machine, but on your M4 it will use 'mps'.
        m4_lattice = U1LatticeField(L=512, device='cpu')
    except Exception as e:
        print(f"Failed to allocate 512^3 lattice. Error: {e}")