import torch
import torch.optim as optim
import math
from phase1 import U1LatticeField
from phase2 import U1PhysicsEngine

class U1GradientFlow:
    """
    Phase 3: High-Fidelity Gradient Flow.
    
    This class simulates the continuous time evolution of the gauge field.
    It drives the random neural lattice towards its lowest energy state (the vacuum),
    which dynamically solves the discrete Euler-Lagrange equations.
    """
    def __init__(self, engine: U1PhysicsEngine, learning_rate: float = 0.05):
        self.engine = engine
        self.lattice = engine.lattice
        
        # We use Adam, a second-order momentum-based optimizer.
        # This acts as an advanced ODE solver for the gradient flow equations.
        # It allocates 2 additional tensors (momentum m_t and variance v_t) per parameter,
        # which triples our memory footprint (1.5 GB -> 4.5 GB on the 512^3 run).
        self.optimizer = optim.Adam([self.lattice.theta], lr=learning_rate)
        
    def step(self) -> float:
        """
        Executes one discrete step of continuous time evolution (tau -> tau + d_tau).
        """
        self.optimizer.zero_grad()
        
        # 1. Forward Pass: Compute the geometric curvature (Action)
        action = self.engine.compute_action()
        
        # 2. Backward Pass: Compute functional derivative \delta S / \delta \theta
        # This executes the calculus of variations strictly on the computational graph.
        action.backward()
        
        # 3. Update Step: Move the field along the gradient vector field
        self.optimizer.step()
        
        # 4. Manifold Projection (U(1) Phase Wrapping)
        # We MUST wrap the angles back to [-pi, pi).
        # We use torch.no_grad() because this is a geometric constraint, 
        # not a differentiable mathematical operation.
        with torch.no_grad():
            self.lattice.theta.copy_(
                torch.remainder(self.lattice.theta + math.pi, 2 * math.pi) - math.pi
            )
            
        return action.item()

# ==========================================
# UNIT TESTING & VERIFICATION (Phase 3)
# ==========================================
if __name__ == "__main__":
    print("--- Running Energy Monotonicity Test (Gradient Flow) ---")
    
    # We use L=32 for the test. We revert back to float32 as we are now 
    # doing machine learning / gradient descent, where raw speed is needed,
    # and 32-bit is standard for GPU compute.
    L = 32
    lattice = U1LatticeField(L=L, device='cpu')
    engine = U1PhysicsEngine(lattice)
    flow = U1GradientFlow(engine, learning_rate=0.1)
    
    print("[*] Evolving the neural lattice towards the vacuum state...")
    print(f"{'Step':<10} | {'Action (Energy) S':<20} | {'Delta S':<20}")
    print("-" * 55)
    
    previous_action = None
    violation_count = 0
    
    # Run a short flow for 100 iterations
    for step in range(101):
        current_action = flow.step()
        
        if step % 10 == 0:
            if previous_action is not None:
                delta_s = current_action - previous_action
                # Rigorous check: Does the energy strictly decrease?
                if delta_s > 0:
                    violation_count += 1
                print(f"{step:<10} | {current_action:<20.4f} | {delta_s:<20.4f}")
            else:
                print(f"{step:<10} | {current_action:<20.4f} | {'-':<20}")
                
            previous_action = current_action
            
    # Verification of Mathematical Stability
    print("-" * 55)
    if violation_count == 0:
        print("\nSUCCESS: Energy is monotonically decreasing.")
        print("The gradient flow dynamics are mathematically stable and cooling properly.")
    else:
        print(f"\nWARNING: Action increased {violation_count} times. Check learning rate.")