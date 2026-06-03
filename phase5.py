import torch
import numpy as np
import pyvista as pv
import os
import time

from phase1 import U1LatticeField
from phase2 import U1PhysicsEngine
from phase3 import U1GradientFlow
from phase4 import U1Observables

class U1SimulationEngine:
    """
    Phase 5: The Grand 3D Simulation Engine.
    
    Executes the massive 512^3 gradient flow on Apple Silicon (MPS).
    Extracts topological energy density and exports to volumetric .vti files
    for final IA 3D rendering.
    """
    def __init__(self, L: int = 512, device: str = 'mps', lr: float = 0.05):
        print(f"=== Initializing U(1) Gauge Theory Simulation [{L}^3] ===")
        self.L = L
        self.lattice = U1LatticeField(L=L, device=device)
        self.engine = U1PhysicsEngine(self.lattice)
        self.flow = U1GradientFlow(self.engine, learning_rate=lr)
        self.observables = U1Observables(self.engine)
        
        # Create output directory for 3D frames
        self.output_dir = "simulation_frames"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[*] Data will be exported to ./{self.output_dir}/")

    def export_vti(self, step: int):
        """
        Extracts the energy density field from the GPU, converts it to 
        a standard VTK ImageData format, and writes to SSD.
        """
        # 1. Compute physical observable on GPU
        # We use torch.no_grad() to ensure we don't accidentally build a compute graph
        with torch.no_grad():
            energy_tensor = self.observables.compute_energy_density()
            
            # 2. Transfer to CPU and convert to Numpy
            # Float32 is sufficient for visualization
            energy_np = energy_tensor.cpu().numpy().astype(np.float32)
            
        # 3. Construct PyVista 3D Grid
        grid = pv.ImageData(dimensions=(self.L, self.L, self.L))
        
        # VTK strictly expects data in Fortran-contiguous order (X changes fastest)
        grid.point_data["MagneticEnergyDensity"] = energy_np.flatten(order="F")
        
        # 4. Write to disk
        filename = os.path.join(self.output_dir, f"flux_tubes_step_{step:05d}.vti")
        grid.save(filename)
        print(f"    -> Exported volumetric data: {filename}")

    def run(self, total_steps: int = 5000, save_interval: int = 500):
        """
        The main simulation loop.
        """
        print("\n[*] Commencing Spatiotemporal Gradient Flow...")
        print(f"{'Step':<8} | {'Total Action (S)':<18} | {'Time/Step (s)':<15}")
        print("-" * 45)
        
        # Export the initial chaotic state (Step 0)
        self.export_vti(0)
        
        start_time = time.time()
        
        for step in range(1, total_steps + 1):
            t0 = time.time()
            
            # Execute one step of the continuous Euler-Lagrange evolution
            current_action = self.flow.step()
            
            t1 = time.time()
            
            if step % 50 == 0 or step == 1:
                step_time = t1 - t0
                print(f"{step:<8} | {current_action:<18.2f} | {step_time:<15.4f}")
                
            if step % save_interval == 0:
                self.export_vti(step)
                
        total_time = (time.time() - start_time) / 60
        print(f"\n=== Simulation Complete in {total_time:.2f} minutes ===")

# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # Rigorous Target: 512^3 lattice on 'mps' (Mac Mini M4)
    # Note: If you want to test it quickly first, change L=128 and total_steps=500.
    
    # Let's run a full scale parameter set for your Mac Mini:
    SIM_SIZE = 512       # Massive grid
    TOTAL_STEPS = 5000   # Will take roughly 1-3 hours depending on M4 sustained cooling
    SAVE_INTERVAL = 500  # Saves 10 frames total
    
    sim = U1SimulationEngine(L=SIM_SIZE, device='mps', lr=0.05)
    sim.run(total_steps=TOTAL_STEPS, save_interval=SAVE_INTERVAL)