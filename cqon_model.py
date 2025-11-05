"""
Coherent Quantum Oscillator Network (CQON) Model - Real Implementation
Quantum coherence mediated energy-to-information transformation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class CQONSimulation:
"""
CQON Model: Emergence of life through quantum coherence
"""

def __init__(self, alpha=0.3, gamma=0.08, K0=1.0, T=0.15,
             grid_size=12, total_time=100, dt=0.1):
    # Parameters with physical meaning
    self.alpha = alpha  # Resonance sensitivity - how quickly coherence builds
    self.gamma = gamma  # Decoherence rate - environmental disruption
    self.K0 = K0  # Coupling strength - energy scale
    self.T = T  # Thermal noise - randomness in system
    self.grid_size = grid_size
    self.total_time = total_time
    self.dt = dt

    # Core CQON arrays
    self.coherence = None  # c_i - quantum coherence level [0,1]
    self.phases = None  # θ_i - quantum phases
    self.energy_density = None  # E_i - local energy

def initialize_system(self):
    """Initialize CQON field with random phases and low coherence"""
    self.coherence = np.random.uniform(0.05, 0.2,
                                       (self.grid_size, self.grid_size))
    self.phases = np.random.uniform(0, 2 * np.pi,
                                    (self.grid_size, self.grid_size))
    self.energy_density = np.zeros((self.grid_size, self.grid_size))

def calculate_local_resonance(self, i, j):
    """
    Calculate R_local = (1/N) Σ cos(θ_i - θ_j)
    This is the CORE of CQON theory - phase alignment creates coherence
    """
    total_cos = 0
    count = 0

    # Check all 8 neighbors (quantum entanglement-like coupling)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue  # Skip self

            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                phase_diff = self.phases[i, j] - self.phases[ni, nj]
                total_cos += np.cos(phase_diff)  # Phase alignment measure
                count += 1

    return total_cos / count if count > 0 else 0

def update_coherence(self):
    """
    dc_i/dt = α·R_local - γ·c_i + noise
    This is the KEY EQUATION of CQON theory
    """
    new_coherence = self.coherence.copy()

    for i in range(self.grid_size):
        for j in range(self.grid_size):
            # 1. Resonance building (α·R_local) - coherence increases with phase alignment
            R_local = self.calculate_local_resonance(i, j)
            resonance_effect = self.alpha * R_local

            # 2. Decoherence loss (γ·c_i) - environment destroys coherence
            decoherence_loss = self.gamma * self.coherence[i, j]

            # 3. Quantum noise - random fluctuations
            noise = np.sqrt(2 * self.T) * np.random.normal(0, 1) * np.sqrt(self.dt)

            # Update coherence (CQON core equation)
            dc_dt = resonance_effect - decoherence_loss + noise
            new_coherence[i, j] = np.clip(self.coherence[i, j] + dc_dt * self.dt, 0, 1)

            # Update energy density: E_i = K₀·c_i·R_local
            # Energy becomes structured when coherence and resonance align
            self.energy_density[i, j] = self.K0 * new_coherence[i, j] * R_local

            # Phase evolution - affected by local resonance
            phase_noise = 0.02 * np.random.normal()
            dtheta_dt = 0.08 * R_local + phase_noise
            self.phases[i, j] = (self.phases[i, j] + dtheta_dt * self.dt) % (2 * np.pi)

    self.coherence = new_coherence

def find_coherence_islands(self, threshold=0.6):
    """
    Identify coherence islands - proto-information structures
    These are the 'seeds' of life-like organization
    """
    high_coherence = self.coherence > threshold
    labeled_array, num_islands = ndimage.label(high_coherence)
    return num_islands, labeled_array

def calculate_entropy(self):
    """Calculate information entropy from coherence distribution"""
    epsilon = 1e-10
    entropy_terms = -self.coherence * np.log(self.coherence + epsilon)
    entropy_terms -= (1 - self.coherence) * np.log(1 - self.coherence + epsilon)
    return np.sum(entropy_terms)

def run(self, verbose=True):
    """
    Run the complete CQON simulation
    Demonstrates the core theory: Energy → Coherence → Information
    """
    if verbose:
        print("🚀 Starting CQON Simulation - Quantum Coherence Model")
        print("📖 Theory: Energy → Coherence → Information Transformation")
        print(f"🔬 Parameters: α={self.alpha} (resonance), γ={self.gamma} (decoherence), "
              f"K₀={self.K0} (energy), T={self.T} (noise)")

    self.initialize_system()

    # Track evolution
    energy_history = []
    entropy_history = []
    coherence_history = []
    islands_history = []

    for step in range(int(self.total_time / self.dt)):
        self.update_coherence()

        # Calculate metrics
        total_energy = np.sum(self.energy_density)
        entropy = self.calculate_entropy()
        num_islands, _ = self.find_coherence_islands()
        avg_coherence = np.mean(self.coherence)

        energy_history.append(total_energy)
        entropy_history.append(entropy)
        coherence_history.append(avg_coherence)
        islands_history.append(num_islands)

        if verbose and step % 50 == 0:
            print(f"⏰ t={step * self.dt:.1f}: ⟨c⟩={avg_coherence:.3f}, "
                  f"E={total_energy:.1f}, S={entropy:.1f}, Islands={num_islands}")

    # Final analysis
    correlation = np.corrcoef(energy_history, entropy_history)[0, 1]
    life_like = (coherence_history[-1] > 0.35 and
                 islands_history[-1] >= 1 and
                 correlation < -0.3)

    results = {
        'final_energy': energy_history[-1],
        'final_entropy': entropy_history[-1],
        'avg_coherence': coherence_history[-1],
        'coherence_islands': islands_history[-1],
        'energy_entropy_correlation': correlation,
        'life_like_organization': life_like,
        'energy_history': energy_history,
        'entropy_history': entropy_history,
        'coherence_history': coherence_history,
        'theory_explanation': self.get_theory_explanation()
    }

    if verbose:
        print(f"\n✅ Simulation completed!")
        print(f"📊 Final: ⟨c⟩={results['avg_coherence']:.3f}, "
              f"E={results['final_energy']:.1f}, S={results['final_entropy']:.1f}")
        print(f"🔗 Energy-Entropy correlation: {results['energy_entropy_correlation']:.3f}")
        print(f"🌊 Coherence islands: {results['coherence_islands']}")
        print(f"🎯 Life-like organization: {'YES' if life_like else 'NO'}")

    return results

def get_theory_explanation(self):
    """Explain the CQON theory in simple terms"""
    return {
        'core_idea': "Quantum coherence transforms energy flow into structured information",
        'key_equation': "dc_i/dt = α·R_local - γ·c_i + noise",
        'energy_transformation': "E_i = K₀·c_i·R_local - energy becomes information when coherent",
        'life_threshold': "System self-organizes when α high, γ low, T moderate",
        'emergence_process': "Random phases → Local resonance → Coherence islands → Information structures"
    }

def demonstrate_theory():
"""
Demonstrate the core CQON theory concepts
"""
print("🔬 CQON THEORY DEMONSTRATION")
print("=" * 50)

# Test different parameter regimes
scenarios = [
    {"name": "LIFE-LIKE", "alpha": 0.5, "gamma": 0.05, "T": 0.1},
    {"name": "DISORDERED", "alpha": 0.1, "gamma": 0.2, "T": 0.3},
    {"name": "MARGINAL", "alpha": 0.3, "gamma": 0.1, "T": 0.2}
]

for scenario in scenarios:
    print(f"\n🧪 Testing {scenario['name']} scenario:")
    sim = CQONSimulation(alpha=scenario['alpha'],
                        gamma=scenario['gamma'],
                        T=scenario['T'])
    results = sim.run(verbose=False)

    print(f"   Final coherence: {results['avg_coherence']:.3f}")
    print(f"   Coherence islands: {results['coherence_islands']}")
    print(f"   Life-like: {'YES' if results['life_like_organization'] else 'NO'}")
    print(f"   E-S correlation: {results['energy_entropy_correlation']:.3f}")

    if name == "main":
    # Run theory demonstration
    demonstrate_theory()

    print("\n" + "=" * 50)
    print("🎯 Running main CQON simulation with realistic parameters...")

    # Main simulation
    sim = CQONSimulation(alpha=0.3, gamma=0.08, K0=1.0, T=0.15)
    results = sim.run()

    print(f"\n📖 THEORY SUMMARY:")
    theory = results['theory_explanation']
    for key, value in theory.items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")

