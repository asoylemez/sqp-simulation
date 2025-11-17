"""
Coherent Quantum Oscillator Network (CQON) Model - Updated Implementation
Quantum coherence mediated energy-to-information transformation
Standardized thresholds based on research paper
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import pearsonr


class CQONSimulation:
    """
    CQON Model: Emergence of life through quantum coherence
    Updated with standardized thresholds and classification system
    """

    # STANDARDIZED THRESHOLDS (Based on research paper)
    COHERENCE_THRESHOLD = 0.30  # ⟨c⟩ threshold for life-like organization
    ISLANDS_THRESHOLD = 3  # Minimum coherence islands
    CORRELATION_THRESHOLD = -0.45  # Energy-entropy correlation threshold

    # CLASSIFICATION BOUNDARIES
    HIGH_COHERENCE = 0.30  # HIGH classification threshold
    MEDIUM_COHERENCE = 0.25  # MEDIUM classification threshold
    LOW_COHERENCE = 0.15  # LOW classification threshold

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
        self.energy_density = None  # E_i - local energy density

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
        Core of CQON theory - phase alignment creates coherence
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
        KEY EQUATION of CQON theory
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

    def find_coherence_islands(self):
        """
        Identify coherence islands - proto-information structures
        Using standardized threshold: 0.30
        """
        high_coherence = self.coherence > self.COHERENCE_THRESHOLD
        labeled_array, num_islands = ndimage.label(high_coherence)
        return num_islands, labeled_array

    def calculate_entropy(self):
        """Calculate information entropy from coherence distribution"""
        epsilon = 1e-10
        entropy_terms = -self.coherence * np.log(self.coherence + epsilon)
        entropy_terms -= (1 - self.coherence) * np.log(1 - self.coherence + epsilon)
        return np.sum(entropy_terms)

    def classify_organization(self, avg_coherence, num_islands, correlation):
        """
        Classify system organization based on standardized criteria
        Returns: classification string and boolean for life-like
        """
        # Check for HIGH (life-like) organization
        life_like = (avg_coherence > self.COHERENCE_THRESHOLD and
                     num_islands >= self.ISLANDS_THRESHOLD and
                     correlation < self.CORRELATION_THRESHOLD)

        if life_like:
            return "HIGH", True
        elif avg_coherence > self.MEDIUM_COHERENCE:
            return "MEDIUM", False
        elif avg_coherence > self.LOW_COHERENCE:
            return "LOW", False
        else:
            return "NO", False

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
            print(f"🎯 Standardized Thresholds: ⟨c⟩>{self.COHERENCE_THRESHOLD}, "
                  f"islands≥{self.ISLANDS_THRESHOLD}, r<{self.CORRELATION_THRESHOLD}")

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

        # Final analysis with standardized classification
        correlation = pearsonr(energy_history, entropy_history)[0]
        classification, life_like = self.classify_organization(
            coherence_history[-1], islands_history[-1], correlation
        )

        results = {
            'final_energy': energy_history[-1],
            'final_entropy': entropy_history[-1],
            'avg_coherence': coherence_history[-1],
            'coherence_islands': islands_history[-1],
            'energy_entropy_correlation': correlation,
            'life_like_organization': life_like,
            'organization_classification': classification,
            'energy_history': energy_history,
            'entropy_history': entropy_history,
            'coherence_history': coherence_history,
            'islands_history': islands_history,
            'theory_explanation': self.get_theory_explanation(),
            'thresholds_used': self.get_thresholds()
        }

        if verbose:
            self.print_final_results(results)

        return results

    def print_final_results(self, results):
        """Print standardized final results"""
        print(f"\n✅ Simulation completed!")
        print(f"📊 Final Metrics:")
        print(f"   • Average Coherence (⟨c⟩): {results['avg_coherence']:.3f}")
        print(f"   • Total Energy: {results['final_energy']:.1f}")
        print(f"   • Information Entropy: {results['final_entropy']:.1f}")
        print(f"   • Coherence Islands: {results['coherence_islands']}")
        print(f"   • Energy-Entropy Correlation: {results['energy_entropy_correlation']:.3f}")
        print(f"🎯 Organization Classification: {results['organization_classification']}")

        # Detailed threshold analysis
        print(f"\n🔍 Threshold Analysis:")
        print(
            f"   • Coherence > {self.COHERENCE_THRESHOLD}: {results['avg_coherence']:.3f} → {'✓' if results['avg_coherence'] > self.COHERENCE_THRESHOLD else '✗'}")
        print(
            f"   • Islands ≥ {self.ISLANDS_THRESHOLD}: {results['coherence_islands']} → {'✓' if results['coherence_islands'] >= self.ISLANDS_THRESHOLD else '✗'}")
        print(
            f"   • Correlation < {self.CORRELATION_THRESHOLD}: {results['energy_entropy_correlation']:.3f} → {'✓' if results['energy_entropy_correlation'] < self.CORRELATION_THRESHOLD else '✗'}")

    def get_theory_explanation(self):
        """Explain the CQON theory in simple terms"""
        return {
            'core_idea': "Quantum coherence transforms energy flow into structured information",
            'key_equation': "dc_i/dt = α·R_local - γ·c_i + noise",
            'energy_transformation': "E_i = K₀·c_i·R_local - energy becomes information when coherent",
            'life_threshold': f"System self-organizes when ⟨c⟩>{self.COHERENCE_THRESHOLD}, islands≥{self.ISLANDS_THRESHOLD}, r<{self.CORRELATION_THRESHOLD}",
            'emergence_process': "Random phases → Local resonance → Coherence islands → Information structures"
        }

    def get_thresholds(self):
        """Return the standardized thresholds used"""
        return {
            'coherence_threshold': self.COHERENCE_THRESHOLD,
            'islands_threshold': self.ISLANDS_THRESHOLD,
            'correlation_threshold': self.CORRELATION_THRESHOLD,
            'high_coherence': self.HIGH_COHERENCE,
            'medium_coherence': self.MEDIUM_COHERENCE,
            'low_coherence': self.LOW_COHERENCE
        }


def demonstrate_standardized_scenarios():
    """
    Demonstrate different scenarios using standardized classification
    """
    print("🔬 CQON STANDARDIZED SCENARIO DEMONSTRATION")
    print("=" * 60)

    # Test scenarios covering all classification categories
    scenarios = [
        {
            "name": "HIGH ORGANIZATION (Life-like)",
            "params": {"alpha": 0.5, "gamma": 0.05, "T": 0.1},
            "expected": "HIGH"
        },
        {
            "name": "MEDIUM ORGANIZATION",
            "params": {"alpha": 0.3, "gamma": 0.1, "T": 0.15},
            "expected": "MEDIUM"
        },
        {
            "name": "LOW ORGANIZATION",
            "params": {"alpha": 0.2, "gamma": 0.15, "T": 0.25},
            "expected": "LOW"
        },
        {
            "name": "NO ORGANIZATION",
            "params": {"alpha": 0.1, "gamma": 0.2, "T": 0.3},
            "expected": "NO"
        }
    ]

    for scenario in scenarios:
        print(f"\n🧪 {scenario['name']}:")
        print("-" * 40)

        sim = CQONSimulation(**scenario['params'])
        results = sim.run(verbose=False)

        print(
            f"   Parameters: α={scenario['params']['alpha']}, γ={scenario['params']['gamma']}, T={scenario['params']['T']}")
        print(
            f"   Results: ⟨c⟩={results['avg_coherence']:.3f}, islands={results['coherence_islands']}, r={results['energy_entropy_correlation']:.3f}")
        print(f"   Classification: {results['organization_classification']} (expected: {scenario['expected']})")
        print(f"   Life-like: {'✓' if results['life_like_organization'] else '✗'}")


def run_planetary_comparison():
    """
    Compare different planetary scenarios using standardized thresholds
    """
    print("\n🪐 PLANETARY SCENARIO COMPARISON")
    print("=" * 60)

    planetary_scenarios = [
        {
            "name": "EARLY EARTH",
            "description": "High energy flow, moderate decoherence",
            "params": {"alpha": 0.45, "gamma": 0.08, "T": 0.12}
        },
        {
            "name": "PAST MARS",
            "description": "Moderate energy, higher decoherence",
            "params": {"alpha": 0.35, "gamma": 0.12, "T": 0.18}
        },
        {
            "name": "PRESENT MARS",
            "description": "Low energy, high decoherence",
            "params": {"alpha": 0.15, "gamma": 0.18, "T": 0.25}
        },
        {
            "name": "EUROPA (Potential)",
            "description": "Moderate energy, low thermal noise",
            "params": {"alpha": 0.4, "gamma": 0.1, "T": 0.08}
        }
    ]

    for planet in planetary_scenarios:
        print(f"\n🌍 {planet['name']}: {planet['description']}")
        print("-" * 50)

        sim = CQONSimulation(**planet['params'])
        results = sim.run(verbose=False)

        print(f"   Final Coherence: {results['avg_coherence']:.3f}")
        print(f"   Coherence Islands: {results['coherence_islands']}")
        print(f"   E-S Correlation: {results['energy_entropy_correlation']:.3f}")
        print(f"   🎯 Classification: {results['organization_classification']}")
        print(
            f"   Life-like Potential: {'HIGH' if results['life_like_organization'] else 'MEDIUM' if results['organization_classification'] == 'MEDIUM' else 'LOW'}")


if __name__ == "__main__":
    # Display standardized thresholds
    print("🎯 CQON MODEL - STANDARDIZED THRESHOLDS")
    print("=" * 50)
    print(f"• Coherence Threshold (⟨c⟩): > {CQONSimulation.COHERENCE_THRESHOLD}")
    print(f"• Islands Threshold: ≥ {CQONSimulation.ISLANDS_THRESHOLD}")
    print(f"• Correlation Threshold (r): < {CQONSimulation.CORRELATION_THRESHOLD}")
    print(f"• Classification: HIGH/MEDIUM/LOW/NO")
    print("=" * 50)

    # Run demonstrations
    demonstrate_standardized_scenarios()
    run_planetary_comparison()

    print("\n" + "=" * 60)
    print("🎯 Running main CQON simulation with Earth-like parameters...")
    print("=" * 60)

    # Main simulation with Earth-like parameters
    sim = CQONSimulation(alpha=0.35, gamma=0.08, K0=1.0, T=0.12)
    results = sim.run()

    print(f"\n📖 THEORY SUMMARY:")
    theory = results['theory_explanation']
    for key, value in theory.items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")

    print(f"\n⚙️ THRESHOLDS USED:")
    thresholds = results['thresholds_used']
    for key, value in thresholds.items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")