🔬 Coherent Quantum Oscillator Network (CQON) Simulation
A novel physical framework for the emergence of life through quantum coherence.

## 🌟 Overview
This repository implements the Coherent Quantum Oscillator Network (CQON) model, a new physical framework for studying how life emerges through quantum coherence and energy-information transformation. The model demonstrates how quantum coherence can mediate the direct conversion of energy flow into structured information.

## 📄 Preprint Status
**Manuscript Title:** "Energy-Information Transformation: Coherent Quantum Oscillator Network Model for the Emergence of Life via Quantum Coherence"  
**Status:** Submitted to BioRxiv - Under Review  
**Official Link:** [Coming soon after approval]  
**Authors:** Akın Söylemez

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate Figure 1 - Life Threshold Region
python Life_Threshold_Zone_Simulation.py

# Generate Figures 2, 3, 4 - Phase Map, Energy-Entropy, Hierarchy
python 2_3_4_Figurs.py

# Run realistic scenario testing
python realistic_test.py

📁 Repository Structure
cqon-simulation/
├── Life_Threshold_Zone_Simulation.py  # Generates Figure 1 (Life-Threshold Region)
├── 2_3_4_Figurs.py                    # Generates Figures 2, 3, 4
├── cqon_model.py                      # Core CQON simulation class
├── realistic_test.py                   # Realistic scenario testing
├── requirements.txt                    # Python dependencies
├── Figure_1_EN.png                    # Life-Threshold Region (English)
├── Figure_2_EN.png                    # Phase Map (English)
├── Figure_3_EN.png                    # Energy-Entropy Evolution (English)
├── Figure_4_EN.png                    # Multi-Scale Hierarchy (English)
└── README.md                          # This file

🔧 Requirements
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
seaborn>=0.11.0

🎯 Key Findings
Life-Threshold Region: Identified at K₀ ≥ 0.8, α ≥ 0.4, T ≤ 0.15
Energy-Information Transformation: Strong inverse correlation (r = -0.873)
Coherence Growth: 246% increase under Early Earth conditions (0.121 → 0.419)
Spontaneous Organization: 13-18 coherence islands emerge spontaneously
Environmental Robustness: Works across noise levels (T = 0.08-0.22)

📊 Experimental Validation
Scenario	Parameters	Avg Coherence	Islands	E-S Correlation	Life-Like
Early Earth	α=0.35, γ=0.07, T=0.15	0.355	14	-0.562	✅ HIGH
Optimal Quantum	α=0.45, γ=0.05, T=0.08	0.320	13	-0.678	✅ Medium
Ocean Depth	α=0.38, γ=0.04, T=0.12	0.306	18	-0.586	✅ Medium
High Noise	α=0.28, γ=0.10, T=0.22	0.315	13	-0.254	⚠️ Low

🔬 Early Earth Simulation Results
Under prebiotic Earth-like conditions (α=0.35, T=0.15):
Coherence Growth: 0.121 → 0.419 (246% increase)
Information Islands: 13 stable coherence regions
Energy-Entropy Correlation: Strong inverse relationship (r = -0.543)
Life-Like Organization: ✅ CONFIRMED

🎯 Model Parameters
α (Resonance Sensitivity): 0.01-1.0 [1/time]
γ (Decoherence Rate): 0.001-0.1 [1/time]
K₀ (Coupling Strength): 0.1-2.0 [energy units]
T (Noise Intensity): 0.01-0.5 [dimensionless]

📄 Citation
If you use this code in your research, please cite:
bibtex
@software{cqon_simulation2024,
  title = {Coherent Quantum Oscillator Network Model for the Emergence of Life},
  author = {Söylemez, Akın},
  year = {2024},
  url = {https://github.com/asoylemez/cqon-simulation}

🔬 Philosophical Implications
Our results support the hypothesis that life represents a natural phase transition in the energy-information continuum:
Universal Life Principle: Environments with quantum coherence can generate proto-organic structures
Information as Physics: Information is fundamental to physical organization
Redefined Habitability: Life-search should focus on "coherence ecosystems"

📞 Contact
Author: Akın Söylemez
Email: soylemez.akin@gmail.com
GitHub: asoylemez
Repository: https://github.com/asoylemez/cqon-simulation

📜 License
This project is licensed under the MIT License - see the repository for details.