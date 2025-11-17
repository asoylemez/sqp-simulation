import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Simple and safe style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.2


def create_life_threshold_region():
    """
    Figure 1: Emergence of Life Threshold Region in α-T Parameter Space
    UPDATED with standardized thresholds
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ==================== PANEL A: LIFE THRESHOLD REGION ====================

    # Create parameter space
    alpha_range = np.linspace(0.1, 0.8, 100)
    T_range = np.linspace(0.01, 0.35, 100)
    Alpha, T_vals = np.meshgrid(alpha_range, T_range)

    # Life probability function - UPDATED for standardized thresholds
    life_probability = np.exp(-((Alpha - 0.32) ** 2 / 0.04 + (T_vals - 0.15) ** 2 / 0.006))
    life_probability = np.clip(life_probability, 0, 1)

    # Contour plot - simple color map
    contour = ax1.contourf(Alpha, T_vals, life_probability,
                           levels=50, cmap='RdYlGn', alpha=0.9)

    # Critical threshold lines - UPDATED levels
    CS = ax1.contour(Alpha, T_vals, life_probability,
                     levels=[0.3, 0.5, 0.7], colors=['white', 'black', 'white'],
                     linewidths=[2, 3, 2], linestyles=['--', '-', '--'])

    ax1.clabel(CS, fmt={0.3: 'Low', 0.5: 'Life Threshold', 0.7: 'High'},
               fontsize=11, colors='black')

    # Scenario points - UPDATED with standardized classification
    scenarios = {
        'Optimal Quantum': {'alpha': 0.45, 'T': 0.08, 'color': 'darkgreen', 'marker': 'P', 'size': 120,
                            'class': 'HIGH'},
        'Early Earth': {'alpha': 0.35, 'T': 0.15, 'color': 'limegreen', 'marker': 'o', 'size': 100, 'class': 'HIGH'},
        'Ocean Depth': {'alpha': 0.38, 'T': 0.12, 'color': 'mediumseagreen', 'marker': '^', 'size': 100,
                        'class': 'HIGH'},
        'Past Mars': {'alpha': 0.30, 'T': 0.18, 'color': 'darkorange', 'marker': 'v', 'size': 100, 'class': 'MEDIUM'},
        'Present Mars': {'alpha': 0.20, 'T': 0.28, 'color': 'red', 'marker': 's', 'size': 100, 'class': 'LOW'}
    }

    for name, params in scenarios.items():
        ax1.scatter(params['alpha'], params['T'], c=params['color'],
                    s=params['size'], marker=params['marker'],
                    edgecolors='black', linewidth=1.5,
                    label=f"{name} ({params['class']})", zorder=5)

    ax1.set_xlabel('Resonance Sensitivity (α)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Thermal Noise (T)', fontweight='bold', fontsize=12)
    ax1.set_title('(A) Life-Threshold Region\nα-T Parameter Space',
                  fontweight='bold', fontsize=14)

    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', framealpha=0.95)

    # Color bar
    cbar = plt.colorbar(contour, ax=ax1, shrink=0.8)
    cbar.set_label('Life-like Organization Probability',
                   rotation=270, labelpad=20, fontweight='bold')

    # ==================== PANEL B: CRITICAL PARAMETER ANALYSIS ====================

    # Coherence evolution vs α at fixed T - UPDATED thresholds
    T_fixed = 0.15
    alpha_test = np.linspace(0.1, 0.8, 50)

    # Coherence values - UPDATED for standardized behavior
    coherence_values = 0.1 + 0.7 * (1 - np.exp(-5 * (alpha_test - 0.25)))
    coherence_values = np.clip(coherence_values, 0.1, 0.8)

    # Highlight life threshold region - UPDATED boundaries
    life_region = (alpha_test > 0.30) & (alpha_test < 0.55)
    high_region = (alpha_test > 0.35) & (alpha_test < 0.50)

    ax2.plot(alpha_test, coherence_values, 'b-', linewidth=3,
             label='Mean Coherence ⟨c⟩')

    # Fill regions with standardized classification colors
    ax2.fill_between(alpha_test[life_region], 0.30, coherence_values[life_region],
                     alpha=0.3, color='orange', label='MEDIUM Region')
    ax2.fill_between(alpha_test[high_region], 0.30, coherence_values[high_region],
                     alpha=0.4, color='green', label='HIGH Region')

    # Critical threshold lines - UPDATED values
    ax2.axhline(y=0.30, color='red', linestyle='--', linewidth=2,
                label='Critical Coherence (0.30)')
    ax2.axvline(x=0.30, color='orange', linestyle='--', linewidth=2,
                label='Min α (0.30)')
    ax2.axvline(x=0.35, color='green', linestyle='--', linewidth=2,
                label='HIGH α (0.35)')

    # Mark specific scenarios
    scenario_alphas = {
        'Present Mars': 0.20,
        'Past Mars': 0.30,
        'Early Earth': 0.35,
        'Optimal': 0.45
    }

    for scenario, alpha_val in scenario_alphas.items():
        coh_val = 0.1 + 0.7 * (1 - np.exp(-5 * (alpha_val - 0.25)))
        color = 'red' if scenario == 'Present Mars' else 'orange' if scenario == 'Past Mars' else 'green'
        ax2.scatter(alpha_val, coh_val, color=color, s=80, zorder=5, edgecolors='black', linewidth=1)
        ax2.annotate(scenario, (alpha_val, coh_val), xytext=(5, 5),
                     textcoords='offset points', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Resonance Sensitivity (α)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Mean Coherence ⟨c⟩', fontweight='bold', fontsize=12)
    ax2.set_title('(B) Critical Parameter Analysis\nStandardized Classification',
                  fontweight='bold', fontsize=14)

    ax2.set_ylim(0.1, 0.85)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', framealpha=0.95)

    # ==================== GRAPH SETTINGS ====================

    plt.suptitle('EMERGENCE OF LIFE-THRESHOLD REGION - STANDARDIZED CQON MODEL',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # Save with updated names
    plt.savefig('Figure1_Life_Threshold_Region_Standardized.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure1_Life_Threshold_Region_Standardized.pdf', dpi=300, bbox_inches='tight')

    plt.show()

    return fig


def create_parameter_sensitivity_analysis():
    """
    NEW: Additional figure showing parameter sensitivity with standardized thresholds
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ==================== PANEL A: α-γ Parameter Space ====================

    alpha_range = np.linspace(0.1, 0.6, 100)
    gamma_range = np.linspace(0.02, 0.2, 100)
    Alpha, Gamma = np.meshgrid(alpha_range, gamma_range)

    # Life probability in α-γ space
    life_prob_ag = np.exp(-((Alpha - 0.35) ** 2 / 0.03 + (Gamma - 0.08) ** 2 / 0.002))
    life_prob_ag = np.clip(life_prob_ag, 0, 1)

    contour1 = ax1.contourf(Alpha, Gamma, life_prob_ag, levels=50, cmap='RdYlGn', alpha=0.9)

    # Standardized scenario points
    scenarios_ag = {
        'Early Earth': {'alpha': 0.35, 'gamma': 0.07, 'color': 'limegreen', 'marker': 'o', 'class': 'HIGH'},
        'Past Mars': {'alpha': 0.30, 'gamma': 0.09, 'color': 'orange', 'marker': 'v', 'class': 'MEDIUM'},
        'Present Mars': {'alpha': 0.20, 'gamma': 0.12, 'color': 'red', 'marker': 's', 'class': 'LOW'},
        'Optimal': {'alpha': 0.45, 'gamma': 0.05, 'color': 'darkgreen', 'marker': 'P', 'class': 'HIGH'}
    }

    for name, params in scenarios_ag.items():
        ax1.scatter(params['alpha'], params['gamma'], c=params['color'],
                    s=100, marker=params['marker'], edgecolors='black', linewidth=1.5,
                    label=f"{name} ({params['class']})", zorder=5)

    ax1.set_xlabel('Resonance Sensitivity (α)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Decoherence Rate (γ)', fontweight='bold', fontsize=12)
    ax1.set_title('(A) α-γ Parameter Space\nStandardized Classification',
                  fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.colorbar(contour1, ax=ax1, shrink=0.8, label='Organization Probability')

    # ==================== PANEL B: Triple Parameter Sensitivity ====================

    # Fixed T analysis
    T_fixed = 0.15
    alpha_vals = np.linspace(0.1, 0.6, 50)
    gamma_vals = np.linspace(0.05, 0.15, 50)

    # Calculate organization score
    org_scores = np.zeros((len(gamma_vals), len(alpha_vals)))
    for i, gamma in enumerate(gamma_vals):
        for j, alpha in enumerate(alpha_vals):
            # Simple model for organization score
            if alpha > 0.30 and gamma < 0.10:
                score = min(1.0, (alpha - 0.30) * 5 * (0.10 - gamma) * 20)
            else:
                score = max(0, (alpha - 0.25) * 4 * (0.12 - gamma) * 15)
            org_scores[i, j] = np.clip(score, 0, 1)

    contour2 = ax2.contourf(alpha_vals, gamma_vals, org_scores, levels=50, cmap='RdYlGn', alpha=0.9)

    # Add classification boundaries
    ax2.contour(alpha_vals, gamma_vals, org_scores, levels=[0.3, 0.6],
                colors=['white', 'black'], linewidths=[2, 3])

    # Standardized classification regions
    ax2.text(0.45, 0.06, 'HIGH', fontsize=12, fontweight='bold',
             ha='center', va='center', color='white')
    ax2.text(0.35, 0.08, 'MEDIUM', fontsize=11, fontweight='bold',
             ha='center', va='center', color='black')
    ax2.text(0.25, 0.11, 'LOW', fontsize=10, fontweight='bold',
             ha='center', va='center', color='black')

    ax2.set_xlabel('Resonance Sensitivity (α)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Decoherence Rate (γ)', fontweight='bold', fontsize=12)
    ax2.set_title('(B) Organization Classification\nT = 0.15 (Fixed)',
                  fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.colorbar(contour2, ax=ax2, shrink=0.8, label='Organization Score')

    plt.suptitle('PARAMETER SENSITIVITY ANALYSIS - STANDARDIZED CQON MODEL',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    plt.savefig('Figure2_Parameter_Sensitivity_Standardized.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure2_Parameter_Sensitivity_Standardized.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


# Create the graphs
print("🚀 Creating Life-Threshold Region graph with standardized thresholds...")
figure1 = create_life_threshold_region()

print("🚀 Creating Parameter Sensitivity Analysis graph...")
figure2 = create_parameter_sensitivity_analysis()

print("✅ All graphs successfully created and saved!")