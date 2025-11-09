import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Simple and safe style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.2


def create_life_threshold_region():
    """
    Figure 1: Emergence of Life Threshold Region in α-T Parameter Space
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ==================== PANEL A: LIFE THRESHOLD REGION ====================

    # Create parameter space
    alpha_range = np.linspace(0.1, 0.8, 100)
    T_range = np.linspace(0.01, 0.35, 100)
    Alpha, T_vals = np.meshgrid(alpha_range, T_range)

    # Life probability function
    life_probability = np.exp(-((Alpha - 0.35) ** 2 / 0.05 + (T_vals - 0.12) ** 2 / 0.008))
    life_probability = np.clip(life_probability, 0, 1)

    # Contour plot - simple color map
    contour = ax1.contourf(Alpha, T_vals, life_probability,
                           levels=50, cmap='RdYlGn', alpha=0.9)

    # Critical threshold lines
    CS = ax1.contour(Alpha, T_vals, life_probability,
                     levels=[0.3, 0.5, 0.7], colors=['white', 'black', 'white'],
                     linewidths=[2, 3, 2], linestyles=['--', '-', '--'])

    ax1.clabel(CS, fmt={0.3: 'Low', 0.5: 'Life Threshold', 0.7: 'High'},
               fontsize=11, colors='black')

    # Scenario points - SIMPLE MARKERS used
    scenarios = {
        'Optimal Quantum': {'alpha': 0.45, 'T': 0.08, 'color': 'darkgreen', 'marker': 'P', 'size': 120},
        'Early Earth': {'alpha': 0.35, 'T': 0.15, 'color': 'limegreen', 'marker': 'o', 'size': 100},
        'Ocean Depth': {'alpha': 0.38, 'T': 0.12, 'color': 'mediumseagreen', 'marker': '^', 'size': 100},
        'High Noise': {'alpha': 0.28, 'T': 0.22, 'color': 'darkorange', 'marker': 'v', 'size': 100},
        'Critical Threshold': {'alpha': 0.32, 'T': 0.18, 'color': 'gold', 'marker': 's', 'size': 100}
    }

    for name, params in scenarios.items():
        ax1.scatter(params['alpha'], params['T'], c=params['color'],
                    s=params['size'], marker=params['marker'],
                    edgecolors='black', linewidth=1.5,
                    label=name, zorder=5)

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

    # Coherence evolution vs α at fixed T
    T_fixed = 0.15
    alpha_test = np.linspace(0.1, 0.8, 50)

    # Coherence values
    coherence_values = 0.1 + 0.7 * (1 - np.exp(-5 * (alpha_test - 0.25)))
    coherence_values = np.clip(coherence_values, 0.1, 0.8)

    # Highlight life threshold region
    life_region = (alpha_test > 0.3) & (alpha_test < 0.6)

    ax2.plot(alpha_test, coherence_values, 'b-', linewidth=3,
             label='Mean Coherence ⟨c⟩')
    ax2.fill_between(alpha_test[life_region], 0.35, coherence_values[life_region],
                     alpha=0.3, color='green', label='Life-Threshold Region')

    # Critical threshold lines
    ax2.axhline(y=0.35, color='red', linestyle='--', linewidth=2,
                label='Critical Coherence Threshold (0.35)')
    ax2.axvline(x=0.3, color='orange', linestyle='--', linewidth=2,
                label='Min Resonance (α=0.3)')
    ax2.axvline(x=0.6, color='orange', linestyle='--', linewidth=2,
                label='Max Resonance (α=0.6)')

    ax2.set_xlabel('Resonance Sensitivity (α)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Mean Coherence ⟨c⟩', fontweight='bold', fontsize=12)
    ax2.set_title('(B) Critical Parameter Analysis\nT = 0.15 (Early Earth)',
                  fontweight='bold', fontsize=14)

    ax2.set_ylim(0.1, 0.85)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', framealpha=0.95)

    # ==================== GRAPH SETTINGS ====================

    plt.suptitle('EMERGENCE OF LIFE-THRESHOLD REGION VIA QUANTUM COHERENCE',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # Save
    plt.savefig('Figure1_Life_Threshold_Region.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure1_Life_Threshold_Region.pdf', dpi=300, bbox_inches='tight')

    plt.show()

    return fig


# Create the graph
print("🚀 Creating Life-Threshold Region graph...")
figure1 = create_life_threshold_region()
print("✅ Graph successfully created and saved!")