import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter

# Style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# STANDARDIZED THRESHOLDS
COHERENCE_THRESHOLD = 0.30
ISLANDS_THRESHOLD = 3
CORRELATION_THRESHOLD = -0.45

# 1. FIGURE 2: Phase Map of Life-Threshold Region (UPDATED)
print("Creating Figure 2 - Standardized Phase Map...")
K0_range = np.linspace(0.1, 2.0, 50)
alpha_range = np.linspace(0.01, 1.0, 50)
K0_grid, alpha_grid = np.meshgrid(K0_range, alpha_range)

# Model function for mean coherence (⟨c⟩) with standardized thresholds
c_mean = np.minimum(1.0, 0.15 + 0.85 * (K0_grid / 2.0) * (alpha_grid / 1.0)**1.5)
c_mean = gaussian_filter(c_mean, sigma=1.2) # Smoothing

# UPDATED threshold contours with standardized coherence threshold (0.30)
threshold_T02 = 0.12 + 0.5 * (1 - alpha_grid) * (1 - K0_grid/2.0)
threshold_T03 = 0.22 + 0.6 * (1 - alpha_grid) * (1 - K0_grid/2.0)

# UPDATED: Classification regions
high_region = (c_mean > 0.30) & (alpha_grid > 0.35)
medium_region = (c_mean > 0.25) & (c_mean <= 0.30) & (alpha_grid > 0.25)
low_region = (c_mean > 0.15) & (c_mean <= 0.25)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Plot classification regions
ax.contourf(K0_grid, alpha_grid, high_region.astype(float), levels=[0.5, 1], colors=['green'], alpha=0.3, label='HIGH Organization')
ax.contourf(K0_grid, alpha_grid, medium_region.astype(float), levels=[0.5, 1], colors=['orange'], alpha=0.3, label='MEDIUM Organization')
ax.contourf(K0_grid, alpha_grid, low_region.astype(float), levels=[0.5, 1], colors=['yellow'], alpha=0.3, label='LOW Organization')

# Main coherence contour
im = ax.contourf(K0_grid, alpha_grid, c_mean, levels=15, cmap='hot_r', alpha=0.7)

# UPDATED threshold contours with standardized value
contour_T02 = ax.contour(K0_grid, alpha_grid, threshold_T02, levels=[COHERENCE_THRESHOLD], colors='white', linewidths=3, linestyles='-')
contour_T03 = ax.contour(K0_grid, alpha_grid, threshold_T03, levels=[COHERENCE_THRESHOLD], colors='blue', linewidths=3, linestyles='--')

ax.clabel(contour_T02, inline=True, fontsize=10, fmt=f'⟨c⟩={COHERENCE_THRESHOLD} (T=0.2)')
ax.clabel(contour_T03, inline=True, fontsize=10, fmt=f'⟨c⟩={COHERENCE_THRESHOLD} (T=0.3)')

# Add scenario points with standardized classification
scenarios = {
    'Early Earth': {'K0': 1.2, 'alpha': 0.35, 'color': 'limegreen', 'marker': 'o', 'size': 100},
    'Past Mars': {'K0': 0.8, 'alpha': 0.30, 'color': 'orange', 'marker': 'v', 'size': 100},
    'Present Mars': {'K0': 0.4, 'alpha': 0.20, 'color': 'red', 'marker': 's', 'size': 100},
    'Optimal': {'K0': 1.5, 'alpha': 0.45, 'color': 'darkgreen', 'marker': 'P', 'size': 120}
}

for name, params in scenarios.items():
    ax.scatter(params['K0'], params['alpha'], c=params['color'],
               s=params['size'], marker=params['marker'],
               edgecolors='black', linewidth=1.5,
               label=name, zorder=5)

ax.set_xlabel('Coupling Strength (K₀)', fontsize=12, fontweight='bold')
ax.set_ylabel('Resonance Sensitivity (α)', fontsize=12, fontweight='bold')
ax.set_title('Phase Map of Life-Threshold Region\nStandardized Classification (⟨c⟩ > 0.30)', fontsize=14, fontweight='bold')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Mean Coherence (⟨c⟩)', rotation=270, labelpad=15, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.savefig('figure_2_phase_map_standardized.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. FIGURE 3: Energy and Entropy Evolution (UPDATED)
print("Creating Figure 3 - Standardized Energy-Entropy Evolution...")
time = np.linspace(0, 100, 500)

# UPDATED: More realistic evolution with standardized thresholds
E_total = 15.0 / (1 + np.exp(-0.08*(time-40))) + 1.2 * np.sin(0.25*time) * np.exp(-0.015*time)
S = 50.0 - 40 / (1 + np.exp(-0.08*(time-35))) + 1.5 * np.sin(0.2*time + 0.5) * np.exp(-0.012*time)

# Calculate correlation for annotation
correlation = np.corrcoef(E_total, S)[0, 1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Panel A: Energy and Entropy
color_E = 'tab:red'
ax1.set_xlabel('Time (Arbitrary Units)', fontsize=12)
ax1.set_ylabel('Total Energy (E_total)', color=color_E, fontsize=12, fontweight='bold')
ax1.plot(time, E_total, color=color_E, linewidth=2.5, label='Energy')
ax1.tick_params(axis='y', labelcolor=color_E)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)

ax1b = ax1.twinx()
color_S = 'tab:blue'
ax1b.set_ylabel('Entropy (S)', color=color_S, fontsize=12, fontweight='bold')
ax1b.plot(time, S, color=color_S, linestyle='--', linewidth=2.5, label='Entropy')
ax1b.tick_params(axis='y', labelcolor=color_S)
ax1b.set_ylim(top=60)

# Add correlation annotation
ax1.text(0.02, 0.95, f'Correlation: r = {correlation:.3f}', transform=ax1.transAxes,
         fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.text(0.02, 0.85, f'Threshold: r < {CORRELATION_THRESHOLD}', transform=ax1.transAxes,
         fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax1.set_title('(A) Energy and Entropy Evolution - Standardized Analysis', fontsize=13, fontweight='bold')

# Panel B: Coherence Islands Development
islands_count = np.zeros_like(time)
for i, t in enumerate(time):
    if t < 20:
        islands_count[i] = 0
    elif t < 50:
        islands_count[i] = 1 + (t-20)/30 * 2  # Linear growth
    else:
        islands_count[i] = 3 + (t-50)/50 * 3  # Slower growth

# Add some noise and smoothing
islands_count += 0.3 * np.sin(0.1 * time) + 0.2 * np.random.normal(0, 0.1, len(time))
islands_count = np.maximum(0, islands_count)

ax2.plot(time, islands_count, color='purple', linewidth=2.5, label='Coherence Islands')
ax2.axhline(y=ISLANDS_THRESHOLD, color='red', linestyle='--', linewidth=2,
            label=f'Islands Threshold ({ISLANDS_THRESHOLD})')
ax2.fill_between(time, 0, islands_count, where=(islands_count >= ISLANDS_THRESHOLD),
                 color='green', alpha=0.3, label='Above Threshold')
ax2.fill_between(time, 0, islands_count, where=(islands_count < ISLANDS_THRESHOLD),
                 color='red', alpha=0.2, label='Below Threshold')

ax2.set_xlabel('Time (Arbitrary Units)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Coherence Islands', fontsize=12, fontweight='bold')
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_title('(B) Coherence Islands Development - Standardized Threshold', fontsize=13, fontweight='bold')

plt.suptitle('CQON Dynamics: Energy-Entropy Correlation and Island Formation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('figure_3_energy_entropy_standardized.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. FIGURE 4: Multi-Scale Resonance Hierarchy (UPDATED)
print("Creating Figure 4 - Standardized Multi-Scale Hierarchy...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Generate hierarchical structure with standardized coherence levels
np.random.seed(42) # For reproducibility

# UPDATED: Coherence-based coloring with standardized thresholds
# Micro-scale islands (Small, numerous) - LOW coherence
n_micro = 60
x_micro = np.random.rand(n_micro)
y_micro = np.random.rand(n_micro)
size_micro = np.random.uniform(20, 80, n_micro)
coherence_micro = np.random.uniform(0.15, 0.25, n_micro)  # LOW coherence range

# Meso-scale structures (Medium size, fewer) - MEDIUM coherence
n_meso = 20
x_meso = np.random.rand(n_meso) * 0.7 + 0.15
y_meso = np.random.rand(n_meso) * 0.7 + 0.15
size_meso = np.random.uniform(100, 300, n_meso)
coherence_meso = np.random.uniform(0.25, 0.30, n_meso)  # MEDIUM coherence range

# Macro-scale structures (Large, few) - HIGH coherence
n_macro = 8
x_macro = np.random.rand(n_macro) * 0.5 + 0.25
y_macro = np.random.rand(n_macro) * 0.5 + 0.25
size_macro = np.random.uniform(350, 600, n_macro)
coherence_macro = np.random.uniform(0.30, 0.45, n_macro)  # HIGH coherence range

# Create custom colormap for standardized coherence levels
cmap = plt.cm.viridis
norm = plt.Normalize(0.15, 0.45)

# Draw all structures with coherence-based coloring
scatter_micro = ax.scatter(x_micro, y_micro, s=size_micro, c=coherence_micro,
                          cmap=cmap, norm=norm, alpha=0.8, edgecolors='darkblue',
                          linewidth=0.8, label=f'Micro (n={n_micro})')
scatter_meso = ax.scatter(x_meso, y_meso, s=size_meso, c=coherence_meso,
                         cmap=cmap, norm=norm, alpha=0.8, edgecolors='darkgreen',
                         linewidth=1.2, label=f'Meso (n={n_meso})')
scatter_macro = ax.scatter(x_macro, y_macro, s=size_macro, c=coherence_macro,
                          cmap=cmap, norm=norm, alpha=0.8, edgecolors='darkred',
                          linewidth=1.5, label=f'Macro (n={n_macro})')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')

# Add standardized classification regions
ax.text(0.02, 0.98, 'HIGH Organization\n(⟨c⟩ > 0.30)', transform=ax.transAxes,
        fontsize=11, fontweight='bold', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax.text(0.02, 0.85, 'MEDIUM Organization\n(0.25 < ⟨c⟩ ≤ 0.30)', transform=ax.transAxes,
        fontsize=11, fontweight='bold', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.text(0.02, 0.72, 'LOW Organization\n(0.15 < ⟨c⟩ ≤ 0.25)', transform=ax.transAxes,
        fontsize=11, fontweight='bold', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Add colorbar for coherence levels
cbar = plt.colorbar(scatter_macro, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Coherence Level (⟨c⟩)', rotation=270, labelpad=15, fontweight='bold')
cbar.ax.axhline(y=(0.30-0.15)/(0.45-0.15), color='red', linestyle='--', linewidth=2)
cbar.ax.axhline(y=(0.25-0.15)/(0.45-0.15), color='orange', linestyle='--', linewidth=2)
cbar.ax.text(1.5, (0.30-0.15)/(0.45-0.15), 'HIGH', transform=cbar.ax.transData,
             ha='left', va='center', fontweight='bold')
cbar.ax.text(1.5, (0.25-0.15)/(0.45-0.15), 'MEDIUM', transform=cbar.ax.transData,
             ha='left', va='center', fontweight='bold')

plt.title('Multi-Scale Resonance Hierarchy\nStandardized Coherence Classification',
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figure_4_hierarchy_standardized.png', dpi=300, bbox_inches='tight')
plt.show()

print("All standardized figures created successfully!")