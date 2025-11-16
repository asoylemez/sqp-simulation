import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from scipy import ndimage
import matplotlib.gridspec as gridspec
from statsmodels.stats.power import TTestIndPower


class CQONMarsEarthComparator:
    def __init__(self, grid_size=12, dt=0.1, K0=1.0):
        self.grid_size = grid_size
        self.dt = dt
        self.K0 = K0
        self.N = grid_size * grid_size

        # Güncellenmiş senaryo tanımları (Present Earth çıkarıldı)
        self.scenarios = {
            'Early_Earth': {'alpha': 0.35, 'gamma': 0.07, 'T': 0.15, 'color': '#2ca02c'},
            'Past_Mars': {'alpha': 0.30, 'gamma': 0.09, 'T': 0.18, 'color': '#ff7f0e'},
            'Mars_Microhabitat': {'alpha': 0.28, 'gamma': 0.08, 'T': 0.16, 'color': '#d62728'},
            'Present_Mars': {'alpha': 0.20, 'gamma': 0.12, 'T': 0.28, 'color': '#9467bd'}
        }

    def simulate_scenario(self, alpha, gamma, T, total_time=150, scenario_name="Scenario"):
        """Bir senaryo için CQON simülasyonu çalıştırır"""
        theta = 2 * np.pi * np.random.random((self.grid_size, self.grid_size))
        c = 0.05 + 0.1 * np.random.random((self.grid_size, self.grid_size))

        time_series = {
            'mean_coherence': [],
            'energy_total': [],
            'entropy': [],
            'coherence_islands': [],
            'time': [],
            'coherence_growth': []
        }

        initial_coherence = np.mean(c)

        for t in range(int(total_time / self.dt)):
            # Lokal rezonans alanı
            R_local = self.calculate_resonance_field(theta, c)

            # Uyumluluk evrimi
            noise_term = np.sqrt(2 * T * self.dt) * np.random.normal(0, 1, (self.grid_size, self.grid_size))
            dc_dt = alpha * R_local - gamma * c + noise_term
            c += dc_dt * self.dt
            c = np.clip(c, 0, 1)

            # Faz evrimi
            phase_coupling = R_local * (1 - c) * alpha
            dtheta_dt = phase_coupling + 0.1 * np.random.normal(0, 1, (self.grid_size, self.grid_size))
            theta += dtheta_dt * self.dt
            theta = theta % (2 * np.pi)

            # Metrikleri hesapla
            current_mean_coherence = np.mean(c)
            total_energy, entropy = self.calculate_thermodynamics(c, R_local)
            coherence_islands = self.count_coherence_islands(c, threshold=0.6)
            coherence_growth = (current_mean_coherence - initial_coherence) / initial_coherence * 100

            if t % 15 == 0:
                time_series['mean_coherence'].append(current_mean_coherence)
                time_series['energy_total'].append(total_energy)
                time_series['entropy'].append(entropy)
                time_series['coherence_islands'].append(coherence_islands)
                time_series['time'].append(t * self.dt)
                time_series['coherence_growth'].append(coherence_growth)

        results = self.compile_results(c, theta, time_series, alpha, gamma, T, scenario_name)
        return results

    def calculate_resonance_field(self, theta, c):
        """Rezonans alanı hesaplama"""
        R_local = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighbor_cos = 0
                neighbor_count = 0

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = (i + di) % self.grid_size, (j + dj) % self.grid_size
                        weight = 0.5 * (c[i, j] + c[ni, nj])
                        neighbor_cos += weight * np.cos(theta[ni, nj] - theta[i, j])
                        neighbor_count += 1

                if neighbor_count > 0:
                    R_local[i, j] = max(neighbor_cos / neighbor_count, 0)

        return R_local

    def calculate_thermodynamics(self, c, R_local):
        """Enerji ve entropi hesaplama"""
        energy_density = self.K0 * c * R_local
        total_energy = np.sum(energy_density)

        coherence_flat = c.flatten()
        coherence_flat = coherence_flat[coherence_flat > 0.01]
        if len(coherence_flat) > 1:
            coherence_flat = coherence_flat / np.sum(coherence_flat)
            entropy = -np.sum(coherence_flat * np.log(coherence_flat + 1e-10))
        else:
            entropy = 0

        return total_energy, entropy

    def count_coherence_islands(self, c_map, threshold=0.6):
        """Uyum adalarını say"""
        binary_map = (c_map > threshold).astype(int)
        labeled_array, num_features = ndimage.label(binary_map)
        return num_features

    def compile_results(self, c, theta, time_series, alpha, gamma, T, scenario_name):
        """Simülasyon sonuçlarını derle"""
        final_mean_coherence = np.mean(c)
        max_coherence = np.max(c)

        initial_c = time_series['mean_coherence'][0] if time_series['mean_coherence'] else 0.05
        coherence_growth = ((final_mean_coherence - initial_c) / initial_c * 100) if initial_c > 0 else 0

        if len(time_series['energy_total']) > 2:
            energy_entropy_corr = np.corrcoef(time_series['energy_total'], time_series['entropy'])[0, 1]
        else:
            energy_entropy_corr = 0

        if len(time_series['mean_coherence']) >= 10:
            stability = np.std(time_series['mean_coherence'][-10:])
        else:
            stability = np.std(time_series['mean_coherence'])

        results = {
            'final_mean_coherence': final_mean_coherence,
            'max_coherence': max_coherence,
            'final_energy': time_series['energy_total'][-1] if time_series['energy_total'] else 0,
            'final_entropy': time_series['entropy'][-1] if time_series['entropy'] else 0,
            'max_coherence_islands': np.max(time_series['coherence_islands']),
            'coherence_growth_percent': coherence_growth,
            'energy_entropy_correlation': energy_entropy_corr,
            'stability': stability,
            'final_coherence_map': c,
            'time_series': time_series,
            'parameters': {'alpha': alpha, 'gamma': gamma, 'T': T, 'K0': self.K0},
            'scenario_name': scenario_name
        }

        return results

    def run_comparative_analysis(self, n_runs=15):
        """Tüm senaryoları karşılaştırmalı analiz eder"""
        all_results = {}

        for scenario_name, params in self.scenarios.items():
            print(f"⏳ {scenario_name} senaryosu çalıştırılıyor...")
            scenario_results = []

            for run in range(n_runs):
                results = self.simulate_scenario(
                    params['alpha'], params['gamma'], params['T'],
                    total_time=150, scenario_name=f"{scenario_name}_Run_{run + 1}"
                )
                scenario_results.append(results)

            all_results[scenario_name] = scenario_results
            print(f"✅ {scenario_name} tamamlandı")

        return all_results


def perform_statistical_analysis(all_results):
    """Comprehensive statistical analysis"""
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 60)

    # Using Early Earth as reference for t-tests
    reference_scenario = 'Early_Earth'
    reference_coherences = [r['final_mean_coherence'] for r in all_results[reference_scenario]]

    statistical_results = []

    for scenario_name, results_list in all_results.items():
        if scenario_name == reference_scenario:
            continue

        test_coherences = [r['final_mean_coherence'] for r in results_list]

        # T-test
        t_stat, p_value = stats.ttest_ind(reference_coherences, test_coherences)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(reference_coherences) ** 2 + np.std(test_coherences) ** 2) / 2)
        cohens_d = (np.mean(reference_coherences) - np.mean(test_coherences)) / pooled_std

        # Power analysis
        analysis = TTestIndPower()
        n_obs = len(reference_coherences)
        power = analysis.power(effect_size=abs(cohens_d), nobs1=n_obs, alpha=0.05)

        statistical_results.append({
            'Comparison': f'{reference_scenario} vs {scenario_name}',
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'statistical_power': power,
            'significant': p_value < 0.05
        })

        print(f"\n{reference_scenario} vs {scenario_name}:")
        print(f"  t({len(reference_coherences) + len(test_coherences) - 2}) = {t_stat:.3f}, p = {p_value:.4f}")
        print(f"  Cohen's d = {cohens_d:.3f} ({interpret_effect_size(abs(cohens_d))})")
        print(f"  Statistical Power = {power:.3f}")
        print(f"  Significant Difference: {'YES' if p_value < 0.05 else 'NO'}")

    return pd.DataFrame(statistical_results)


def interpret_effect_size(d):
    """Cohen's d efekt büyüklüğünü yorumla"""
    if d < 0.2:
        return "Çok küçük efekt"
    elif d < 0.5:
        return "Küçük efekt"
    elif d < 0.8:
        return "Orta efekt"
    else:
        return "Büyük efekt"


def create_comprehensive_visualization(all_results, stats_df):
    """Create comprehensive visualization in three separate figures"""

    colors = {'Early_Earth': '#2ca02c', 'Past_Mars': '#ff7f0e',
              'Mars_Microhabitat': '#d62728', 'Present_Mars': '#9467bd'}

    scenarios = list(all_results.keys())
    labels = [scenario.replace('_', '\n') for scenario in scenarios]

    # FIGURE 1: Main Performance Metrics
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1A. Coherence Distribution Box Plot
    coherence_data = []
    for scenario in scenarios:
        coherences = [r['final_mean_coherence'] for r in all_results[scenario]]
        coherence_data.append(coherences)

    box_plot = ax1.boxplot(coherence_data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(box_plot['boxes'], [colors[s] for s in scenarios]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(y=0.35, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Life Threshold (0.35)')
    ax1.set_ylabel('Mean Coherence <c>', fontsize=12, fontweight='bold')
    ax1.set_title('A) Final Coherence Distribution Across Planetary Scenarios', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1B. Coherence Time Series
    for scenario_name, results_list in all_results.items():
        time_series = results_list[0]['time_series']
        ax2.plot(time_series['time'], time_series['mean_coherence'],
                 label=scenario_name.replace('_', ' '),
                 color=colors[scenario_name], linewidth=2.5)

    ax2.axhline(y=0.35, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time (Arbitrary Units)')
    ax2.set_ylabel('Mean Coherence <c>')
    ax2.set_title('B) Coherence Evolution Time Series', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 1C. Coherence Island Formation
    island_data = []
    for scenario in scenarios:
        islands = [r['max_coherence_islands'] for r in all_results[scenario]]
        island_data.append(islands)

    bar_plot = ax3.bar(labels, [np.mean(data) for data in island_data],
                       yerr=[np.std(data) for data in island_data],
                       color=[colors[s] for s in scenarios], alpha=0.7,
                       capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})

    ax3.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Min Island Threshold')
    ax3.set_ylabel('Number of Coherence Islands')
    ax3.set_title('C) Coherence Island Formation', fontsize=12, fontweight='bold')
    ax3.legend()

    # 1D. Energy-Entropy Correlation
    correlation_data = []
    for scenario in scenarios:
        corrs = [r['energy_entropy_correlation'] for r in all_results[scenario]]
        correlation_data.append(corrs)

    violin_parts = ax4.violinplot(correlation_data, showmeans=True, showmedians=True)
    for pc, color in zip(violin_parts['bodies'], [colors[s] for s in scenarios]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax4.axhline(y=-0.4, color='red', linestyle='--', alpha=0.7, label='Correlation Threshold')
    ax4.set_xticks(range(1, len(labels) + 1))
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('Energy-Entropy Correlation (r)')
    ax4.set_title('D) Energy-Information Transformation', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('CQON_Performance_Metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # FIGURE 2: Statistical Analysis
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 2A. Statistical Significance
    if stats_df is not None and not stats_df.empty:
        comparisons = stats_df['Comparison'].tolist()
        p_values = stats_df['p_value'].tolist()
        cohens_d = stats_df['cohens_d'].tolist()

        x_pos = np.arange(len(comparisons))
        bars1 = ax1.bar(x_pos - 0.2, p_values, 0.4, label='p-value', alpha=0.7, color='skyblue')
        bars2 = ax1.bar(x_pos + 0.2, np.abs(cohens_d), 0.4, label="|Cohen's d|", alpha=0.7, color='lightcoral')

        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significance Threshold (p=0.05)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([comp.replace('Early_Earth vs ', '') for comp in comparisons], rotation=45)
        ax1.set_ylabel('Values')
        ax1.set_title('A) Statistical Comparison: Early Earth vs Other Scenarios', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (p_val, d_val) in enumerate(zip(p_values, cohens_d)):
            ax1.text(i - 0.2, p_val + 0.01, f'{p_val:.3f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + 0.2, abs(d_val) + 0.01, f'{abs(d_val):.2f}', ha='center', va='bottom', fontsize=9)

    # 2B. Performance Score Comparison
    performance_scores = []
    for scenario_name in scenarios:
        mean_coh = np.mean([r['final_mean_coherence'] for r in all_results[scenario_name]])
        islands = np.mean([r['max_coherence_islands'] for r in all_results[scenario_name]])
        mean_corr = np.mean([r['energy_entropy_correlation'] for r in all_results[scenario_name]])
        score = calculate_life_like_score(mean_coh, islands, mean_corr)
        performance_scores.append(score)

    bars = ax2.bar(labels, performance_scores, color=[colors[s] for s in scenarios], alpha=0.7)
    ax2.set_ylabel('Life-Like Organization Score')
    ax2.set_title('B) Life-Like Organization Scoring', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 5)

    # Add value labels on bars
    for bar, score in zip(bars, performance_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{score}/4', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('CQON_Statistical_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # FIGURE 3: Coherence Maps
    fig3, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, scenario_name in enumerate(scenarios):
        coherence_map = all_results[scenario_name][0]['final_coherence_map']
        im = axs[i].imshow(coherence_map, cmap='viridis', vmin=0, vmax=1, aspect='equal')

        # Performance metrics for title
        mean_coh = np.mean([r['final_mean_coherence'] for r in all_results[scenario_name]])
        islands = np.mean([r['max_coherence_islands'] for r in all_results[scenario_name]])
        score = calculate_life_like_score(mean_coh, islands, -0.4)
        status = "HIGH" if score >= 3 else "MEDIUM" if score >= 2 else "LOW"

        axs[i].set_title(f'{scenario_name.replace("_", " ")}\nScore: {status} (Coherence: {mean_coh:.3f})',
                         fontweight='bold', fontsize=11)

        # Add colorbar for each subplot
        plt.colorbar(im, ax=axs[i], shrink=0.8)

    plt.tight_layout()
    plt.savefig('CQON_Coherence_Maps.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_life_like_score(coh, islands, corr):
    """Life-like organization scoring"""
    score = 0
    if coh > 0.35: score += 1
    if islands >= 2: score += 1
    if corr < -0.4: score += 1
    if coh > 0.4 and islands >= 3: score += 1
    return score


def calculate_life_like_score(coh, islands, corr):
    """Yaşam-benzeri organizasyon skoru"""
    score = 0
    if coh > 0.35: score += 1
    if islands >= 2: score += 1
    if corr < -0.4: score += 1
    if coh > 0.4 and islands >= 3: score += 1
    return score


def main():
    print("CQON Mars-Dünya Karşılaştırmalı Analizi Başlatılıyor...")
    print("=" * 60)

    comparator = CQONMarsEarthComparator(grid_size=12, dt=0.1, K0=1.0)
    all_results = comparator.run_comparative_analysis(n_runs=15)

    # İstatistiksel analiz
    stats_df = perform_statistical_analysis(all_results)

    # Sonuç tablosu
    print("\n" + "=" * 80)
    print("KARŞILAŞTIRMALI PERFORMANS TABLOSU")
    print("=" * 80)

    summary_data = []
    for scenario_name, results_list in all_results.items():
        coherences = [r['final_mean_coherence'] for r in results_list]
        growths = [r['coherence_growth_percent'] for r in results_list]
        islands = [r['max_coherence_islands'] for r in results_list]
        correlations = [r['energy_entropy_correlation'] for r in results_list]

        mean_coh = np.mean(coherences)
        mean_islands = np.mean(islands)
        mean_corr = np.mean(correlations)

        summary_data.append({
            'Senaryo': scenario_name.replace('_', ' '),
            'Parametreler (α,γ,T)': f"{results_list[0]['parameters']['alpha']}, {results_list[0]['parameters']['gamma']}, {results_list[0]['parameters']['T']}",
            'Ort. Uyum': f"{mean_coh:.3f} ± {np.std(coherences):.3f}",
            'Uyum Büyümesi': f"{np.mean(growths):.1f}%",
            'Ort. Ada': f"{mean_islands:.1f} ± {np.std(islands):.1f}",
            'E-S Korelasyon': f"{mean_corr:.3f}",
            'Yaşam-Benzeri': 'EVET' if calculate_life_like_score(mean_coh, mean_islands,
                                                                 mean_corr) >= 3 else 'KISMEN' if calculate_life_like_score(
                mean_coh, mean_islands, mean_corr) >= 2 else 'HAYIR'
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Görselleştirme
    create_comprehensive_visualization(all_results, stats_df)

    return all_results, stats_df, summary_df


if __name__ == "__main__":
    all_results, stats_df, summary_df = main()