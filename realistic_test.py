"""
CQON Model - Gerçek Koşullarda Test - UPDATED
Standardized thresholds and classification system
Doğal sistemlere uygun parametrelerle test
"""

import numpy as np
import matplotlib.pyplot as plt
from cqon_model import CQONSimulation

# STANDARDIZED THRESHOLDS
COHERENCE_THRESHOLD = 0.30
ISLANDS_THRESHOLD = 3
CORRELATION_THRESHOLD = -0.45

# TÜRKÇE KARAKTER DÜZELTMESİ
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def realistic_parameter_sweep():
    """Gerçekçi parametre taraması - UPDATED with standardized thresholds"""
    print("🔬 GERÇEK KOŞULLARDA CQON TESTİ - STANDARDIZED")
    print("=" * 60)
    print(f"🎯 Standard Eşikler: ⟨c⟩>{COHERENCE_THRESHOLD}, adalar≥{ISLANDS_THRESHOLD}, r<{CORRELATION_THRESHOLD}")

    # Gerçekçi senaryolar - standardized thresholds ile güncellendi
    scenarios = [
        {
            "name": "OPTIMUM KUANTUM ORTAM",
            "desc": "Dusuk sicaklik, yuksek koherans - laboratuvar kosullari",
            "alpha": 0.45, "gamma": 0.05, "T": 0.08, "K0": 1.1,
            "grid_size": 12, "total_time": 80, "dt": 0.2,
            "expected_class": "HIGH"
        },
        {
            "name": "ERKEN DUNYA BENZERI",
            "desc": "Orta seviye gurultu - prebiyotik Dunya kosullari",
            "alpha": 0.35, "gamma": 0.07, "T": 0.15, "K0": 0.9,
            "grid_size": 12, "total_time": 100, "dt": 0.2,
            "expected_class": "HIGH"
        },
        {
            "name": "OKYANUS DIPI KOSULLARI",
            "desc": "Yuksek basinc/kararlilik - hidrotermal bacalar",
            "alpha": 0.38, "gamma": 0.04, "T": 0.12, "K0": 1.0,
            "grid_size": 12, "total_time": 120, "dt": 0.2,
            "expected_class": "HIGH"
        },
        {
            "name": "GECMIS MARS BENZERI",
            "desc": "Orta duzey enerji - eski Mars kosullari",
            "alpha": 0.30, "gamma": 0.09, "T": 0.18, "K0": 0.8,
            "grid_size": 12, "total_time": 100, "dt": 0.2,
            "expected_class": "MEDIUM"
        },
        {
            "name": "GUNUMUZ MARS BENZERI",
            "desc": "Dusuk enerji, yuksek dekoherans - modern Mars",
            "alpha": 0.20, "gamma": 0.12, "T": 0.28, "K0": 0.7,
            "grid_size": 12, "total_time": 150, "dt": 0.2,
            "expected_class": "LOW"
        }
    ]

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. 📋 {scenario['name']}")
        print(f"   📝 {scenario['desc']}")
        print(f"   ⚙️  Parametreler: α={scenario['alpha']}, γ={scenario['gamma']}, "
              f"T={scenario['T']}, K₀={scenario['K0']}")
        print(f"   🎯 Beklenen Sınıf: {scenario['expected_class']}")

        try:
            sim = CQONSimulation(
                alpha=scenario['alpha'],
                gamma=scenario['gamma'],
                T=scenario['T'],
                K0=scenario['K0'],
                grid_size=scenario['grid_size'],
                total_time=scenario['total_time'],
                dt=scenario['dt']
            )

            # Detaylı sonuçlar için
            results_dict = sim.run(verbose=False)
            results_dict['scenario'] = scenario['name']
            results_dict['expected_class'] = scenario['expected_class']
            results.append(results_dict)

            # Detaylı analiz - UPDATED with standardized classification
            print(f"   📊 SONUCLAR:")
            print(f"      • Ortalama koherans: {results_dict['avg_coherence']:.3f} "
                  f"{'✅' if results_dict['avg_coherence'] > COHERENCE_THRESHOLD else '❌'}")
            print(f"      • Koherans adalari: {results_dict['coherence_islands']} "
                  f"{'✅' if results_dict['coherence_islands'] >= ISLANDS_THRESHOLD else '❌'}")
            print(f"      • Enerji: {results_dict['final_energy']:.1f}")
            print(f"      • Entropi: {results_dict['final_entropy']:.1f}")
            print(f"      • E-S Korelasyon: {results_dict['energy_entropy_correlation']:.3f} "
                  f"{'✅' if results_dict['energy_entropy_correlation'] < CORRELATION_THRESHOLD else '❌'}")
            print(f"      • Sınıflandırma: {results_dict['organization_classification']} "
                  f"{'✅' if results_dict['organization_classification'] == scenario['expected_class'] else '⚠️'}")
            print(f"      • Yaşam-benzeri: {'EVET' if results_dict['life_like_organization'] else 'HAYIR'}")

            # Gelişmiş yaşam analizi - UPDATED
            life_status = analyze_life_likelihood(results_dict)
            print(f"      • 🎯 YASAM OLASILIGI: {life_status}")

        except Exception as e:
            print(f"   ❌ Hata: {e}")
            continue

    return results


def analyze_life_likelihood(results):
    """Yaşam olasılığını detaylı analiz et - UPDATED with standardized thresholds"""
    score = 0
    feedback = []

    # Koherans puanı - UPDATED threshold
    if results['avg_coherence'] > 0.35:
        score += 3
        feedback.append("Yuksek koherans (>0.35) ✅")
    elif results['avg_coherence'] > COHERENCE_THRESHOLD:
        score += 2
        feedback.append(f"Yeterli koherans (>{COHERENCE_THRESHOLD}) ✅")
    else:
        feedback.append(f"Dusuk koherans (<{COHERENCE_THRESHOLD}) ❌")

    # Ada puanı - UPDATED threshold
    if results['coherence_islands'] >= 4:
        score += 3
        feedback.append("Coklu kararli adalar (≥4) ✅")
    elif results['coherence_islands'] >= ISLANDS_THRESHOLD:
        score += 2
        feedback.append(f"Yeterli ada olusumu (≥{ISLANDS_THRESHOLD}) ✅")
    else:
        feedback.append(f"Yetersiz ada olusumu (<{ISLANDS_THRESHOLD}) ❌")

    # Enerji-Entropi puanı - UPDATED threshold
    corr = results['energy_entropy_correlation']
    if corr < -0.5:
        score += 3
        feedback.append("Guclu enerji-enformasyon donusumu (<-0.5) ✅")
    elif corr < CORRELATION_THRESHOLD:
        score += 2
        feedback.append(f"Yeterli enerji-enformasyon donusumu (<{CORRELATION_THRESHOLD}) ✅")
    else:
        feedback.append(f"Zayif enerji-enformasyon donusumu (≥{CORRELATION_THRESHOLD}) ❌")

    # Karar - UPDATED scoring
    if score >= 8:
        return "YUKSEK - Guclu yasam-benzeri organizasyon 🎯"
    elif score >= 6:
        return "ORTA - Zayif yasam-benzeri organizasyon 📈"
    elif score >= 4:
        return "DUSUK - On-yasamsal organizasyon 📉"
    else:
        return "YOK - Kaotik durum ❌"


def plot_comprehensive_results(results):
    """Kapsamlı sonuç görselleştirmesi - UPDATED with standardized thresholds"""
    if not results:
        print("❌ Görselleştirme için sonuç yok!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Senaryo karşılaştırması - UPDATED with classification colors
    scenarios = [r['scenario'] for r in results]
    coherence = [r['avg_coherence'] for r in results]
    islands = [r['coherence_islands'] for r in results]
    classifications = [r['organization_classification'] for r in results]

    x_pos = np.arange(len(scenarios))

    # Classification colors
    color_map = {'HIGH': 'green', 'MEDIUM': 'orange', 'LOW': 'red', 'NO': 'gray'}
    bar_colors = [color_map[cls] for cls in classifications]

    axes[0, 0].bar(x_pos - 0.2, coherence, 0.4, label='Ortalama Koherans', alpha=0.7, color=bar_colors)
    axes[0, 0].bar(x_pos + 0.2, islands, 0.4, label='Koherans Adalari', alpha=0.7, color=bar_colors)

    # UPDATED threshold lines
    axes[0, 0].axhline(y=COHERENCE_THRESHOLD, color='red', linestyle='--',
                       label=f'Koherans Eşiği ({COHERENCE_THRESHOLD})')
    axes[0, 0].axhline(y=ISLANDS_THRESHOLD, color='blue', linestyle='--',
                       label=f'Ada Eşiği ({ISLANDS_THRESHOLD})', alpha=0.5)

    axes[0, 0].set_xlabel('Senaryolar')
    axes[0, 0].set_ylabel('Degerler')
    axes[0, 0].set_title('CQON Senaryo Karşılaştırması - Standardize Sınıflandırma')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([s[:15] + '...' for s in scenarios], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Enerji-Entropi korelasyonu - UPDATED threshold
    correlations = [r['energy_entropy_correlation'] for r in results]
    axes[0, 1].bar(range(len(scenarios)), correlations, color=bar_colors, alpha=0.7)
    axes[0, 1].axhline(y=CORRELATION_THRESHOLD, color='red', linestyle='--',
                       label=f'Kritik Eşik ({CORRELATION_THRESHOLD})')
    axes[0, 1].set_xlabel('Senaryolar')
    axes[0, 1].set_ylabel('Korelasyon Katsayisi')
    axes[0, 1].set_title('Enerji-Entropi Korelasyonu - Standardize Eşik')
    axes[0, 1].set_xticks(range(len(scenarios)))
    axes[0, 1].set_xticklabels([s[:15] + '...' for s in scenarios], rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Zaman evrimi örneği (ilk senaryo) - UPDATED
    if results:
        axes[1, 0].plot(results[0]['energy_history'], 'r-', label='Enerji', linewidth=2)
        axes[1, 0].plot(results[0]['entropy_history'], 'b-', label='Entropi', linewidth=2)
        axes[1, 0].plot(results[0]['coherence_history'], 'g-', label='Koherans', linewidth=2)
        axes[1, 0].set_xlabel('Zaman Adimlari')
        axes[1, 0].set_ylabel('Degerler')
        axes[1, 0].set_title(f"{results[0]['scenario']} - Zaman Evrimi (Standardize)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Yaşam olasılığı skorları - UPDATED with standardized scoring
    life_scores = []
    for result in results:
        score = 0
        if result['avg_coherence'] > COHERENCE_THRESHOLD: score += 1
        if result['coherence_islands'] >= ISLANDS_THRESHOLD: score += 1
        if result['energy_entropy_correlation'] < CORRELATION_THRESHOLD: score += 1
        if result['life_like_organization']: score += 1  # Bonus for meeting all criteria
        life_scores.append(score)

    colors = ['red' if s < 2 else 'orange' if s < 3 else 'green' for s in life_scores]
    bars = axes[1, 1].bar(range(len(scenarios)), life_scores, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('Senaryolar')
    axes[1, 1].set_ylabel('Yaşam Skoru (0-4)')
    axes[1, 1].set_title('CQON Yaşam-Benzeri Organizasyon Skoru - Standardize')
    axes[1, 1].set_xticks(range(len(scenarios)))
    axes[1, 1].set_xticklabels([s[:15] + '...' for s in scenarios], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, score, cls in zip(bars, life_scores, classifications):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                        f'{score}/4\n{cls}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('CQON Model - Standardize Edilmiş Senaryo Analizi', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('realistic_cqon_analysis_standardized.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_detailed_single_simulation():
    """Tek bir senaryoda detaylı analiz - UPDATED with standardized parameters"""
    print("\n" + "=" * 60)
    print("🔍 TEK SENARYO DETAYLI ANALIZ - STANDARDIZE CQON MODEL")
    print("=" * 60)
    print(f"🎯 Standard Eşikler: ⟨c⟩>{COHERENCE_THRESHOLD}, adalar≥{ISLANDS_THRESHOLD}, r<{CORRELATION_THRESHOLD}")

    # Erken Dünya benzeri koşullar - STANDARDIZE EDILMIS
    sim = CQONSimulation(
        alpha=0.35, gamma=0.07, T=0.15, K0=0.9,  # Standardize parametreler
        grid_size=14, total_time=120, dt=0.15
    )

    print("📖 Senaryo: STANDARDIZE ERKEN DUNYA")
    print("   - Optimal rezonans hassasiyeti (α=0.35)")
    print("   - Dengeli dekoherans (γ=0.07)")
    print("   - Orta seviye termal gurultu (T=0.15)")
    print("   - CQON Teorisi: Enerji → Koherans → Enformasyon")
    print("\n⏳ CQON simülasyonu calisiyor...")

    results = sim.run(verbose=True)

    # Ek analiz - UPDATED
    if results:
        print(f"\n📈 DETAYLI ANALIZ:")
        print(f"   • Baslangic koherans: {results['coherence_history'][0]:.3f}")
        print(f"   • Maksimum koherans: {max(results['coherence_history']):.3f}")
        print(f"   • Koherans artisi: {results['coherence_history'][-1] - results['coherence_history'][0]:.3f}")
        print(f"   • Enerji kazanimi: {results['energy_history'][-1] - results['energy_history'][0]:.1f}")
        print(f"   • Entropi azalimi: {results['entropy_history'][0] - results['entropy_history'][-1]:.1f}")

        # Standardize threshold analysis
        print(f"\n🎯 STANDARDIZE ESIK ANALIZI:")
        print(f"   • Koherans > {COHERENCE_THRESHOLD}: {results['avg_coherence']:.3f} → "
              f"{'✅ GEÇERLİ' if results['avg_coherence'] > COHERENCE_THRESHOLD else '❌ GEÇERSİZ'}")
        print(f"   • Adalar ≥ {ISLANDS_THRESHOLD}: {results['coherence_islands']} → "
              f"{'✅ GEÇERLİ' if results['coherence_islands'] >= ISLANDS_THRESHOLD else '❌ GEÇERSİZ'}")
        print(f"   • Korelasyon < {CORRELATION_THRESHOLD}: {results['energy_entropy_correlation']:.3f} → "
              f"{'✅ GEÇERLİ' if results['energy_entropy_correlation'] < CORRELATION_THRESHOLD else '❌ GEÇERSİZ'}")

        # Teori açıklaması
        print(f"\n📖 CQON TEORISI OZETI:")
        theory = results['theory_explanation']
        for key, value in theory.items():
            print(f"   • {key.replace('_', ' ').title()}: {value}")

    return results


if __name__ == "__main__":
    # Tüm senaryoları test et
    print("🚀 CQON Standardize Test Başlatılıyor...")
    print("🎯 Model: Coherent Quantum Oscillator Network - STANDARDIZED")
    all_results = realistic_parameter_sweep()

    # Detaylı tek senaryo analizi - STANDARDIZE ERKEN DUNYA
    detailed_results = run_detailed_single_simulation()

    # Görselleştirme
    if all_results:
        print("\n📊 Sonuclar görselleştiriliyor...")
        plot_comprehensive_results(all_results)

        print("\n✅ CQON STANDARDIZE TEST TAMAMLANDI!")
        print("📁 'realistic_cqon_analysis_standardized.png' kaydedildi")

        # İstatistiksel özet - UPDATED
        successful_simulations = len(all_results)
        life_like_count = sum(1 for r in all_results if r.get('life_like_organization', False))
        classification_counts = {}
        for r in all_results:
            cls = r.get('organization_classification', 'UNKNOWN')
            classification_counts[cls] = classification_counts.get(cls, 0) + 1

        print(f"📊 ISTATISTIK:")
        print(f"   • Toplam simulasyon: {successful_simulations}")
        print(f"   • Yaşam-benzeri organizasyon: {life_like_count}")
        print(f"   • Sınıflandırma dağılımı: {classification_counts}")

    else:
        print("\n❌ Test sonuc alinamadi!")