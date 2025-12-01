from engine import ElementaryCA
from complexity import ZlibComplexity, MutualInfoComplexity
from experiments import SimplicityBiasExperiment, RobustnessExperiment, PopulationGrowthExperiment
import matplotlib.pyplot as plt

def main():
    # 1. Setup Configuration
    # TODO with bigger L, it seems further from the limit, weird
    L = 12
    T = 64
    NUM_SEEDS = 100

    # 2. Instantiate Objects
    engine = ElementaryCA(L, T)
    metric = ZlibComplexity()  # Can easily swap this with MutualInfoComplexity()

    # --- Experiment A: Simplicity Bias ---
    """ 
    We generate NUM_SEEDS (default 100) different seeds per the 256 different rules. 
    """
    print("--- Starting Simplicity Bias Experiment ---")
    sb_exp = SimplicityBiasExperiment(engine, metric)
    sb_exp.run_batch(num_seeds=NUM_SEEDS)

    # Analyze original
    freqs, comps = sb_exp.analyze(shuffle_control=False)
    sb_exp.plot_results(freqs, comps)

    # Analyze control (shuffled)
    freqs_shuff, comps_shuff = sb_exp.analyze(shuffle_control=True)
    sb_exp.plot_results(freqs_shuff, comps_shuff, title_add="Shuffled Control")

    # --- Experiment B: Population Growth Simplicity Bias ---
    print("--- Starting Population Growth Experiment ---")
    pop_exp = PopulationGrowthExperiment(engine, metric)
    pop_exp.run_batch(num_seeds=NUM_SEEDS)
    pop_exp.analyze_and_plot()

    # --- Experiment C: Robustness ---
    print("--- Starting Robustness Experiments ---")
    rob_exp = RobustnessExperiment(engine, metric)

    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_seed_mut(num_seeds=20)
    rob_exp.analyze_and_plot(robustness_scores, phenotype_complexities, mutation_type="Mutating Seeds")


    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_rule_mut(num_seeds=20)
    rob_exp.analyze_and_plot(robustness_scores, phenotype_complexities, mutation_type="Mutating Rules")

    # Get average complexity per rule for the final plot
    # avg_complexity = {r: rob_exp.compute_rule_complexity(r) for r in range(256)}

    # # Plot Robustness vs Complexity
    # plt.scatter(list(phenotype_complexities.values()), list(robustness_scores.values()))
    # plt.xlabel("Complexity")
    # plt.ylabel("Robustness")
    # plt.title(f"1D CA Robustness VS Complexity, Mutating Rules. L={L}, T={T}")
    # plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
    #          verticalalignment='top', horizontalalignment='right', bbox=props)
    # plt.show()


if __name__ == "__main__":
    main()