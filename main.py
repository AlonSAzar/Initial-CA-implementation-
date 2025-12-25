from engine import ElementaryCA
from complexity import ZlibComplexity, MutualInfoComplexity, PatchComplexity2D
from experiments import *
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

    # # --- Experiment A: Simplicity Bias ---
    # """ 
    # We generate NUM_SEEDS (default 100) different seeds per the 256 different rules. 
    # """
    # print("--- Starting Simplicity Bias Experiment ---")
    # sb_exp = SimplicityBiasExperiment(engine, metric)
    # sb_exp.run_batch(num_seeds=NUM_SEEDS)
    # 
    # # Analyze original
    # freqs, comps = sb_exp.analyze(shuffle_control=False)
    # sb_exp.plot_results(freqs, comps)
    # 
    # # Analyze control (shuffled)
    # freqs_shuff, comps_shuff = sb_exp.analyze(shuffle_control=True)
    # sb_exp.plot_results(freqs_shuff, comps_shuff, title_add="Shuffled Control")
    # 
    # # --- Experiment B: Population Growth Simplicity Bias ---
    # print("--- Starting Population Growth Experiment ---")
    # pop_exp = PopulationGrowthExperiment(engine, metric)
    # pop_exp.run_batch(num_seeds=NUM_SEEDS)
    # pop_exp.analyze_and_plot()
    
    
    # --- Experiment: Conditional Complexity ---
    cond_trans_exp = ConditionalTransitionExperiment(engine, metric)
    freq_map, complexity_map = cond_trans_exp.run(100)
    cond_trans_exp.plot_results(freq_map, complexity_map)
    

    # --- Experiment C: Robustness ---
    print("--- Starting Robustness Experiments ---")
    rob_exp = RobustnessExperiment(engine, metric)

    # TODO if I want I can do a function that has a toggle to either execute just this or both

    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_rule_or_seed_mut(num_seeds=20, mut_type="random_seed")
    rob_exp.analyze_and_plot(robustness_scores, phenotype_complexities, mutation_type="Mutating Rules, Random Seed")

    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_rule_or_seed_mut(num_seeds=20, mut_type="random_everything")
    rob_exp.analyze_and_plot(robustness_scores, phenotype_complexities, mutation_type="Mutating Rules, Random Seed & Rule")

    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_rule_or_seed_mut(num_seeds=20, mut_type="random_rule")
    rob_exp.analyze_and_plot(robustness_scores, phenotype_complexities, mutation_type="Mutating Rules, Random Rule")
    
    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_rule_or_seed_mut(num_seeds=20, variable="seed")
    rob_exp.analyze_and_plot(robustness_scores, phenotype_complexities, mutation_type="Mutating Seeds")

    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_rule_or_seed_mut(num_seeds=20, variable="seed", shuffle_control=True)
    rob_exp.analyze_and_plot(robustness_scores, phenotype_complexities, mutation_type="Mutating Seeds, Shuffled Control")

    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_rule_or_seed_mut(num_seeds=20, variable="rule")
    rob_exp.analyze_and_plot(robustness_scores, phenotype_complexities, mutation_type="Mutating Rules")

    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_rule_or_seed_mut(num_seeds=20, variable="rule", shuffle_control=True)
    rob_exp.analyze_and_plot(robustness_scores, phenotype_complexities, mutation_type="Mutating Rules, Shuffled Control")

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