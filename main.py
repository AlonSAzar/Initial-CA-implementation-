from engine import ElementaryCA
from complexity import ZlibComplexity, MutualInfoComplexity
from experiments import SimplicityBiasExperiment, RobustnessExperiment, PopulationGrowthExperiment
import matplotlib.pyplot as plt

# TODO when we create graphs, we need the params to be written
def main():
    # 1. Setup Configuration
    # TODO with bigger L, it seems further from the limit, weird
    L = 32
    T = 64
    NUM_SEEDS = 100

    # 2. Instantiate Objects
    engine = ElementaryCA(L, T)
    metric = ZlibComplexity()  # Can easily swap this with MutualInfoComplexity()

    # --- Experiment A.2: Population Growth Simplicity Bias ---
    print("--- Starting Population Growth Experiment ---")
    pop_exp = PopulationGrowthExperiment(engine, metric)
    pop_exp.run_batch(num_seeds=NUM_SEEDS)
    pop_exp.analyze_and_plot()

    # --- Experiment A: Simplicity Bias ---
    """ 
    We generate NUM_SEEDS (default 100) different seeds per the 256 different rules. 
    """
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
    # sb_exp.plot_results(freqs_shuff, comps_shuff)

    # --- Experiment B: Robustness ---
    print("--- Starting Robustness Experiments ---")
    rob_exp = RobustnessExperiment(engine, metric)
    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_seed_mut(num_seeds=20)

    # Plot Robustness vs Complexity
    plt.scatter(list(phenotype_complexities.values()), list(robustness_scores.values()))
    plt.xlabel("Complexity")
    plt.ylabel("Robustness")
    plt.title("Evolutionary Trade-off, mutating seeds")
    plt.show()




    robustness_scores, phenotype_complexities = rob_exp.compute_ncc_robustness_rule_mut(num_seeds=20)

    # Get average complexity per rule for the final plot
    # avg_complexity = {r: rob_exp.compute_rule_complexity(r) for r in range(256)}

    # Plot Robustness vs Complexity
    plt.scatter(list(phenotype_complexities.values()), list(robustness_scores.values()))
    plt.xlabel("Complexity")
    plt.ylabel("Robustness")
    plt.title("Evolutionary Trade-off")
    plt.show()


if __name__ == "__main__":
    main()