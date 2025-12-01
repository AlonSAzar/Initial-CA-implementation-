from collections import Counter
from typing import Tuple, Dict, Any

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Import from other modules
from engine import ElementaryCA
from complexity import ComplexityMetric

class SimplicityBiasExperiment:
    """
    Tests the AIT hypothesis: P(x) ~ 2^-K(x).
    Generates distribution plots.
    """

    # TODO why is it elementaryCA?
    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric):
        self.engine = engine
        self.metric = metric
        self.results = {}  # rule -> list of images

    def run_batch(self, num_seeds: int, rules=range(256)):
        """Generates data."""
        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        # tqdm displays a progress bar in the console
        for rule in tqdm(rules, desc="Simulating Rules"):
            phenotypes = []
            for seed in seeds:
                # Discard t=0 (seed) usually
                img = self.engine.run(rule, seed)[1:]
                phenotypes.append(img)
            self.results[rule] = phenotypes

    # TODO fix the damn return statement
    def analyze(self, shuffle_control=False):
        """
        Hashes phenotypes and calculates complexity.
        returns: freq_map, complexity_map
            freq_map: Hash of image -> count
            complexity_map: Dict hash of image -> complexity score
            """
        # Dict that will store the count of each unique phenotype.
        # Hash of image -> count
        freq_map = Counter()
        # Dict hash of image -> complexity score
        complexity_map = {}

        # Flatten all results (which are a list of lists) into one big pool of phenotypes
        all_images = [img for sublist in self.results.values() for img in sublist]

        for img in tqdm(all_images, desc="Analyzing Complexity"):
            # Optional: Shuffle image to test control
            if shuffle_control:
                img = self.shuffle_image_rows(img)

            h = img.tobytes()  # Hash
            freq_map[h] += 1

            if h not in complexity_map:
                complexity_map[h] = self.metric.calculate(img)

        return freq_map, complexity_map

    def shuffle_image_rows(self, image: np.ndarray) -> np.ndarray:
        shuffled = image.copy()
        for row in shuffled:
            np.random.shuffle(row)
        return shuffled

    # TODO add an optional argument for additions to the title
    def plot_results(self, freq_map, complexity_map):
        """
        Visualizes the Simplicity Bias.
        Creates a graph of complexity against the log of probabilities.
        """
        Ks = []  # Complexities
        log_probs = []  # log(Probability)
        total = sum(freq_map.values())

        for h, count in freq_map.items():
            Ks.append(complexity_map[h])
            log_probs.append(np.log10(count / total))

        plt.scatter(Ks, log_probs, alpha=0.5)
        plt.xlabel(f"Complexity ({self.metric.name()})")
        plt.ylabel("Log Probability")
        plt.title("Simplicity Bias")
        plt.show()


def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Normalized Cross-Correlation (NCC) between two images.
    Images must be the same shape.
    Returns value in [-1, 1], where 1 mean perfect match.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for NCC.")

    # Convert to float32 for accurate floating
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Calculate average value of pixels
    img1_mean = img1.mean()
    img2_mean = img2.mean()

    # Scale the images, and multiply each pixel by the value of the equivalent.
    # If 2 pixels have opposite values compared to the mean, the result will be neg.
    # If they have same sign values compared to the mean, it will be positive.
    numerator = np.sum((img1 - img1_mean) * (img2 - img2_mean))

    denominator = np.sqrt(np.sum((img1 - img1_mean) ** 2) * np.sum((img2 - img2_mean) ** 2))

    if denominator == 0:
        return 0.0  # Avoid division by zero

    return numerator / denominator


def int_to_binary_array_numpy(number: int, num_bits: int = 8) -> np.ndarray:
    """
    Converts an integer into a binary array of a specified length (MSB first).
    """
    if number < 0 or number >= (1 << num_bits):
        raise ValueError(f"Number {number} out of range for {num_bits} bits.")

    # Create an array of bit positions to check (e.g., for 8 bits: [7, 6, 5, 4, 3, 2, 1, 0])
    bit_positions = np.arange(num_bits - 1, -1, -1)

    # Perform bitwise right shift (>>) and check the last bit (& 1)
    # This is broadcast across all bit positions simultaneously.
    binary_array = ((number >> bit_positions) & 1).astype(np.uint8)

    return binary_array


class RobustnessExperiment:
    """
    Tests the relationship between Complexity and Evolutionary Robustness.
    """

    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric):
        self.engine = engine
        self.metric = metric

    def compute_ncc_robustness_rule_mut(self, num_seeds=50) -> tuple[dict[int, float], dict[int, float]]:
        """
        Returns dictionaries {rule_id: robustness_score}, {rule_id, phenotype_complexity}.
        Robustness = Average NCC between Rule(seed) and MutantRule(seed).
        """
        rule_robustness_scores = {}
        rule_phenotype_complexity_scores = {}
        rule_images_dict = {}
        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        for rule in tqdm(range(256), desc="Evolutionary Robustness"):
            rule_score = 0
            rule_images_list = []

            for seed in seeds:
                base_img = self.engine.run(rule, seed)[1:]
                rule_images_list.append(base_img)

                # Check all 8 1-bit neighbors
                mutant_scores = []
                for bit in range(8):
                    mut_rule = self.engine.mutate_rule(rule, bit)
                    mut_img = self.engine.run(mut_rule, seed)[1:]

                    # Compute NCC (Normalized Cross Correlation)
                    ncc = compute_ncc(base_img, mut_img)
                    mutant_scores.append(ncc)

                rule_score += np.mean(mutant_scores)

            # Calculate average across seeds, per rule
            rule_robustness_scores[rule] = rule_score / len(seeds)
            rule_images_dict[rule] = rule_images_list

        rule_phenotype_complexity_scores = self.mean_phenotype_complexity(rule_images_dict)

        return rule_robustness_scores, rule_phenotype_complexity_scores

    # TODO make sure the logic is sound
    def compute_ncc_robustness_seed_mut(self, num_seeds=50) -> tuple[dict[int, float], dict[int, float]]:
        """
        Returns dictionaries {rule_id: robustness_score}, {rule_id, phenotype_complexity}.
        Robustness = Average NCC between Rule(seed) and Rule(mutant_seed).
        """
        rule_robustness_scores = {}
        rule_phenotype_complexity_scores = {}
        rule_images_dict = {}
        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        for rule in tqdm(range(256), desc="Evolutionary Robustness"):
            rule_score = 0
            rule_images_list = []

            for seed in seeds:
                base_img = self.engine.run(rule, seed)[1:]
                rule_images_list.append(base_img)

                # Check all 1-bit neighbors of the seed
                mutant_scores = []
                for bit in range(len(seed)):
                    mut_seed = self.engine.mutate_seed(seed, bit)
                    mut_img = self.engine.run(rule, mut_seed)[1:]

                    # Compute NCC (Normalized Cross Correlation)
                    ncc = compute_ncc(base_img, mut_img)
                    mutant_scores.append(ncc)

                rule_score += np.mean(mutant_scores)

            # Calculate average across seeds, per rule
            rule_robustness_scores[rule] = rule_score / len(seeds)
            rule_images_dict[rule] = rule_images_list

        rule_phenotype_complexity_scores = self.mean_phenotype_complexity(rule_images_dict)

        return rule_robustness_scores, rule_phenotype_complexity_scores


    def compute_rule_complexity(self, rule: int):
        """Compute the complexity of the array represented by the rule itself."""
        binary_rule = int_to_binary_array_numpy(rule)
        return self.metric.calculate(binary_rule)

    def mean_phenotype_complexity(self, results):
        """Compute mean phenotype complexity per rule over all seeds."""
        mean_complexities = {}
        for rule, images in results.items():
            complexities = [self.metric.calculate(img) for img in images]
            mean_complexities[rule] = np.mean(complexities)
        return mean_complexities

"""This experiment was done close to deadline, so mostly AI-generated"""
class PopulationGrowthExperiment:
    """
    Analyzes the population dynamics (growth/shrinkage) of CA.
    Maps population change time-series to complexity.
    """

    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric):
        self.engine = engine
        self.metric = metric
        self.results = {}  # rule -> list of binary sequences

    def run_batch(self, num_seeds: int, rules=range(256)):
        """Generates population change sequences."""
        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        for rule in tqdm(rules, desc="Simulating Population Rules"):
            binary_sequences = []
            for seed in seeds:
                # Run simulation (including t=0)
                img = self.engine.run(rule, seed)

                # 1. Calculate population at each step (sum of active cells)
                population = np.sum(img, axis=1).astype(int)

                # 2. Calculate differences between steps
                # diff[i] = population[i+1] - population[i]
                diffs = np.diff(population)

                # 3. Create binary sequence: 1 if grew/same, 0 if shrank
                # astype(np.uint8) makes it 0 or 1
                binary_seq = (diffs >= 0).astype(np.uint8)

                binary_sequences.append(binary_seq)

            self.results[rule] = binary_sequences

    def analyze_and_plot(self):
        """
        Calculates complexity of the binary population sequences vs their Log Probability.
        """
        freq_map = Counter()
        complexity_map = {}

        # Flatten all sequences
        all_seqs = [seq for sublist in self.results.values() for seq in sublist]

        for seq in tqdm(all_seqs, desc="Analyzing Population Complexity"):
            # Hash the numpy array (sequence)
            h = seq.tobytes()
            freq_map[h] += 1

            if h not in complexity_map:
                # Calculate complexity of the 1D binary sequence
                complexity_map[h] = self.metric.calculate(seq)

        # Plotting
        Ks = []
        log_probs = []
        total = sum(freq_map.values())

        for h, count in freq_map.items():
            Ks.append(complexity_map[h])
            log_probs.append(np.log10(count / total))

        plt.figure(figsize=(8, 6))
        plt.scatter(Ks, log_probs, alpha=0.5, c='purple')
        plt.xlabel(f"Complexity of Population Graph ({self.metric.name()})")
        plt.ylabel("Log Probability")
        plt.title("Simplicity Bias in Population Dynamics")
        plt.grid(True, alpha=0.3)
        plt.show()
