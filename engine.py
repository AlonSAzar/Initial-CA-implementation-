import numpy as np
from numba import njit


# --- Numba Optimized Functions (Static/Global scope for performance) ---
@njit
def _simulate_1d_numba(trans_rule, seed, T):

    # Determine length of the 1D lattice
    L = len(seed)
    # Create the entire space-time diagram, where T is time
    output = np.zeros((T, L), dtype=np.uint8)
    # Set the first row to be determined by seed
    output[0] = seed

    for t in range(1, T):
        for i in range(L):
            # Modulo so that for i=0, the neighbor is L-1, creating circular like behavior, etc.
            left = output[t - 1][(i - 1) % L]
            center = output[t - 1][i]
            right = output[t - 1][(i + 1) % L]
            # The three binary states  are packed into a 3 bit integer using bitwise shifts
            # Left becomes the 4's bit, center the 2's, right is 1's
            neighborhood = (left << 2) | (center << 1) | right
            # Calculate new cell value based on trans_rule and write it
            output[t][i] = trans_rule[neighborhood]
    return output


class CASimulator:
    """Base class for CA Simulations."""

    def __init__(self, L: int, T: int):
        self.L = L  # Space
        self.T = T  # Time

    def generate_seed(self, seed_type="random") -> np.ndarray:
        if seed_type == "random":
            # Basically flips a coin for each cell in the initial grid
            return np.random.randint(0, 2, size=self.L, dtype=np.uint8)
        # Create just 1 zero at the center
        elif seed_type == "center":
            s = np.zeros(self.L, dtype=np.uint8)
            s[self.L // 2] = 1
            return s
        # Else just return all zeroes
        return np.zeros(self.L, dtype=np.uint8)


class ElementaryCA(CASimulator):
    """Handles 1D Elementary Cellular Automata (Rules 0-255)."""

    def _rule_to_lut(self, rule: int) -> np.ndarray:
        """Convert rule number to lookup table of 8 values (3-bit neighborhoods)."""
        # Example: For Rule 30 (binary 00011110), the LUT would be [0, 1, 1, 1, 1, 1, 0, 0].
        return np.array([(rule >> i) & 1 for i in range(7, -1, -1)], dtype=np.uint8)

    def run(self, rule: int, seed: np.ndarray = None) -> np.ndarray:
        """Runs the simulation, returns (T, L) image."""
        if seed is None:
            seed = self.generate_seed()

        lut = self._rule_to_lut(rule)
        return _simulate_1d_numba(lut, seed, self.T)

    def mutate_rule(self, rule: int, bit_index: int) -> int:
        """Flips a single bit in the rule definition (bit_index)."""
        return rule ^ (1 << bit_index)

    def mutate_seed(self, seed: np.ndarray, bit_index: int) -> np.ndarray:
        """Flips a single bit in the seed (bit_index)."""
        seed[bit_index] = 1 - seed[bit_index]  # Flip the bit
        return seed