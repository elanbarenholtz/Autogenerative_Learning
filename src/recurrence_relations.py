"""
Generalized framework for different recurrence relations.
"""
from typing import List, Tuple, Callable
import numpy as np


class RecurrenceRelation:
    """Base class for recurrence relations."""

    def __init__(self, name: str, order: int):
        self.name = name
        self.order = order  # How many previous terms needed

    def generate_sequence(self, seed: Tuple[int, ...], length: int) -> List[int]:
        """Generate a sequence from given seed."""
        raise NotImplementedError

    def get_training_seeds(self) -> List[Tuple[int, ...]]:
        """Get training seeds for this relation."""
        raise NotImplementedError

    def get_test_seeds(self) -> List[Tuple[int, ...]]:
        """Get test seeds for this relation."""
        raise NotImplementedError


class Fibonacci(RecurrenceRelation):
    """F(n) = F(n-1) + F(n-2)"""

    def __init__(self):
        super().__init__("Fibonacci: F(n) = F(n-1) + F(n-2)", order=2)

    def generate_sequence(self, seed: Tuple[int, int], length: int) -> List[int]:
        if length <= 0:
            return []
        if length == 1:
            return [seed[0]]

        sequence = [seed[0], seed[1]]
        for i in range(2, length):
            next_val = sequence[i-1] + sequence[i-2]
            sequence.append(next_val)
        return sequence

    def get_training_seeds(self) -> List[Tuple[int, int]]:
        return [
            (0, 1), (1, 1), (2, 3), (1, 2), (5, 8),
            (3, 5), (7, 11), (2, 5), (4, 7), (6, 10),
            (1, 3), (3, 7), (4, 6), (8, 13), (5, 12),
            (9, 14), (11, 18), (7, 12), (10, 16), (12, 19)
        ]

    def get_test_seeds(self) -> List[Tuple[int, int]]:
        return [(17, 29), (23, 37), (13, 21), (31, 50), (19, 31)]


class LinearRecurrence(RecurrenceRelation):
    """F(n) = 2*F(n-1) + F(n-2)"""

    def __init__(self):
        super().__init__("Linear: F(n) = 2*F(n-1) + F(n-2)", order=2)

    def generate_sequence(self, seed: Tuple[int, int], length: int) -> List[int]:
        if length <= 0:
            return []
        if length == 1:
            return [seed[0]]

        sequence = [seed[0], seed[1]]
        for i in range(2, length):
            next_val = 2 * sequence[i-1] + sequence[i-2]
            sequence.append(next_val)
        return sequence

    def get_training_seeds(self) -> List[Tuple[int, int]]:
        return [
            (0, 1), (1, 1), (1, 2), (2, 3), (1, 3),
            (2, 5), (3, 5), (4, 7), (3, 7), (5, 8),
            (4, 9), (6, 10), (5, 11), (7, 11), (8, 13),
            (6, 13), (9, 14), (7, 15), (10, 16), (8, 17)
        ]

    def get_test_seeds(self) -> List[Tuple[int, int]]:
        return [(11, 19), (13, 21), (12, 23), (15, 25), (14, 27)]


class Tribonacci(RecurrenceRelation):
    """F(n) = F(n-1) + F(n-2) + F(n-3)"""

    def __init__(self):
        super().__init__("Tribonacci: F(n) = F(n-1) + F(n-2) + F(n-3)", order=3)

    def generate_sequence(self, seed: Tuple[int, int, int], length: int) -> List[int]:
        if length <= 0:
            return []
        if length == 1:
            return [seed[0]]
        if length == 2:
            return [seed[0], seed[1]]

        sequence = [seed[0], seed[1], seed[2]]
        for i in range(3, length):
            next_val = sequence[i-1] + sequence[i-2] + sequence[i-3]
            sequence.append(next_val)
        return sequence

    def get_training_seeds(self) -> List[Tuple[int, int, int]]:
        return [
            (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 2), (1, 2, 3),
            (2, 3, 5), (1, 2, 4), (2, 3, 4), (1, 3, 5), (2, 4, 6),
            (3, 5, 7), (1, 4, 6), (2, 5, 7), (3, 4, 8), (4, 6, 9),
            (2, 6, 8), (3, 7, 9), (5, 7, 10), (4, 8, 11), (6, 9, 12)
        ]

    def get_test_seeds(self) -> List[Tuple[int, int, int]]:
        return [(7, 11, 15), (8, 13, 17), (9, 14, 19), (10, 16, 21), (11, 17, 23)]


class GeometricRecurrence(RecurrenceRelation):
    """F(n) = 2 * F(n-1) - Pure multiplicative"""

    def __init__(self):
        super().__init__("Geometric: F(n) = 2 * F(n-1)", order=1)

    def generate_sequence(self, seed: Tuple[int], length: int) -> List[int]:
        if length <= 0:
            return []

        sequence = [seed[0]]
        for i in range(1, length):
            next_val = 2 * sequence[i-1]
            sequence.append(next_val)
        return sequence

    def get_training_seeds(self) -> List[Tuple[int]]:
        return [
            (1,), (2,), (3,), (4,), (5,),
            (6,), (7,), (8,), (9,), (10,),
            (11,), (12,), (13,), (14,), (15,),
            (16,), (17,), (18,), (19,), (20,)
        ]

    def get_test_seeds(self) -> List[Tuple[int]]:
        return [(25,), (30,), (35,), (40,), (45,)]


class FibonacciPlusConstant(RecurrenceRelation):
    """F(n) = F(n-1) + F(n-2) + 1"""

    def __init__(self):
        super().__init__("Fibonacci+1: F(n) = F(n-1) + F(n-2) + 1", order=2)

    def generate_sequence(self, seed: Tuple[int, int], length: int) -> List[int]:
        if length <= 0:
            return []
        if length == 1:
            return [seed[0]]

        sequence = [seed[0], seed[1]]
        for i in range(2, length):
            next_val = sequence[i-1] + sequence[i-2] + 1
            sequence.append(next_val)
        return sequence

    def get_training_seeds(self) -> List[Tuple[int, int]]:
        return [
            (0, 1), (1, 1), (2, 3), (1, 2), (5, 8),
            (3, 5), (7, 11), (2, 5), (4, 7), (6, 10),
            (1, 3), (3, 7), (4, 6), (8, 13), (5, 12),
            (9, 14), (11, 18), (7, 12), (10, 16), (12, 19)
        ]

    def get_test_seeds(self) -> List[Tuple[int, int]]:
        return [(17, 29), (23, 37), (13, 21), (31, 50), (19, 31)]


class ModularArithmetic(RecurrenceRelation):
    """F(n) = (F(n-1) + F(n-2)) mod M"""

    # Hardcoded seeds for mod-31 (backward compatibility)
    _MOD31_TRAIN_SEEDS = [
        (0, 1), (1, 1), (2, 3), (5, 8), (7, 11),
        (10, 15), (13, 20), (17, 25), (21, 28), (24, 30),
        (3, 14), (6, 19), (9, 22), (12, 27), (16, 4),
        (18, 9), (23, 16), (26, 11), (29, 7), (1, 30)
    ]
    _MOD31_TEST_SEEDS = [(4, 17), (8, 23), (14, 26), (20, 5), (27, 12)]

    def __init__(self, modulus: int = 31):
        self.modulus = modulus
        super().__init__(f"Modular: F(n) = (F(n-1) + F(n-2)) mod {modulus}", order=2)

    def generate_sequence(self, seed: Tuple[int, int], length: int) -> List[int]:
        if length <= 0:
            return []
        if length == 1:
            return [seed[0]]

        sequence = [seed[0], seed[1]]
        for i in range(2, length):
            next_val = (sequence[i-1] + sequence[i-2]) % self.modulus
            sequence.append(next_val)
        return sequence

    def generate_random_seeds(self, count: int, rng: np.random.RandomState) -> List[Tuple[int, int]]:
        """Generate `count` unique seed pairs (a, b) with 0 <= a, b < modulus."""
        all_pairs = [(a, b) for a in range(self.modulus) for b in range(self.modulus)]
        rng.shuffle(all_pairs)
        return all_pairs[:count]

    def get_training_seeds(self) -> List[Tuple[int, int]]:
        if self.modulus == 31:
            return list(self._MOD31_TRAIN_SEEDS)
        # Other moduli: generate deterministically
        rng = np.random.RandomState(42)
        return self.generate_random_seeds(20, rng)

    def get_test_seeds(self) -> List[Tuple[int, int]]:
        if self.modulus == 31:
            return list(self._MOD31_TEST_SEEDS)
        # Other moduli: generate deterministically, skip first 20 (training)
        rng = np.random.RandomState(42)
        all_pairs = [(a, b) for a in range(self.modulus) for b in range(self.modulus)]
        rng.shuffle(all_pairs)
        return all_pairs[20:25]


# Registry of all relations
ALL_RELATIONS = {
    'fibonacci': Fibonacci(),
    'linear': LinearRecurrence(),
    'tribonacci': Tribonacci(),
    'geometric': GeometricRecurrence(),
    'fibonacci_plus_constant': FibonacciPlusConstant(),
    'modular_arithmetic': ModularArithmetic(31),
    'modular_arithmetic_97': ModularArithmetic(97),
    'modular_arithmetic_127': ModularArithmetic(127),
}


def get_relation(name: str) -> RecurrenceRelation:
    """Get a recurrence relation by name."""
    if name not in ALL_RELATIONS:
        raise ValueError(f"Unknown relation: {name}. Available: {list(ALL_RELATIONS.keys())}")
    return ALL_RELATIONS[name]


def get_modular_relation(modulus: int) -> ModularArithmetic:
    """Get a ModularArithmetic relation for a specific modulus."""
    key = f'modular_arithmetic_{modulus}' if modulus != 31 else 'modular_arithmetic'
    if key in ALL_RELATIONS:
        return ALL_RELATIONS[key]
    return ModularArithmetic(modulus)
