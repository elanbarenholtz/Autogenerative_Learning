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


# Registry of all relations
ALL_RELATIONS = {
    'fibonacci': Fibonacci(),
    'linear': LinearRecurrence(),
    'tribonacci': Tribonacci(),
    'geometric': GeometricRecurrence(),
    'fibonacci_plus_constant': FibonacciPlusConstant()
}


def get_relation(name: str) -> RecurrenceRelation:
    """Get a recurrence relation by name."""
    if name not in ALL_RELATIONS:
        raise ValueError(f"Unknown relation: {name}. Available: {list(ALL_RELATIONS.keys())}")
    return ALL_RELATIONS[name]
