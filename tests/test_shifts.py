import random
from itertools import product

from shifterator import EntropyShift, JSDivergenceShift

system_1_a = {"A": 10, "B": 10, "C": 10, "D": 10, "E": 10, "F": 10, "G": 10}
system_2_a = {"B": 10, "C": 10, "D": 10, "E": 10, "F": 10, "G": 10, "H": 10}
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
system_1_b = {
    c[0] + c[1]: int(random.expovariate(1 / 100)) + 1
    for c in product(chars, repeat=2)
    if random.random() > 0.05
}
system_2_b = {
    c[0] + c[1]: int(random.expovariate(1 / 100)) + 1
    for c in product(chars, repeat=2)
    if random.random() > 0.05
}


def test_jsd_shift_1():
    shift = JSDivergenceShift(system_1_a, system_2_a)
    shift.get_shift_graph(system_names=["1A", "2A"])


def test_entropy_shift_1():
    shift = EntropyShift(system_1_a, system_2_a)
    shift.get_shift_graph(system_names=["1A", "2A"])


def test_tsallis_shift_plus_1():
    shift = EntropyShift(system_1_a, system_2_a, alpha=2)
    shift.get_shift_graph(system_names=["1A", "2A"])


def test_tsallis_shift_minus_1():
    shift = EntropyShift(system_1_a, system_2_a, alpha=0.5)
    shift.get_shift_graph(system_names=["1A", "2A"])


def test_jsd_shift_2():
    shift = JSDivergenceShift(system_1_b, system_2_b)
    shift.get_shift_graph(system_names=["1B", "2B"])


def test_entropy_shift_2():
    shift = EntropyShift(system_1_b, system_2_b)
    shift.get_shift_graph(system_names=["1B", "2B"])


def test_tsallis_shift_plus_2():
    shift = EntropyShift(system_1_b, system_2_b, alpha=2)
    shift.get_shift_graph(system_names=["1B", "2B"])


def test_tsallis_shift_minus_2():
    shift = EntropyShift(system_1_b, system_2_b, alpha=0.5)
    shift.get_shift_graph(system_names=["1B", "2B"])


if __name__ == "__main__":
    test_jsd_shift_1()
    test_jsd_shift_2()

    test_tsallis_shift_plus_1()
    test_tsallis_shift_plus_2()

    test_tsallis_shift_minus_1()
    test_tsallis_shift_minus_2()

    test_entropy_shift_1()
    test_entropy_shift_2()
