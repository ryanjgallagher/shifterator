import random
from itertools import product

from shifterator import EntropyShift, JSDivergenceShift, TsallisDivergenceShift

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
    shift.get_shift_graph()


def test_entropy_shift_1():
    shift = EntropyShift(system_1_a, system_2_a)
    shift.get_shift_graph()


def test_tsallis_shift_1():
    shift = TsallisDivergenceShift(system_1_a, system_2_a)
    shift.get_shift_graph()


def test_jsd_shift_2():
    shift = JSDivergenceShift(system_1_b, system_2_b)
    shift.get_shift_graph()


def test_entropy_shift_2():
    shift = EntropyShift(system_1_b, system_2_b)
    shift.get_shift_graph()


def test_tsallis_shift_2():
    shift = TsallisDivergenceShift(system_1_b, system_2_b)
    shift.get_shift_graph()


if __name__ == "__main__":
    test_jsd_shift_1()
    test_jsd_shift_2()

    test_tsallis_shift_1()
    test_tsallis_shift_2()

    test_entropy_shift_1()
    test_entropy_shift_2()
