import random
from itertools import product

from shifterator import EntropyShift


def test_entropy_shift_1():
    system_1 = {"A": 10, "B": 10, "C": 10, "D": 10, "E": 10, "F": 10, "G": 10}
    system_2 = {"B": 10, "C": 10, "D": 10, "E": 10, "F": 10, "G": 10, "H": 10}

    shift = EntropyShift(system_1, system_2)
    shift.get_shift_graph()


def test_entropy_shift_2():
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    system_1 = {
        c[0] + c[1]: int(random.expovariate(1 / 100)) + 1
        for c in product(chars, repeat=2)
        if random.random() > 0.05
    }
    system_2 = {
        c[0] + c[1]: int(random.expovariate(1 / 100)) + 1
        for c in product(chars, repeat=2)
        if random.random() > 0.05
    }

    shift = EntropyShift(system_1, system_2)
    shift.get_shift_graph()


if __name__ == "__main__":
    test_entropy_shift_1()
    test_entropy_shift_2()
