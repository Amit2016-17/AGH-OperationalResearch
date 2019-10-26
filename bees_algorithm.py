import random

import numpy as np

from model import Solution, Settings


def mutate_trucks_allocation(solution: Solution, settings: Settings):
    """ Mutates trucks allocation in given solution by randomly replacing one allocation with random value """
    i = random.randrange(settings.trucks_number)
    c = random.randrange(settings.crossings_number)
    solution.trucks_allocation[i] = c


def mutate_goods_allocation(solution: Solution, settings: Settings):
    """
    Mutates goods allocation in given solution by randomly moving goods from one truck to another while ensuring that
    restrictions are still met.
    """

    # calculate free space in each truck
    spaces = settings.truck_capacity - solution.goods_allocation.sum(axis=1)

    # select random truck that has free space
    t_to = random.choice(np.argwhere(spaces > 0).flatten())

    # select random column - goods type
    k = random.randrange(settings.goods_types_number)

    # select random truck to take the goods from
    t_from = random.randrange(settings.trucks_number)

    # calculate how much can be moved of this type between these trucks
    max_amount = min(
        solution.goods_allocation[t_from, k],    # available goods in truck that we take from
        settings.truck_capacity - spaces[t_to],  # available space in truck that we put
    )

    # move random amount
    amount = random.uniform(0.0, max_amount)
    solution.goods_allocation[t_from, k] -= amount
    solution.goods_allocation[t_to, k] += amount


def mutate_solution(solution: Solution, settings: Settings, goods_mutations: int, trucks_mutations: int):
    """ Mutates solution by applying goods and trucks allocation mutations specified number of times. """

    for _ in range(goods_mutations):
        mutate_goods_allocation(solution, settings)

    for _ in range(trucks_mutations):
        mutate_trucks_allocation(solution, settings)
