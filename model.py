from typing import NamedTuple

import numpy as np


# ==========================================================
#  TYPES
# ==========================================================
class Settings(NamedTuple):
    """ Tuple that represents settings that defines our problem to solve """
    crossings_number: int           # n - number of border crossings
    goods_types_number: int         # m - number of types of goods
    trucks_number: int              # p - number of available trucks
    truck_capacity: float           # v - capacity of single truck, in kg
    fuel_cost: float                # f - cost of fuel for single truck for one km
    duties: np.ndarray              # c - array of shape (n, m) that defines duties
    distances: np.ndarray           # d - array of shape (n) that defines distance in km to crossings
    goods_amounts: np.ndarray       # t - array of shape (m) that defines amounts available goods


class Solution(NamedTuple):
    """ Tuple that represents solution of our problem """
    trucks_allocation: np.ndarray   # a - array of shape (p) that defines crossing for each truck
    goods_allocation: np.ndarray    # b - array of shape (p, m) that defines allocation of goods for given truck


# ==========================================================
#  TYPES VALIDATION
# ==========================================================
def validate_trucks_capacity(solution: Solution, settings: Settings) -> bool:
    """ Validates (1): each truck has a sum of goods less or equal to truck capacity """
    return np.all(solution.goods_allocation.sum(axis=1) <= settings.truck_capacity)


def validate_goods_total(solution: Solution, settings: Settings) -> bool:
    """ Validates (2): sum of goods of specific type across all trucks has to be equal to goods amount of this type """
    return np.allclose(solution.goods_allocation.sum(axis=0), settings.goods_amounts)


def validate_solution(solution: Solution, settings: Settings) -> bool:
    """ Validates that all restrictions (1, 2) are meet """
    return validate_trucks_capacity(solution, settings) and validate_goods_total(solution, settings)


# ==========================================================
#  COST CALCULATION
# ==========================================================
def calculate_cost(solution: Solution, settings: Settings) -> float:
    """ Calculates cost of given solution """
    duties_cost = np.sum(solution.goods_allocation * settings.duties[solution.trucks_allocation, :])
    fuel_cost = np.sum(settings.fuel_cost * settings.distances[solution.trucks_allocation])
    return duties_cost + fuel_cost


# ==========================================================
#  RANDOM SOLUTION GENERATION
# ==========================================================
def generate_random_truck_allocation(settings: Settings) -> np.ndarray:
    """ Generates random truck """
    return np.random.randint(low=0, high=settings.crossings_number, size=settings.trucks_number)


def generate_random_goods_allocation(settings: Settings) -> np.ndarray:
    """ Generates random goods allocation that meets restrictions (1, 2) """

    # check if there exists any proper allocation
    if settings.trucks_number * settings.truck_capacity < settings.goods_amounts.sum():
        raise ValueError('Total amount of goods exceeds total capacity of trucks, therefore no solution exists')

    # generate random array
    allocation = np.random.rand(settings.trucks_number, settings.goods_types_number)

    # ensure that restriction 2 is meet by normalizing columns so that sums of columns match total amount of goods
    allocation /= allocation.sum(axis=0)
    allocation *= settings.goods_amounts

    # ensure that restriction 1 is meet by by moving goods from overloaded trucks to the most empty trucks
    while np.any(allocation.sum(axis=1) > settings.truck_capacity):
        # find the most and the least loaded trucks
        sums = allocation.sum(axis=1)
        k1 = sums.argmax()
        k2 = sums.argmin()
        # select which of the goods to move
        j = allocation[k1].argmax()
        # calculate how much to move between trucks
        amount = min(allocation[k1, j], sums[k1]-settings.truck_capacity)
        # move from k1-th truck to k2-truck
        allocation[k1, j] -= amount
        allocation[k2, j] += amount

    return allocation


def generate_random_solution(settings: Settings) -> Solution:
    """ Generates random solution that meets restrictions """
    solution = Solution(
        trucks_allocation=generate_random_truck_allocation(settings),
        goods_allocation=generate_random_goods_allocation(settings)
    )

    # ensure that generates solution meets restrictions
    if not validate_solution(solution, settings):
        raise RuntimeError('Generated solution does not meet restrictions')

    return solution
