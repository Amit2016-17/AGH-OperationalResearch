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
    return all(solution.goods_allocation.sum(axis=1) <= settings.truck_capacity)


def validate_goods_total(solution: Solution, settings: Settings) -> bool:
    """ Validates (2): sum of goods of specific type across all trucks has to be equal to goods amount of this type """
    return all(solution.goods_allocation.sum(axis=0) == settings.goods_amounts)


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
