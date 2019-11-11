import random
import copy
from typing import List, Callable

import numpy as np

from model import Solution, Settings, calculate_cost, generate_random_solution


class BeesSolver:
    """ Solver that uses bees algorithm to solve our problem. """

    def __init__(self, settings: Settings, population_size: int,
                 goods_mutations: int, trucks_mutations: int,
                 elite_sites: int, normal_sites: int,
                 elite_site_size: int, normal_site_size: int):
        """
        Creates bees solver.

        :param settings: settings of our problem
        :param population_size: size of each population
        :param goods_mutations: how many times mutation on goods allocation should be applied for each neighbour
        :param trucks_mutations: how many times mutation on trucks allocation should be applied for each neighbour
        :param elite_sites: number of solutions that will be selected as elite
        :param normal_sites: number of solutions that will be selected as normal
        :param elite_site_size:
        :param normal_site_size:
        """
        self.settings = settings
        self.population_size = population_size
        self.goods_mutations = goods_mutations
        self.trucks_mutations = trucks_mutations
        self.elite_sites = elite_sites
        self.normal_sites = normal_sites
        self.elite_site_size = elite_site_size
        self.normal_site_size = normal_site_size

        self.population: List[Solution] = []

    def _mutate_trucks_allocation(self, solution: Solution):
        """
        Mutates (inplace) trucks allocation in given solution by randomly replacing one allocation with random value.
        """
        i = random.randrange(self.settings.trucks_number)
        c = random.randrange(self.settings.crossings_number)
        solution.trucks_allocation[i] = c

    def _mutate_goods_allocation(self, solution: Solution):
        """
        Mutates (inplace) goods allocation in given solution by randomly moving goods from one truck to another while
        ensuring that restrictions are still met.
        """

        # calculate free space in each truck
        spaces = self.settings.truck_capacity - solution.goods_allocation.sum(axis=1)

        # select random column - goods type
        k = random.randrange(self.settings.goods_types_number)

        # select random truck that has free space
        t_to = random.choice(np.argwhere(spaces > 0).flatten())

        # select random truck that has some goods of type k
        t_from = random.choice(np.argwhere(solution.goods_allocation[:, k] > 0).flatten())

        # calculate how much can be moved of this type between these trucks
        max_amount = min(
            solution.goods_allocation[t_from, k],    # available goods in truck that we take from
            spaces[t_to],  # available space in truck that we put
        )

        # move random amount
        amount = random.randint(1, max_amount)
        solution.goods_allocation[t_from, k] -= amount
        solution.goods_allocation[t_to, k] += amount

    def _mutate_solution(self, solution: Solution):
        """
        Mutates (inplace) solution by applying goods and trucks allocation mutations specified number of times.
        """

        for _ in range(self.goods_mutations):
            self._mutate_goods_allocation(solution)

        for _ in range(self.trucks_mutations):
            self._mutate_trucks_allocation(solution)

    def _find_best_neighbour(self, solution: Solution, neighbours_count: int) -> Solution:
        """
        Finds best neighbour from a randomly mutated population of neighbours of given solution.

        :returns: best solution from neighbourhood, or given solution if it was the best
        """

        # create neighbours
        neighbours = [copy.deepcopy(solution) for _ in range(neighbours_count)]

        # mutate neighbours
        for n in neighbours:
            self._mutate_solution(n)

        # append original solution to also be included in evaluation
        neighbours.append(solution)

        # find and return best neighbour
        return sorted(neighbours, key=lambda sol: calculate_cost(sol, self.settings))[0]

    def simulate_population(self):
        """ Applies one step of bees algorithm. """

        # sort population
        self.population.sort(key=lambda sol: calculate_cost(sol, self.settings))

        # find best neighbours in elite sites
        for i in range(0, self.elite_sites):
            self.population[i] = self._find_best_neighbour(self.population[i], self.elite_site_size)

        # find best neighbours in normal sites
        for i in range(self.elite_sites, self.elite_sites + self.normal_sites):
            self.population[i] = self._find_best_neighbour(self.population[i], self.normal_site_size)

        # replace all other solutions with random ones
        for i in range(self.elite_sites + self.normal_sites, len(self.population)):
            self.population[i] = generate_random_solution(self.settings)

    def init_population(self):
        """ Generates random population. """
        self.population = [generate_random_solution(self.settings) for _ in range(self.population_size)]

    def current_cost(self) -> float:
        """ Returns current cost of first solution from population. """
        return calculate_cost(self.population[0], self.settings)

    def find_best_solution(self, stop_func: Callable[[int, float, float], bool]) -> Solution:
        """
        Finds best solution using this solver.

        :param stop_func: function that receives loops count, last cost and new cost and returns true if iteration
                process should stop
        :return: found solution
        """
        loops = 0
        last_cost = float('inf')

        # create random population
        self.init_population()

        # while stop function returns true simulate population
        while not stop_func(loops, last_cost, self.current_cost()):
            loops += 1
            last_cost = self.current_cost()
            self.simulate_population()

        return self.population[0]


def stop_delta(delta: float):
    """ Creates stop function that stops when change in cost is lower than delta. """
    def stop_func(last_cost: float, new_cost: float):
        return last_cost - new_cost < delta
    return stop_func


def stop_iterations(iterations: int):
    """ Creates stop function that stops after certain number of loops. """
    def stop_func(loops: int, last_cost: float, new_cost: float):
        return loops >= iterations
    return stop_func
