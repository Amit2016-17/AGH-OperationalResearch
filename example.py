import numpy as np

from goods_duty_optimizer import Settings, BeesSolver, stop_delta, stop_iterations, calculate_cost

# ==========================================================
#  PROBLEM SETTINGS
# ==========================================================
duties = np.array([
    [1.3, 2.3, 3.3, 7.2],
    [0.2, 1.1, 9.0, 4.1],
    [0.6, 3.4, 7.7, 2.2],
])
distances = np.array([1.2, 4.2, 6.0])
goods_amounts = np.array([10, 6, 14, 11])

settings = Settings(
    crossings_number=3,
    goods_types_number=4,
    trucks_number=5,
    truck_capacity=15,
    fuel_cost=5.3,
    duties=duties,
    distances=distances,
    goods_amounts=goods_amounts
)

# ==========================================================
#  SOLVER SETTINGS
# ==========================================================
solver = BeesSolver(
    settings=settings,
    population_size=10,
    goods_mutations=2,
    trucks_mutations=1,
    elite_sites=2,
    normal_sites=3,
    elite_site_size=7,
    normal_site_size=2
)

# ==========================================================
#  FINDING SOLUTION
# ==========================================================
solution_delta = solver.find_best_solution(stop_delta(0.001))
solution_iters = solver.find_best_solution(stop_iterations(1000))

print("Solution delta - trucks:\n", solution_delta.trucks_allocation)
print("Solution delta - goods:\n", solution_delta.goods_allocation)
print("Solution delta - cost:", calculate_cost(solution_delta, settings))

print("Solution iters - trucks:\n", solution_iters.trucks_allocation)
print("Solution iters - goods:\n", solution_iters.goods_allocation)
print("Solution iters - cost:", calculate_cost(solution_iters, settings))
