import numpy as np
import time
from core.fitness import RestorationFitness, repair_radiality


def crossover(p1, p2):
    n = len(p1)
    pt = np.random.randint(1, n)
    child = np.concatenate([p1[:pt], p2[pt:]])
    return child


def mutate(p, rate=0.1):
    child = p.copy()
    for i in range(len(child)):
        if np.random.rand() < rate:
            child[i] = 1.0 - child[i]
    return child


def run_hybrid_pso_ga(network, fault_bus=None, weights=None,
                      n_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5,
                      ga_fraction=0.2, mutation_rate=0.1,
                      callback=None):
    """
    Hybrid PSO + GA: After each PSO iteration, apply crossover/mutation 
    to the top ga_fraction of particles to escape local optima.
    Returns: (best_position, best_cost, history, metrics)
    """
    start_time = time.time()
    fitness_fn = RestorationFitness(network, fault_bus, weights)

    n_sw = network.get_num_switches()
    if n_sw == 0:
        return None, None, [], {}

    pos = np.random.randint(0, 2, (n_particles, n_sw)).astype(float)
    vel = np.zeros((n_particles, n_sw))

    orig_states = network.get_switch_states()
    pos[0] = orig_states.copy()

    for i in range(n_particles):
        pos[i] = repair_radiality(pos[i], network, fault_bus)

    pbest_pos = pos.copy()
    pbest_cost = np.array([fitness_fn.compute(p) for p in pos])

    gbest_idx = np.argmin(pbest_cost)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]

    history = [float(gbest_cost)]
    eval_count = n_particles
    n_ga = max(2, int(n_particles * ga_fraction))

    for it in range(max_iter):
        # --- PSO update ---
        r1 = np.random.rand(n_particles, n_sw)
        r2 = np.random.rand(n_particles, n_sw)

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        vel = np.clip(vel, -6, 6)

        S = 1.0 / (1.0 + np.exp(-vel))
        pos = (np.random.rand(n_particles, n_sw) < S).astype(float)

        for i in range(n_particles):
            pos[i] = repair_radiality(pos[i], network, fault_bus)

        costs = np.array([fitness_fn.compute(p) for p in pos])
        eval_count += n_particles

        # --- GA crossover/mutation on top particles ---
        sorted_idx = np.argsort(costs)
        top_indices = sorted_idx[:n_ga]

        new_children = []
        for k in range(n_ga // 2):
            p1 = pos[top_indices[k * 2]]
            p2 = pos[top_indices[k * 2 + 1 if k * 2 + 1 < n_ga else 0]]
            child1 = crossover(p1, p2)
            child2 = crossover(p2, p1)
            child1 = mutate(repair_radiality(child1, network, fault_bus), mutation_rate)
            child2 = mutate(repair_radiality(child2, network, fault_bus), mutation_rate)
            child1 = repair_radiality(child1, network, fault_bus)
            child2 = repair_radiality(child2, network, fault_bus)
            new_children.extend([child1, child2])

        # Replace worst particles with children if they're better
        worst_indices = sorted_idx[-len(new_children):]
        for j, child in enumerate(new_children):
            c_cost = fitness_fn.compute(child)
            eval_count += 1
            if c_cost < costs[worst_indices[j]]:
                pos[worst_indices[j]] = child
                costs[worst_indices[j]] = c_cost

        improved = costs < pbest_cost
        pbest_pos[improved] = pos[improved]
        pbest_cost[improved] = costs[improved]

        best_idx = np.argmin(pbest_cost)
        if pbest_cost[best_idx] < gbest_cost:
            gbest_pos = pbest_pos[best_idx].copy()
            gbest_cost = pbest_cost[best_idx]

        history.append(float(gbest_cost))

        if callback:
            callback(it + 1, max_iter, float(gbest_cost))

    elapsed = time.time() - start_time
    metrics = fitness_fn.decompose(gbest_pos)
    metrics['runtime_s'] = round(elapsed, 3)
    metrics['eval_count'] = eval_count
    metrics['algorithm'] = 'Hybrid PSO-GA'
    metrics['convergence_iter'] = int(np.argmin(history))

    return gbest_pos, float(gbest_cost), history, metrics
