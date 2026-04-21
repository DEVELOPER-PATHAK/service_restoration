import numpy as np
import time
from core.fitness import RestorationFitness, repair_radiality


def run_bpso(network, fault_bus=None, weights=None,
             n_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5,
             callback=None):
    """
    Binary PSO for supply restoration.
    Returns: (best_position, best_cost, history, metrics)
    """
    start_time = time.time()
    fitness_fn = RestorationFitness(network, fault_bus, weights)

    n_sw = network.get_num_switches()
    if n_sw == 0:
        return None, None, [], {}

    # Initialize population
    pos = np.random.randint(0, 2, (n_particles, n_sw)).astype(float)
    vel = np.zeros((n_particles, n_sw))

    # Include original configuration as one particle
    orig_states = network.get_switch_states()
    pos[0] = orig_states.copy()

    # Repair each particle for radiality
    for i in range(n_particles):
        pos[i] = repair_radiality(pos[i], network, fault_bus)

    pbest_pos = pos.copy()
    pbest_cost = np.array([fitness_fn.compute(p) for p in pos])

    gbest_idx = np.argmin(pbest_cost)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]

    history = [float(gbest_cost)]
    eval_count = n_particles

    for it in range(max_iter):
        r1 = np.random.rand(n_particles, n_sw)
        r2 = np.random.rand(n_particles, n_sw)

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))

        # Clamp velocity
        vel = np.clip(vel, -6, 6)

        # Sigmoid transfer function
        S = 1.0 / (1.0 + np.exp(-vel))

        # Update position
        pos = (np.random.rand(n_particles, n_sw) < S).astype(float)

        # Repair radiality
        for i in range(n_particles):
            pos[i] = repair_radiality(pos[i], network, fault_bus)

        costs = np.array([fitness_fn.compute(p) for p in pos])
        eval_count += n_particles

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
    metrics['algorithm'] = 'BPSO'
    metrics['convergence_iter'] = int(np.argmin(history))

    return gbest_pos, float(gbest_cost), history, metrics
