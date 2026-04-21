import numpy as np
import time
from core.fitness import RestorationFitness, repair_radiality


def run_cpso(network, fault_bus=None, weights=None,
             n_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5,
             callback=None):
    """
    Continuous PSO with threshold binarization for supply restoration.
    Returns: (best_position, best_cost, history, metrics)
    """
    start_time = time.time()
    fitness_fn = RestorationFitness(network, fault_bus, weights)

    n_sw = network.get_num_switches()
    if n_sw == 0:
        return None, None, [], {}

    # Initialize continuous positions in [0, 1]
    pos = np.random.rand(n_particles, n_sw)
    vel = np.random.uniform(-0.5, 0.5, (n_particles, n_sw))

    # Include original configuration
    orig_states = network.get_switch_states()
    pos[0] = orig_states.copy()

    def binarize(p):
        return (p > 0.5).astype(float)

    def repair(p):
        return repair_radiality(binarize(p), network, fault_bus)

    pbest_pos = pos.copy()
    pbest_bin = np.array([repair(p) for p in pos])
    pbest_cost = np.array([fitness_fn.compute(b) for b in pbest_bin])

    gbest_idx = np.argmin(pbest_cost)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_bin = pbest_bin[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]

    history = [float(gbest_cost)]
    eval_count = n_particles

    for it in range(max_iter):
        r1 = np.random.rand(n_particles, n_sw)
        r2 = np.random.rand(n_particles, n_sw)

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))

        vel = np.clip(vel, -2.0, 2.0)
        pos = np.clip(pos + vel, 0.0, 1.0)

        bin_pos = np.array([repair(p) for p in pos])
        costs = np.array([fitness_fn.compute(b) for b in bin_pos])
        eval_count += n_particles

        improved = costs < pbest_cost
        pbest_pos[improved] = pos[improved]
        pbest_bin[improved] = bin_pos[improved]
        pbest_cost[improved] = costs[improved]

        best_idx = np.argmin(pbest_cost)
        if pbest_cost[best_idx] < gbest_cost:
            gbest_pos = pbest_pos[best_idx].copy()
            gbest_bin = pbest_bin[best_idx].copy()
            gbest_cost = pbest_cost[best_idx]

        history.append(float(gbest_cost))

        if callback:
            callback(it + 1, max_iter, float(gbest_cost))

    elapsed = time.time() - start_time
    metrics = fitness_fn.decompose(gbest_bin)
    metrics['runtime_s'] = round(elapsed, 3)
    metrics['eval_count'] = eval_count
    metrics['algorithm'] = 'Continuous PSO'
    metrics['convergence_iter'] = int(np.argmin(history))

    return gbest_bin, float(gbest_cost), history, metrics
