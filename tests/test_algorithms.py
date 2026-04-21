"""
Run this file to verify the algorithms work correctly:
  python tests/test_algorithms.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.network import DistributionNetwork
from core.fitness import RestorationFitness, repair_radiality
from algorithms.bpso import run_bpso
from algorithms.cpso import run_cpso
from algorithms.hybrid_pso_ga import run_hybrid_pso_ga


def test_network_loading():
    print("Testing network loading...")
    net = DistributionNetwork()
    net.load_from_json(os.path.join(os.path.dirname(__file__), '..', 'data', 'ieee33.json'))
    assert net.G.number_of_nodes() == 33, f"Expected 33 nodes, got {net.G.number_of_nodes()}"
    assert net.get_num_switches() > 0, "No switches found"
    print(f"  ✓ Loaded {net.G.number_of_nodes()} buses, {net.get_num_switches()} switches")
    print(f"  ✓ Total load: {net.get_total_load()} kW")
    return net


def test_fitness(net):
    print("Testing fitness function...")
    states = net.get_switch_states()
    fn = RestorationFitness(net, fault_bus=26)
    cost = fn.compute(states)
    metrics = fn.decompose(states)
    assert 0 <= cost <= 10, f"Unexpected cost: {cost}"
    print(f"  ✓ Base cost: {cost:.4f}")
    print(f"  ✓ Restored load: {metrics['restored_load_kw']} kW ({metrics['restoration_efficiency']}%)")
    print(f"  ✓ Radial: {metrics['is_radial']}")


def test_bpso(net):
    print("Testing BPSO (5 iterations)...")
    pos, cost, history, metrics = run_bpso(
        net, fault_bus=26, n_particles=10, max_iter=5
    )
    assert pos is not None, "BPSO returned None"
    assert len(history) == 6, f"Expected 6 history entries, got {len(history)}"
    print(f"  ✓ Best cost: {cost:.4f}")
    print(f"  ✓ Efficiency: {metrics['restoration_efficiency']}%")
    print(f"  ✓ Switches operated: {metrics['num_switches_operated']}")
    print(f"  ✓ Runtime: {metrics['runtime_s']}s")


def test_cpso(net):
    print("Testing Continuous PSO (5 iterations)...")
    pos, cost, history, metrics = run_cpso(
        net, fault_bus=26, n_particles=10, max_iter=5
    )
    assert pos is not None
    print(f"  ✓ Best cost: {cost:.4f}")
    print(f"  ✓ Efficiency: {metrics['restoration_efficiency']}%")


def test_hybrid(net):
    print("Testing Hybrid PSO-GA (5 iterations)...")
    pos, cost, history, metrics = run_hybrid_pso_ga(
        net, fault_bus=26, n_particles=10, max_iter=5
    )
    assert pos is not None
    print(f"  ✓ Best cost: {cost:.4f}")
    print(f"  ✓ Efficiency: {metrics['restoration_efficiency']}%")


def test_repair_radiality(net):
    print("Testing radiality repair...")
    import numpy as np
    # All-closed position (likely non-radial)
    states = np.ones(net.get_num_switches())
    repaired = repair_radiality(states, net, fault_bus=26)
    net2 = net.apply_switch_states(repaired)
    print(f"  ✓ Repaired state radial: {net2.is_radial()}")


if __name__ == '__main__':
    print("=" * 50)
    print("Supply Restoration Optimizer — Unit Tests")
    print("=" * 50)
    try:
        net = test_network_loading()
        test_fitness(net)
        test_repair_radiality(net)
        test_bpso(net)
        test_cpso(net)
        test_hybrid(net)
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
