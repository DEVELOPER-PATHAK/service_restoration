import numpy as np
import networkx as nx


class RestorationFitness:
    def __init__(self, network, fault_bus=None, weights=None):
        self.base_network = network
        self.fault_bus = fault_bus
        self.original_states = network.get_switch_states()
        if weights is None:
            self.weights = {
                'ens': 0.5,
                'priority': 0.2,
                'seq': 0.1,
                'num_sw': 0.1,
                'type': 0.1
            }
        else:
            self.weights = weights

    def update_weights(self, w_ens, w_priority, w_seq, w_num, w_type):
        total = w_ens + w_priority + w_seq + w_num + w_type
        if total == 0:
            total = 1.0
        self.weights = {
            'ens': w_ens / total,
            'priority': w_priority / total,
            'seq': w_seq / total,
            'num_sw': w_num / total,
            'type': w_type / total
        }

    def compute(self, states):
        net = self.base_network.apply_switch_states(states)

        # Challenge A: Energy Not Supplied
        total_load = self.base_network.get_total_load()
        restored_load = net.get_restored_load(self.fault_bus)
        # Exclude fault bus load from total
        fault_load = 0
        if self.fault_bus:
            fault_load = self.base_network.buses.get(self.fault_bus, {}).get('load_kw', 0)
        ens = (total_load - fault_load - restored_load) / max(total_load, 1)
        ens = max(0.0, ens)

        # Challenge F: Priority penalty
        oos = net.get_out_of_service_buses(self.fault_bus)
        priority_penalty = 0.0
        for bus_id in oos:
            p = self.base_network.buses.get(bus_id, {}).get('priority', 3)
            priority_penalty += (4 - p)  # priority 1 gives penalty 3, priority 3 gives 1
        max_pp = sum((4 - b.get('priority', 3)) for bid, b in self.base_network.buses.items()
                     if bid != self.base_network.substation and bid != self.fault_bus)
        priority_penalty /= max(max_pp, 1)

        # Challenge K: Switch sequence cost (simplified - manual switches cost more)
        switch_info = self.base_network.get_switch_info()
        seq_cost = 0.0
        num_changed = 0
        for idx, info in enumerate(switch_info):
            orig = self.original_states[idx]
            curr = states[idx]
            if abs(orig - curr) > 0.5:
                num_changed += 1
                if info.get('type', 'auto') == 'manual':
                    seq_cost += 3.0
                else:
                    seq_cost += 1.0
        max_seq = len(switch_info) * 3.0
        seq_cost /= max(max_seq, 1)

        # Challenge L: Number of operated switches
        num_sw_cost = num_changed / max(len(switch_info), 1)

        # Challenge M: Type penalty (manual switches)
        type_penalty = 0.0
        for idx, info in enumerate(switch_info):
            orig = self.original_states[idx]
            curr = states[idx]
            if abs(orig - curr) > 0.5 and info.get('type', 'auto') == 'manual':
                type_penalty += 1.0
        type_penalty /= max(len(switch_info), 1)

        # Radiality penalty
        radial_penalty = 0.0 if net.is_radial() else 1.0

        total_cost = (
            self.weights['ens'] * ens
            + self.weights['priority'] * priority_penalty
            + self.weights['seq'] * seq_cost
            + self.weights['num_sw'] * num_sw_cost
            + self.weights['type'] * type_penalty
            + 2.0 * radial_penalty
        )
        return total_cost

    def decompose(self, states):
        net = self.base_network.apply_switch_states(states)
        total_load = self.base_network.get_total_load()
        restored_load = net.get_restored_load(self.fault_bus)
        fault_load = 0
        if self.fault_bus:
            fault_load = self.base_network.buses.get(self.fault_bus, {}).get('load_kw', 0)
        ens = max(0.0, total_load - fault_load - restored_load)
        oos = net.get_out_of_service_buses(self.fault_bus)
        n_restored = len(self.base_network.buses) - 1 - len(oos)
        if self.fault_bus:
            n_restored = len(self.base_network.buses) - 2 - len(oos)

        switch_info = self.base_network.get_switch_info()
        num_changed = sum(
            1 for idx, _ in enumerate(switch_info)
            if abs(self.original_states[idx] - states[idx]) > 0.5
        )
        manual_ops = sum(
            1 for idx, info in enumerate(switch_info)
            if abs(self.original_states[idx] - states[idx]) > 0.5
            and info.get('type', 'auto') == 'manual'
        )
        restoration_efficiency = restored_load / max(total_load - fault_load, 1) * 100

        return {
            'ens_kw': round(ens, 2),
            'restored_load_kw': round(restored_load, 2),
            'total_load_kw': round(total_load, 2),
            'restoration_efficiency': round(restoration_efficiency, 2),
            'num_oos_buses': len(oos),
            'num_restored_buses': n_restored,
            'num_switches_operated': num_changed,
            'manual_switches_operated': manual_ops,
            'is_radial': net.is_radial(),
            'total_cost': round(self.compute(states), 6)
        }


def repair_radiality(states, network, fault_bus=None):
    """Repair operator to enforce radiality by opening loops."""
    states = np.array(states, dtype=float)

    # Force switches around fault bus to be open
    if fault_bus is not None:
        for idx, sid in enumerate(network.switch_ids):
            br = network.branches[sid]
            if br['from'] == fault_bus or br['to'] == fault_bus:
                states[idx] = 0.0

    # Remove loops iteratively
    max_iter = 200
    for _ in range(max_iter):
        net = network.apply_switch_states(states)
        try:
            cycle = nx.find_cycle(net.G)
            # Open a switch in this cycle (prefer tie/normally-open switches)
            opened = False
            for edge in reversed(cycle):
                for idx, sid in enumerate(network.switch_ids):
                    br = network.branches[sid]
                    if ((br['from'] == edge[0] and br['to'] == edge[1]) or
                            (br['from'] == edge[1] and br['to'] == edge[0])):
                        states[idx] = 0.0
                        opened = True
                        break
                if opened:
                    break
        except nx.exception.NetworkXNoCycle:
            break

    return states
