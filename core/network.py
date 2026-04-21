import networkx as nx
import numpy as np
import json
import os


class DistributionNetwork:
    def __init__(self):
        self.G = nx.Graph()
        self.substation = 1
        self.buses = {}
        self.branches = {}
        self.switch_ids = []
        self.switch_index_map = {}
        self.name = "Custom Network"

    def load_from_json(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.name = data.get('name', 'Network')
        self.substation = data.get('substation', 1)
        self.buses = {b['id']: b for b in data['buses']}
        for node_id, attrs in self.buses.items():
            self.G.add_node(node_id, **attrs)
        self.branches = {}
        for br in data['branches']:
            self.branches[br['id']] = br
            self.G.add_edge(br['from'], br['to'], **br)
        self._build_switch_index()
        return self

    def load_from_dict(self, data):
        self.name = data.get('name', 'Custom')
        self.substation = data.get('substation', 1)
        self.buses = {b['id']: b for b in data['buses']}
        self.G = nx.Graph()
        for node_id, attrs in self.buses.items():
            self.G.add_node(node_id, **attrs)
        self.branches = {}
        for br in data['branches']:
            self.branches[br['id']] = br
            self.G.add_edge(br['from'], br['to'], **br)
        self._build_switch_index()
        return self

    def _build_switch_index(self):
        self.switch_ids = sorted([
            bid for bid, br in self.branches.items() if br.get('switch', False)
        ])
        self.switch_index_map = {sid: idx for idx, sid in enumerate(self.switch_ids)}

    def get_switch_states(self):
        return np.array([
            1.0 if self.branches[sid]['state'] == 'closed' else 0.0
            for sid in self.switch_ids
        ])

    def apply_switch_states(self, states):
        net = DistributionNetwork()
        net.name = self.name
        net.substation = self.substation
        net.buses = dict(self.buses)
        net.branches = {k: dict(v) for k, v in self.branches.items()}
        net.G = nx.Graph()
        for nid, attrs in net.buses.items():
            net.G.add_node(nid, **attrs)
        for idx, sid in enumerate(self.switch_ids):
            net.branches[sid]['state'] = 'closed' if states[idx] > 0.5 else 'open'
        for bid, br in net.branches.items():
            if br['state'] == 'closed':
                net.G.add_edge(br['from'], br['to'], **br)
        net.switch_ids = list(self.switch_ids)
        net.switch_index_map = dict(self.switch_index_map)
        return net

    def is_radial(self):
        connected = nx.is_connected(self.G)
        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        return connected and (n_edges == n_nodes - 1)

    def get_energized_buses(self):
        if self.substation not in self.G:
            return set()
        return set(nx.node_connected_component(self.G, self.substation))

    def get_out_of_service_buses(self, fault_bus=None):
        energized = self.get_energized_buses()
        all_buses = set(self.buses.keys())
        oos = all_buses - energized
        if fault_bus and fault_bus in oos:
            oos.discard(fault_bus)
        return oos

    def get_total_load(self):
        return sum(b.get('load_kw', 0) for b in self.buses.values())

    def get_restored_load(self, fault_bus=None):
        energized = self.get_energized_buses()
        total = 0.0
        for bid in energized:
            if bid == self.substation:
                continue
            if fault_bus and bid == fault_bus:
                continue
            total += self.buses[bid].get('load_kw', 0)
        return total

    def get_num_switches(self):
        return len(self.switch_ids)

    def get_switch_info(self):
        return [self.branches[sid] for sid in self.switch_ids]

    def to_dict(self):
        return {
            'name': self.name,
            'substation': self.substation,
            'buses': list(self.buses.values()),
            'branches': list(self.branches.values())
        }
