import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, concurrence, state_fidelity, entropy as qiskit_entropy, mutual_information, negativity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import umap.umap_ as umap
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

class SimulationEngine:
    def __init__(self, noise_level=0.001):
        self.simulator = AerSimulator()
        self.noise_model = self._create_noise_model(noise_level)

    def _create_noise_model(self, noise_level):
        if noise_level == 0: return None
        error_1 = depolarizing_error(noise_level, 1)
        error_2 = depolarizing_error(noise_level, 2)
        noise_model = NoiseModel()
        one_qubit_gates = ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z', 'id', 'rz', 'sx']
        two_qubit_gates = ['cx', 'swap', 'cu1', 'cz']
        noise_model.add_all_qubit_quantum_error(error_1, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2, two_qubit_gates)
        return noise_model

    def get_stepwise_snapshots(self, qc: QuantumCircuit):
        snapshots = []
        cumulative_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
        initial_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
        initial_qc.save_density_matrix()
        result = self.simulator.run(initial_qc, noise_model=self.noise_model).result()
        snapshots.append({"step": 0, "gate": "Initial State", "state": result.data()['density_matrix']})
        for i, instruction in enumerate(qc.data):
            cumulative_qc.append(instruction)
            snapshot_qc = cumulative_qc.copy()
            snapshot_qc.save_density_matrix()
            result = self.simulator.run(snapshot_qc, noise_model=self.noise_model).result()
            gate_info = f"{instruction.operation.name.upper()} q{', '.join([str(qc.qubits.index(q)) for q in instruction.qubits])}"
            snapshots.append({"step": i + 1, "gate": gate_info, "state": result.data()['density_matrix']})
        return snapshots

class MetricComputer:
    def compute_all_metrics(self, state: DensityMatrix, ideal_state: DensityMatrix):
        num_qubits = state.num_qubits
        metrics = {"qubits": [], "pairs": {}, "global": {}}
        metrics["global"]["fidelity"] = state_fidelity(ideal_state, state)
        
        qubit_states = [partial_trace(state, [q for q in range(num_qubits) if q != i]) for i in range(num_qubits)]
        for i in range(num_qubits):
            metrics["qubits"].append({"entropy": qiskit_entropy(qubit_states[i])})

        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                pair_key = f"{i}-{j}"
                qubit_list = [i, j]
                try:
                    rho_pair = partial_trace(state, [q for q in range(num_qubits) if q not in qubit_list])
                    conc = concurrence(rho_pair)
                    neg = negativity(rho_pair)
                    mut_info = mutual_information(state, qubit_list)
                except Exception:
                    conc, neg, mut_info = 0.0, 0.0, 0.0
                metrics["pairs"][pair_key] = {"concurrence": conc, "negativity": neg, "mutual_information": mut_info}
        return metrics

class DiscoveryModules:
    @staticmethod
    def qecart_analysis(step_data: pd.Series):
        num_qubits = len(step_data['metrics']['qubits'])
        correlation_matrix = np.identity(num_qubits)
        for pair, data in step_data['metrics']['pairs'].items():
            q1, q2 = map(int, pair.split('-'))
            score = (data.get('concurrence', 0) + data.get('negativity', 0) + data.get('mutual_information', 0)) / 3
            correlation_matrix[q1, q2] = correlation_matrix[q2, q1] = score
        return correlation_matrix

    @staticmethod
    def qecg_analysis(qc: QuantumCircuit, history_df: pd.DataFrame, error_qubit=0, error_gate_idx=1):
        if len(qc.data) <= error_gate_idx: return {"error": "Circuit too short."}
        baseline_fidelities = history_df['metrics'].apply(lambda x: x['global']['fidelity']).tolist()
        error_qc = qc.copy()
        error_qc.x(error_qubit).c_if(0, 0)
        causal_graph = nx.DiGraph()
        causal_graph.add_node(error_gate_idx, label=f"Error on Q{error_qubit}", color='red')
        for i in range(error_gate_idx + 1, len(history_df)):
            prev_fidelity = baseline_fidelities[i-1]
            curr_fidelity = baseline_fidelities[i]
            impact = max(0, prev_fidelity - curr_fidelity)
            if impact > 0.01:
                causal_graph.add_node(i, label=history_df.iloc[i]['gate'], color='orange')
                causal_graph.add_edge(i-1, i, weight=impact)
        return causal_graph

    @staticmethod
    def qip_analysis(history_df: pd.DataFrame):
        num_steps = len(history_df)
        num_qubits = len(history_df.iloc[0]['metrics']['qubits'])
        pressure_matrix = np.zeros((num_qubits, num_steps))
        for step in range(1, num_steps):
            prev_entropies = np.array([q['entropy'] for q in history_df.iloc[step-1]['metrics']['qubits']])
            curr_entropies = np.array([q['entropy'] for q in history_df.iloc[step]['metrics']['qubits']])
            pressure_matrix[:, step] = curr_entropies - prev_entropies
        return pressure_matrix

    @staticmethod
    def qdf_analysis(noisy_df: pd.DataFrame, ideal_df: pd.DataFrame):
        noisy_fidelities = noisy_df['metrics'].apply(lambda x: x['global']['fidelity'])
        ideal_fidelities = ideal_df['metrics'].apply(lambda x: x['global']['fidelity'])
        divergence = ideal_fidelities - noisy_fidelities
        fingerprint = {
            "total_divergence": np.sum(divergence),
            "peak_divergence_step": np.argmax(divergence),
            "divergence_per_step": divergence.tolist()
        }
        return fingerprint

    @staticmethod
    def qag_analysis(history_df: pd.DataFrame, qc: QuantumCircuit):
        num_qubits = qc.num_qubits
        depth = qc.depth()
        
        all_metrics = [
            p_metrics for metrics in history_df['metrics'] 
            for p_metrics in metrics.get('pairs', {}).values()
        ]
        
        all_concs = [m.get('concurrence', 0) for m in all_metrics]
        all_negs = [m.get('negativity', 0) for m in all_metrics]
        all_mi = [m.get('mutual_information', 0) for m in all_metrics]
        
        genome = {
            'Complexity': np.log1p(depth * num_qubits),
            'Max Concurrence': max(all_concs) if all_concs else 0,
            'Avg Negativity': np.mean(all_negs) if all_negs else 0,
            'Total Mutual Info': np.sum(all_mi) if all_mi else 0,
            'Decoherence Resilience': history_df.iloc[-1]['metrics']['global']['fidelity']
        }
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(np.array(list(genome.values())).reshape(-1, 1)).flatten()
        return dict(zip(genome.keys(), scaled_values))

class QuantumAnalysisPipeline:
    def __init__(self, qc: QuantumCircuit):
        self.qc = qc
        self.noisy_engine = SimulationEngine(noise_level=0.005)
        self.ideal_engine = SimulationEngine(noise_level=0)
        self.metric_computer = MetricComputer()

    def run(self):
        noisy_snapshots = self.noisy_engine.get_stepwise_snapshots(self.qc)
        ideal_snapshots = self.ideal_engine.get_stepwise_snapshots(self.qc)

        noisy_history = [self.metric_computer.compute_all_metrics(snap['state'], ideal_snapshots[i]['state']) for i, snap in enumerate(noisy_snapshots)]
        ideal_history = [self.metric_computer.compute_all_metrics(snap['state'], snap['state']) for snap in ideal_snapshots]

        noisy_df = pd.DataFrame({'step': range(len(noisy_history)), 'gate': [s['gate'] for s in noisy_snapshots], 'metrics': noisy_history})
        ideal_df = pd.DataFrame({'step': range(len(ideal_history)), 'gate': [s['gate'] for s in ideal_snapshots], 'metrics': ideal_history})

        qag_genome = DiscoveryModules.qag_analysis(noisy_df, self.qc)
        qdf_fingerprint = DiscoveryModules.qdf_analysis(noisy_df, ideal_df)
        
        return {
            "noisy_df": noisy_df,
            "ideal_df": ideal_df,
            "qag": qag_genome,
            "qdf": qdf_fingerprint,
            "circuit": self.qc
        }