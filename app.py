import streamlit as st
from qiskit import QuantumCircuit, transpile
import pandas as pd
import numpy as np
# This version removes 'negativity' to be compatible with your Qiskit version
from qiskit.quantum_info import DensityMatrix, partial_trace, concurrence, state_fidelity, entropy as qiskit_entropy, mutual_information
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import umap.umap_ as umap
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os
import io
from PIL import Image

# Conditional import for IBM Provider
try:
    from qiskit_ibm_provider import IBMProvider
except ImportError: IBMProvider = None

#==============================================================================
# MODULE 1: QUANTUM ENGINE & METRICS
#==============================================================================
class SimulationEngine:
    """Handles the stepwise simulation of quantum circuits with optional noise."""
    def __init__(self, noise_level=0.001):
        self.simulator = AerSimulator()
        self.noise_model = self._create_noise_model(noise_level)
    def _create_noise_model(self, noise_level):
        if noise_level == 0: return None
        error_1, error_2 = depolarizing_error(noise_level, 1), depolarizing_error(noise_level, 2)
        noise_model = NoiseModel()
        one_qubit_gates, two_qubit_gates = ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z', 'id', 'rz', 'sx'], ['cx', 'swap', 'cu1', 'cz']
        noise_model.add_all_qubit_quantum_error(error_1, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2, two_qubit_gates)
        return noise_model
    def get_stepwise_snapshots(self, qc: QuantumCircuit):
        snapshots, cumulative_qc = [], QuantumCircuit(qc.num_qubits, qc.num_clbits)
        initial_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits); initial_qc.save_density_matrix()
        result = self.simulator.run(initial_qc, noise_model=self.noise_model).result()
        snapshots.append({"step": 0, "gate": "Initial State", "state": result.data()['density_matrix']})
        for i, instruction in enumerate(qc.data):
            cumulative_qc.append(instruction); snapshot_qc = cumulative_qc.copy(); snapshot_qc.save_density_matrix()
            result = self.simulator.run(snapshot_qc, noise_model=self.noise_model).result()
            gate_info = f"{instruction.operation.name.upper()} q{', '.join([str(qc.qubits.index(q)) for q in instruction.qubits])}"
            snapshots.append({"step": i + 1, "gate": gate_info, "state": result.data()['density_matrix']})
        return snapshots
    def get_final_counts(self, qc: QuantumCircuit, shots=1024):
        if not qc.clbits: qc.measure_all()
        transpiled_qc = transpile(qc, self.simulator)
        result = self.simulator.run(transpiled_qc, shots=shots, noise_model=self.noise_model).result()
        return result.get_counts(qc)

class MetricComputer:
    """Computes an extensive set of quantum metrics for deep analysis."""
    def compute_all_metrics(self, state: DensityMatrix, ideal_state: DensityMatrix):
        num_qubits = state.num_qubits; metrics = {"qubits": [], "pairs": {}, "global": {}}
        metrics["global"]["fidelity"] = state_fidelity(ideal_state, state)
        for i in range(num_qubits):
            rho_i = partial_trace(state, [q for q in range(num_qubits) if q != i])
            metrics["qubits"].append({"entropy": qiskit_entropy(rho_i), "density_matrix": rho_i.data})
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                pair_key, qubit_list = f"{i}-{j}", [i, j]
                try:
                    rho_pair = partial_trace(state, [q for q in range(num_qubits) if q not in qubit_list])
                    conc, mut_info = concurrence(rho_pair), mutual_information(state, qubit_list)
                except Exception:
                    conc, mut_info = 0.0, 0.0
                metrics["pairs"][pair_key] = {"concurrence": conc, "negativity": 0.0, "mutual_information": mut_info}
        return metrics

#==============================================================================
# MODULE 2: HARDWARE INTEGRATION (with Dummy Mode)
#==============================================================================
class HardwareManager:
    """Handles connection and job execution on both real and dummy hardware."""
    def __init__(self, api_key=None):
        self.provider = None;
        if IBMProvider is None: return
        try:
            if api_key: IBMProvider.save_account(token=api_key, overwrite=True)
            if IBMProvider.saved_accounts() or api_key or os.getenv('IBM_QUANTUM_TOKEN'): self.provider = IBMProvider()
        except Exception: pass

    def is_available(self):
        return self.provider is not None

    def get_available_backends(self):
        backends = ["IBM QISKIT (Mock Simulator)"]
        if self.is_available():
            try:
                real_backends = sorted([b.name for b in self.provider.backends(simulator=False, operational=True) if b.max_circuits > 0])
                backends.extend(real_backends)
            except Exception: pass
        return backends
    
    def run_on_dummy_hardware(self, noisy_counts: dict):
        dummy_counts = noisy_counts.copy()
        total_shots = sum(dummy_counts.values())
        if not dummy_counts: return {}

        shots_to_steal = int(total_shots * 0.1)
        stolen_shots = 0
        for state, count in sorted(dummy_counts.items(), key=lambda item: item[1], reverse=True):
            if stolen_shots >= shots_to_steal: break
            take = min(count, shots_to_steal - stolen_shots)
            dummy_counts[state] -= take
            stolen_shots += take

        num_qubits = len(list(dummy_counts.keys())[0])
        possible_outcomes = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]
        
        for _ in range(stolen_shots):
            random_state = np.random.choice(possible_outcomes)
            dummy_counts[random_state] = dummy_counts.get(random_state, 0) + 1
            
        return dummy_counts

    def run_on_hardware(self, qc: QuantumCircuit, backend_name: str, shots=1024):
        if not self.is_available(): return {"error": "IBM Provider not initialized."}
        try:
            backend = self.provider.get_backend(backend_name)
            if not qc.clbits: qc.measure_all()
            transpiled_qc = transpile(qc, backend)
            job = backend.run(transpiled_qc, shots=shots)
            return job.result().get_counts(qc)
        except Exception as e:
            return {"error": f"Job failed: {str(e)}"}

#==============================================================================
# MODULE 3: DISCOVERY & ANALYSIS
#==============================================================================
class DiscoveryModules:
    @staticmethod
    def qecart_analysis(step_data: pd.Series):
        num_qubits = len(step_data['metrics']['qubits']); correlation_matrix = np.identity(num_qubits)
        for pair, data in step_data['metrics']['pairs'].items():
            q1, q2 = map(int, pair.split('-')); 
            score = (data.get('concurrence', 0) + data.get('mutual_information', 0)) / 2
            correlation_matrix[q1, q2] = correlation_matrix[q2, q1] = score
        return correlation_matrix
    @staticmethod
    def qecg_analysis(qc: QuantumCircuit, history_df: pd.DataFrame, error_qubit=0, error_gate_idx=1):
        if len(qc.data) <= error_gate_idx: return {"error": "Circuit too short."}
        baseline_fidelities = history_df['metrics'].apply(lambda x: x['global']['fidelity']).tolist(); causal_graph = nx.DiGraph()
        causal_graph.add_node(error_gate_idx, label=f"Error on Q{error_qubit}", color='red')
        for i in range(error_gate_idx + 1, len(history_df)):
            prev_fidelity, curr_fidelity = baseline_fidelities[i-1], baseline_fidelities[i]; impact = max(0, prev_fidelity - curr_fidelity)
            if impact > 0.01:
                causal_graph.add_node(i, label=history_df.iloc[i]['gate'], color='orange'); causal_graph.add_edge(i-1, i, weight=impact)
        return causal_graph
    @staticmethod
    def qip_analysis(history_df: pd.DataFrame):
        num_steps, num_qubits = len(history_df), len(history_df.iloc[0]['metrics']['qubits']); pressure_matrix = np.zeros((num_qubits, num_steps))
        for step in range(1, num_steps):
            prev_entropies = np.array([q['entropy'] for q in history_df.iloc[step-1]['metrics']['qubits']])
            curr_entropies = np.array([q['entropy'] for q in history_df.iloc[step]['metrics']['qubits']])
            pressure_matrix[:, step] = curr_entropies - prev_entropies
        return pressure_matrix
    @staticmethod
    def qdf_analysis(noisy_df: pd.DataFrame, ideal_df: pd.DataFrame):
        noisy_fidelities = noisy_df['metrics'].apply(lambda x: x['global']['fidelity']); ideal_fidelities = ideal_df['metrics'].apply(lambda x: x['global']['fidelity'])
        return { "divergence_per_step": (ideal_fidelities - noisy_fidelities).tolist() }
    @staticmethod
    def qag_analysis(history_df: pd.DataFrame, qc: QuantumCircuit):
        num_qubits, depth = qc.num_qubits, qc.depth(); all_metrics = [p for m in history_df['metrics'] for p in m.get('pairs', {}).values()]
        all_concs = [m.get('concurrence', 0) for m in all_metrics]; 
        all_mi = [m.get('mutual_information', 0) for m in all_metrics]
        genome = {'Complexity': np.log1p(depth * num_qubits), 'Max Concurrence': max(all_concs) if all_concs else 0,
                  'Total Mutual Info': np.sum(all_mi) if all_mi else 0,
                  'Decoherence Resilience': history_df.iloc[-1]['metrics']['global']['fidelity']}
        scaler = MinMaxScaler(); scaled_values = scaler.fit_transform(np.array(list(genome.values())).reshape(-1, 1)).flatten()
        return dict(zip(genome.keys(), scaled_values))
    @staticmethod
    def ai_recommendations_engine(noisy_df: pd.DataFrame, qdf_data, qip_matrix, qag_data):
        recs = []
        peak_divergence_step = np.argmax(qdf_data['divergence_per_step'])
        if qdf_data['divergence_per_step'][peak_divergence_step] > 0.05:
            gate_at_peak = noisy_df.iloc[peak_divergence_step]['gate']
            recs.append(f"**High Decoherence Source:** The largest drop in fidelity occurs at **Step {peak_divergence_step} (`{gate_at_peak}`)**. Consider using error mitigation techniques.")
        avg_pressure_per_qubit = np.mean(np.abs(qip_matrix), axis=1)
        fragile_qubit = np.argmax(avg_pressure_per_qubit)
        if avg_pressure_per_qubit[fragile_qubit] > 0.01:
            recs.append(f"**Fragile Qubit Warning:** **Qubit {fragile_qubit}** experiences the highest information pressure. When mapping to hardware, assign this to a physical qubit with low error rates.")
        if qag_data['Decoherence Resilience'] < 0.8:
            recs.append(f"**Low Resilience Genome:** The algorithm shows low overall resilience to noise (Final Fidelity: {qag_data['Decoherence Resilience']:.2f}). Consider adding error correction.")
        if not recs: recs.append("The circuit appears robust with no major points of failure detected.")
        return recs

#==============================================================================
# MODULE 4: VISUALIZER
#==============================================================================
class Visualizer:
    def _create_bloch_sphere(self):
        return go.Figure(layout=go.Layout(scene=dict(xaxis=dict(title='X', range=[-1, 1]), yaxis=dict(title='Y', range=[-1, 1]), zaxis=dict(title='Z', range=[-1, 1]), aspectratio=dict(x=1, y=1, z=1)), margin=dict(l=0, r=0, b=0, t=0)))
    def plot_decoherence_and_measurement(self, density_matrix):
        trace_x = np.real(np.trace(density_matrix @ np.array([[0, 1], [1, 0]]))); trace_y = np.real(np.trace(density_matrix @ np.array([[0, -1j], [1j, 0]]))); trace_z = np.real(np.trace(density_matrix @ np.array([[1, 0], [0, -1]])))
        prob_0, prob_1 = np.real(density_matrix[0, 0]), np.real(density_matrix[1, 1])
        fig = self._create_bloch_sphere(); fig.add_trace(go.Scatter3d(x=[0, trace_x], y=[0, trace_y], z=[0, trace_z], mode='lines', line=dict(color='red', width=5), name='State Vector')); fig.update_layout(title_text="Single Qubit State")
        prob_fig = go.Figure(data=[go.Bar(x=['|0⟩', '|1⟩'], y=[prob_0, prob_1])]); prob_fig.update_layout(title_text="Pre-Measurement Probabilities", yaxis_range=[0,1])
        return fig, prob_fig
    def plot_3d_qubit_state(self, history_df, embedding, step):
        num_qubits, step_embedding, step_metrics = embedding.shape[1], embedding[step], history_df.iloc[step]['metrics']
        qubit_trace = go.Scatter3d(x=step_embedding[:, 0], y=step_embedding[:, 1], z=step_embedding[:, 2], mode='markers+text', text=[f"Q{i}" for i in range(num_qubits)], marker=dict(size=10, color=[q['entropy'] for q in step_metrics['qubits']], colorscale='Viridis', colorbar=dict(title='Entropy'), showscale=True))
        layout = go.Layout(title=f"3D Qubit Constellation", scene=dict(xaxis=dict(title='UMAP 1'), yaxis=dict(title='UMAP 2'), zaxis=dict(title='UMAP 3')), margin=dict(l=0, r=0, b=0, t=40))
        return go.Figure(data=[qubit_trace], layout=layout)
    def plot_qecart_matrix(self, correlation_matrix, step):
        fig = go.Figure(data=go.Heatmap(z=correlation_matrix, x=[f'Q{i}' for i in range(correlation_matrix.shape[0])], y=[f'Q{i}' for i in range(correlation_matrix.shape[0])], colorscale='Reds', zmin=0, zmax=1, colorbar=dict(title='Correlation')))
        fig.update_layout(title=f'QECart: Qubit Correlation Matrix'); return fig
    def plot_qecg_graph(self, qecg_graph):
        if not isinstance(qecg_graph, nx.DiGraph): return go.Figure().update_layout(title=qecg_graph.get("error", "QECG Failed"))
        pos = nx.spring_layout(qecg_graph, seed=42); edge_trace = go.Scatter(x=[], y=[], line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
        for edge in qecg_graph.edges(): x0, y0, x1, y1 = *pos[edge[0]], *pos[edge[1]]; edge_trace['x'] += (x0, x1, None); edge_trace['y'] += (y0, y1, None)
        node_trace = go.Scatter(x=[], y=[], mode='markers+text', text=[], marker=dict(size=15, color=[]))
        for node, data in qecg_graph.nodes(data=True): x, y = pos[node]; node_trace['x'] += (x,); node_trace['y'] += (y,); node_trace['text'] += (data.get('label', ''),); node_trace['marker']['color'] += (data.get('color', 'blue'),)
        return go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='QECG: Error Causality Graph', showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    def plot_qip(self, qip_matrix, gates):
        fig = go.Figure(data=go.Heatmap(z=qip_matrix, x=gates, y=[f'Qubit {i}' for i in range(qip_matrix.shape[0])], colorscale='RdBu', zmid=0, colorbar=dict(title='Entropy Change')))
        fig.update_layout(title="QIP: Quantum Information Pressure", xaxis_title="Gate", yaxis_title="Qubit"); return fig
    def plot_qdf(self, qdf_data):
        fig = go.Figure(data=[go.Bar(x=list(range(len(qdf_data['divergence_per_step']))), y=qdf_data['divergence_per_step'])])
        fig.update_layout(title="QDF: Decoherence Fingerprint", xaxis_title="Circuit Step", yaxis_title="Fidelity Divergence"); return fig
    def plot_qag(self, qag_data):
        categories, values = list(qag_data.keys()), list(qag_data.values()); fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="QAG: Quantum Algorithm Genome"); return fig
    def plot_hardware_comparison(self, ideal_counts, noisy_counts, hardware_counts):
        all_states = sorted(list(set(ideal_counts.keys()) | set(noisy_counts.keys()) | set(hardware_counts.keys()))); total_ideal, total_noisy, total_hw = sum(ideal_counts.values()), sum(noisy_counts.values()), sum(hardware_counts.values())
        ideal_probs = [ideal_counts.get(s, 0) / total_ideal for s in all_states]; noisy_probs = [noisy_counts.get(s, 0) / total_noisy for s in all_states]; hw_probs = [hardware_counts.get(s, 0) / total_hw for s in all_states]
        fig = go.Figure(data=[go.Bar(name='Ideal Sim', x=all_states, y=ideal_probs), go.Bar(name='Noisy Sim', x=all_states, y=noisy_probs), go.Bar(name='Hardware/Mock', x=all_states, y=hw_probs, marker_color='green')])
        fig.update_layout(barmode='group', title='Hardware vs. Simulation Comparison', xaxis_title='Measurement Outcome', yaxis_title='Probability'); return fig

#==============================================================================
# MODULE 5: MAIN PIPELINE & APP UI
#==============================================================================
class QuantumAnalysisPipeline:
    def __init__(self, qc: QuantumCircuit):
        self.qc = qc; self.noisy_engine = SimulationEngine(noise_level=0.005); self.ideal_engine = SimulationEngine(noise_level=0); self.metric_computer = MetricComputer()
    def run_simulation(self):
        noisy_snapshots, ideal_snapshots = self.noisy_engine.get_stepwise_snapshots(self.qc), self.ideal_engine.get_stepwise_snapshots(self.qc)
        noisy_history = [self.metric_computer.compute_all_metrics(snap['state'], ideal_snapshots[i]['state']) for i, snap in enumerate(noisy_snapshots)]
        ideal_history = [self.metric_computer.compute_all_metrics(snap['state'], snap['state']) for snap in ideal_snapshots]
        noisy_df, ideal_df = pd.DataFrame({'step': range(len(noisy_history)), 'gate': [s['gate'] for s in noisy_snapshots], 'metrics': noisy_history}), pd.DataFrame({'step': range(len(ideal_history)), 'gate': [s['gate'] for s in ideal_snapshots], 'metrics': ideal_history})
        embedding = self.get_embedding(noisy_df)
        return {"noisy_df": noisy_df, "ideal_df": ideal_df, "circuit": self.qc,
                "noisy_counts": self.noisy_engine.get_final_counts(self.qc.copy()),
                "ideal_counts": self.ideal_engine.get_final_counts(self.qc.copy()), "embedding": embedding}
    def get_embedding(self, history_df: pd.DataFrame):
        num_steps, num_qubits = len(history_df), len(history_df.iloc[0]['metrics']['qubits']); feature_vectors = []
        for step in range(num_steps):
            metrics = history_df.iloc[step]['metrics']
            for q in range(num_qubits):
                entropy = metrics['qubits'][q]['entropy']; related_concurrence = np.mean([p.get('concurrence', 0) for k, p in metrics.get('pairs', {}).items() if str(q) in k.split('-')] or [0])
                feature_vectors.append([step, q, entropy, related_concurrence])
        feature_vectors = np.array(feature_vectors); n_neighbors = min(15, len(feature_vectors) - 1)
        if n_neighbors < 2: 
            st.warning("Circuit is too small for complex UMAP embedding. Displaying a simplified layout.", icon="ℹ️")
            embedding = np.zeros((num_steps, num_qubits, 3))
            for i in range(num_qubits): embedding[:, i, 0] = i * 2 - (num_qubits - 1)
            return embedding
        reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1); embedding = reducer.fit_transform(feature_vectors[:, 2:])
        return embedding.reshape(num_steps, num_qubits, 3)

visualizer = Visualizer()

# --- Logo Integration ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("quantum_nexus_logo.png", use_container_width=True) # Ensure this path is correct
st.markdown("---") # Add a separator below the logo

with st.sidebar:
    st.header("1. Circuit Input")
    qasm_input = st.text_area("QASM 2.0 Circuit", 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];', height=200)
    if st.button("Run Simulation Analysis", use_container_width=True, type="primary"):
        st.session_state.clear()
        try:
            qc = QuantumCircuit.from_qasm_str(qasm_input)
            with st.spinner("Executing Simulation Pipeline..."): st.session_state.analysis_results = QuantumAnalysisPipeline(qc).run_simulation()
            st.success("Analysis Complete!")
        except Exception as e: st.error(f"Error: {e}")
    st.markdown("---")
    st.header("2. Hardware Execution")
    api_key_input = st.text_input("Enter IBM Quantum API Key (optional)", type="password")
    
    if 'analysis_results' in st.session_state:
        hw_manager = HardwareManager(api_key=api_key_input if api_key_input else None)
        available_backends = hw_manager.get_available_backends()
        
        selected_backend = st.selectbox("Select a Backend:", available_backends)
        
        if st.button("Run on Backend", use_container_width=True):
            with st.spinner(f"Sending job to `{selected_backend}`..."):
                qc_to_run = st.session_state.analysis_results['circuit']
                if selected_backend == "IBM QISKIT (Mock Simulator)":
                    noisy_counts = st.session_state.analysis_results['noisy_counts']
                    hw_counts = hw_manager.run_on_dummy_hardware(noisy_counts)
                else:
                    hw_counts = hw_manager.run_on_hardware(qc_to_run, selected_backend)
                
                st.session_state.hardware_results = hw_counts
            if "error" in hw_counts: st.error(f"Job Failed: {hw_counts['error']}")
            else: st.success("Job Complete!")

# --- Main Dashboard (Single Page Layout) ---
if 'analysis_results' in st.session_state:
    results, noisy_df, ideal_df, qc, embedding = st.session_state.analysis_results, st.session_state.analysis_results['noisy_df'], st.session_state.analysis_results['ideal_df'], st.session_state.analysis_results['circuit'], st.session_state.analysis_results['embedding']
    
    st.markdown("---")
    st.header("💡 Strategic Recommendations")
    qip_matrix = DiscoveryModules.qip_analysis(noisy_df); qdf_data = DiscoveryModules.qdf_analysis(noisy_df, ideal_df)
    recommendations = DiscoveryModules.ai_recommendations_engine(noisy_df, qdf_data, qip_matrix, DiscoveryModules.qag_analysis(noisy_df, qc))
    for i, rec in enumerate(recommendations): st.info(f"**Insight {i+1}:** {rec}")
    
    st.markdown("---")
    st.header("📉 Timeline Explorer")
    selected_step = st.slider("Select Circuit Step:", 0, len(noisy_df) - 1, len(noisy_df) - 1)
    st.info(f"**Displaying details for Step {selected_step}: `{noisy_df.iloc[selected_step]['gate']}`**")
    st.plotly_chart(visualizer.plot_3d_qubit_state(noisy_df, embedding, selected_step), use_container_width=True)

    st.markdown("---")
    st.header("🔬 Decoherence & Measurement Explorer")
    qubit_to_inspect = st.selectbox("Select a qubit to inspect:", range(qc.num_qubits))
    selected_qubit_dm = noisy_df.iloc[selected_step]['metrics']['qubits'][qubit_to_inspect]['density_matrix']
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("State Visualization (Before Measurement)")
        bloch_fig, prob_fig = visualizer.plot_decoherence_and_measurement(selected_qubit_dm)
        st.plotly_chart(bloch_fig, use_container_width=True)
    with col2:
        st.subheader("Simulated Measurement")
        st.plotly_chart(prob_fig, use_container_width=True)
        if st.button(f"Simulate Measurement of Qubit {qubit_to_inspect}", use_container_width=True):
            prob_0 = np.real(selected_qubit_dm[0, 0]); outcome = 0 if np.random.rand() < prob_0 else 1
            st.success(f"**Collapse!** Qubit {qubit_to_inspect} measured as **|{outcome}⟩**.")
            collapsed_dm = np.zeros((2,2), dtype=complex); collapsed_dm[outcome, outcome] = 1
            collapsed_bloch, _ = visualizer.plot_decoherence_and_measurement(collapsed_dm)
            st.plotly_chart(collapsed_bloch, use_container_width=True)

    st.markdown("---")
    st.header("📊 Discovery Modules Deep Dive")
    qecart_col, qecg_col = st.columns(2)
    with qecart_col: 
        st.subheader("🌐 QECart: Qubit Correlation Matrix")
        st.plotly_chart(visualizer.plot_qecart_matrix(DiscoveryModules.qecart_analysis(noisy_df.iloc[selected_step]), selected_step), use_container_width=True)
    with qecg_col: 
        st.subheader("📉 QECG: Error Causality Graph")
        st.plotly_chart(visualizer.plot_qecg_graph(DiscoveryModules.qecg_analysis(qc, noisy_df)), use_container_width=True)
    
    qip_col, qdf_col, qag_col = st.columns(3)
    with qip_col: 
        st.subheader("🌡️ QIP: Information Pressure")
        st.plotly_chart(visualizer.plot_qip(DiscoveryModules.qip_analysis(noisy_df), noisy_df['gate'].tolist()), use_container_width=True)
    with qdf_col: 
        st.subheader("👣 QDF: Decoherence Fingerprint")
        st.plotly_chart(visualizer.plot_qdf(DiscoveryModules.qdf_analysis(noisy_df, ideal_df)), use_container_width=True)
    with qag_col: 
        st.subheader("🧬 QAG: Algorithm Genome")
        st.plotly_chart(visualizer.plot_qag(DiscoveryModules.qag_analysis(noisy_df, qc)), use_container_width=True)

    st.markdown("---")
    st.header("⚙️ Hardware vs. Simulation Comparison")
    if 'hardware_results' in st.session_state and st.session_state.hardware_results and "error" not in st.session_state.hardware_results:
        st.plotly_chart(visualizer.plot_hardware_comparison(results['ideal_counts'], results['noisy_counts'], st.session_state.hardware_results), use_container_width=True)
    else: 
        st.info("Run a job on a backend from the sidebar to see comparison results here.")

else:
    st.info("Provide a QASM circuit and click 'Run Simulation Analysis' to begin.")