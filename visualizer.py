# visualizer.py
import plotly.graph_objects as go
import numpy as np
import networkx as nx

class Visualizer:

    def plot_qecart_matrix(self, correlation_matrix, step):
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=[f'Q{i}' for i in range(correlation_matrix.shape[0])],
            y=[f'Q{i}' for i in range(correlation_matrix.shape[0])],
            colorscale='Reds', zmin=0, zmax=1,
            colorbar=dict(title='Correlation Score')
        ))
        fig.update_layout(title=f'QECart: Qubit Correlation Matrix (Step {step})')
        return fig

    def plot_qecg_graph(self, qecg_graph):
        if not isinstance(qecg_graph, nx.DiGraph):
            fig = go.Figure()
            fig.update_layout(title=qecg_graph.get("error", "QECG Analysis Failed"),
                              xaxis={'visible': False}, yaxis={'visible': False},
                              annotations=[{"text": "Circuit too short for error simulation.", "showarrow": False}])
            return fig
            
        pos = nx.spring_layout(qecg_graph, seed=42)
        
        edge_x, edge_y = [], []
        for edge in qecg_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
        
        node_x, node_y, node_text, node_color = [], [], [], []
        for node, data in qecg_graph.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(data.get('label', ''))
            node_color.append(data.get('color', 'blue'))
            
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text,
                                marker=dict(size=15, color=node_color))
                                
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='QECG: Error Causality Graph', showlegend=False,
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        return fig
    
    def plot_qip(self, qip_matrix, gates):
        fig = go.Figure(data=go.Heatmap(
            z=qip_matrix, x=gates, y=[f'Qubit {i}' for i in range(qip_matrix.shape[0])],
            colorscale='RdBu', zmid=0, colorbar=dict(title='Entropy Change')))
        fig.update_layout(title="QIP: Quantum Information Pressure", xaxis_title="Gate", yaxis_title="Qubit")
        return fig

    def plot_qdf(self, qdf_data):
        fig = go.Figure(data=[go.Bar(x=list(range(len(qdf_data['divergence_per_step']))), y=qdf_data['divergence_per_step'])])
        fig.update_layout(title="QDF: Decoherence Fingerprint",
                          xaxis_title="Circuit Step", yaxis_title="Fidelity Divergence")
        return fig

    def plot_qag(self, qag_data):
        categories = list(qag_data.keys())
        values = list(qag_data.values())
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="QAG: Quantum Algorithm Genome")
        return fig
    def plot_hardware_comparison(self, ideal_counts, noisy_counts, hardware_counts):
        
        all_states = sorted(list(set(ideal_counts.keys()) | set(noisy_counts.keys()) | set(hardware_counts.keys())))
        
        ideal_probs = [ideal_counts.get(s, 0) for s in all_states]
        noisy_probs = [noisy_counts.get(s, 0) for s in all_states]
        hw_probs = [hardware_counts.get(s, 0) for s in all_states]
        
        fig = go.Figure(data=[
            go.Bar(name='Ideal Simulation', x=all_states, y=ideal_probs),
            go.Bar(name='Noisy Simulation', x=all_states, y=noisy_probs),
            go.Bar(name='IBM Hardware', x=all_states, y=hw_probs)
        ])
        
        fig.update_layout(
            barmode='group',
            title='Hardware vs. Simulation Comparison',
            xaxis_title='Measurement Outcome',
            yaxis_title='Counts'
        )
        return fig