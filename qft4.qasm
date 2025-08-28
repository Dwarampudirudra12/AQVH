OPENQASM 2.0;
include "qelib1.inc";

// 4-Qubit Variational Quantum Ansatz Circuit
// This circuit creates complex entanglement patterns using layered gates,
// common in quantum machine learning and VQE algorithms.
qreg q[4];

// --- Repetition 1 ---

// Rotation layer 1
ry(1.91) q[0];
ry(2.95) q[1];
ry(4.33) q[2];
ry(0.78) q[3];

// Entanglement layer 1 (linear entanglement)
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];


// --- Repetition 2 ---

// Rotation layer 2
ry(3.34) q[0];
ry(5.02) q[1];
ry(4.08) q[2];
ry(5.55) q[3];

// Entanglement layer 2
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];


// --- Repetition 3 ---

// Rotation layer 3
ry(0.15) q[0];
ry(2.59) q[1];
ry(1.12) q[2];
ry(3.8) q[3];

// Entanglement layer 3
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[0]; // Add a circular entanglement