# hardware_integration.py
import os
from qiskit import QuantumCircuit
from qiskit_serverless import QiskitServerless
import logging

logging.basicConfig(level=logging.WARNING)

class ServerlessManager:
    def __init__(self, api_key=None):
        self.serverless = None
        token = api_key or os.getenv("QISKIT_IBM_TOKEN")
        
        if token:
            try:
                self.serverless = QiskitServerless(token=token)
            except Exception as e:
                print(f"Failed to initialize QiskitServerless: {e}")
                self.serverless = None
        
    def is_available(self):
        return self.serverless is not None

    def run_with_serverless(self, qc: QuantumCircuit, shots=1024):
        if not self.is_available():
            return {"error": "Qiskit Serverless not initialized. Check API key."}
        
        try:
            if not qc.clbits:
                qc.measure_all()
                
            job = self.serverless.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts(qc)
            return counts
        except Exception as e:
            return {"error": f"Serverless job failed: {str(e)}"}