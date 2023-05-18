import pprint
import numpy as np
import retworkx as rx
from qiskit.algorithms import MinimumEigensolver, VQEResult
from qiskit_nature.problems.second_quantization.lattice import Lattice
from heisenberg_model import HeisenbergModel
from qiskit_nature.mappers.second_quantization import LogarithmicMapper
from lattice_tools import *

# Define a custome VQE class to orchestra the ansatz, classical optimizers, 
# initial point, callback, and final result
class CustomVQE(MinimumEigensolver):
    task_done           = False
    last_value			= 0
    _expected_energy    = 0     # expected ground energy
    _threshold     		= 1     # def error threshold [1-100] (if reached skip quantum)
    _weight             = 1.0    # default weight
    _num_qubits         = 16
    _q_layout           = [1, 2, 3, 5, 8, 11, 14, 13, 12, 10, 7, 4]
    _ham16              = None
    _fix_ham16          = False
    _model              = None  # Ising model (heisenberg, fermihubbard)
    _recursions         = 0
    _initial_point      = None
    res                 = None
    
    def __init__(self, estimator, circuit, optimizer, callback=None):
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
       
    def rel_err(self, target, measured):
        return abs((target - measured) / target)
    
    def set_expected_values(self, energy, threshold = 1, weight = 1.0, num_qubits = 16, q_layout = [], model = None) :
        self._expected_energy 	= energy
        self._threshold 		= threshold
        self._weight            = weight
        self._num_qubits        = num_qubits
        self._q_layout          = q_layout
        self._model             = model
        
    def compute_minimum_eigenvalue(self, operators, aux_operators=None, lattice=None):
        self._ham16 = operators
        
        # Define objective function to classically minimize over
        def objective(x):
            if self.task_done :
                return self.last_value
		
            # Execute job with estimator primitive
            job = self._estimator.run([self._circuit], [self._ham16], [x])
            
            if self._fix_ham16:
                self._ham16,_       = get_hamiltonian(self._model, lattice, self._weight) 
                #self._ham16, _     = hamiltonian_heisenberg(lattice, self._weight)
                self._fix_ham16     = False
                
            # Get results from jobs
            # EstimatorResult(values=array([-12.13125]), metadata=[{'variance': 55.10802734375001, 'shots': 1024}])
            est_result = job.result()
            
            # Get the measured energy value
            self.last_value = value = est_result.values[0]

            # if we go below the target, add a bias to the Hamiltonian
            if self._expected_energy != None :
                if ( value < self._expected_energy ) :
                    delta = np.abs(value) - np.abs(self._expected_energy)
                    if ( delta > 10 ) :
                        if np.abs(self._weight) > 1.0 :
                            self._weight    -= 0.25
                            self._fix_ham16 = True
                    
                # If we reach the error threshold, skip
                if self._expected_energy != 0 :
                    err     = 100 * self.rel_err(self._expected_energy, value)
                    if err <= self._threshold :
                        self.task_done = True
			
            # Save result information using callback function
            if self._callback is not None:
                self._callback(value)
				
            return value
            
        # Select an initial point for the ansatzs' parameters
        if self._initial_point is None :
            x0 = np.pi/4 * np.random.rand(self._circuit.num_parameters)
        else :
            x0 = self._initial_point
        
        # Run optimization
        self.res    = self._optimizer.minimize(objective, x0=x0)
  
        if self._expected_energy != None :
            delta       = np.abs(self.res.fun) - np.abs(self._expected_energy)
            err         = 100 * self.rel_err(self._expected_energy, self.res.fun)
            
            # If ended above the target energy, try again w/ a higher weight
            if err > self._threshold and self._recursions < 5 and (not self.task_done) :
                self._recursions    += 1
                if delta < 0 :
                    self._weight    += np.abs(delta)/100 + 0.3
                else :
                    self._weight    -= np.abs(delta)/100 + 0.3
                    
                self._ham16, _       = get_hamiltonian(self._model, lattice, self._weight) 
                #self._ham16, _      = hamiltonian_heisenberg(lattice, self._weight)
                self._initial_point = self.res.x 
                #print (f"Final value: {self.res.fun} Expected: {self._expected_energy} delta {delta} new weight: {self._weight}")
                self.compute_minimum_eigenvalue(self._ham16, lattice = lattice)
        
        # Populate VQE result
        result                      = VQEResult()
        result.cost_function_evals  = self.res.nfev
        result.eigenvalue           = self.res.fun
        result.optimal_parameters   = self.res.x
        return result 
