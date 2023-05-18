from math import pi
import sys
import numpy as np
import rustworkx as rx
import argparse
from qiskit_nature.second_q.hamiltonians.lattices import *
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
import matplotlib.pyplot as plt
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import *
from time import time
#from qiskit_nature.mappers.second_quantization import LogarithmicMapper
from qiskit.algorithms import NumPyEigensolver
from qiskit import *

# Custom Heisenberg couplings
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, '../solution')
from heisenberg_model import HeisenbergModel
#from fermi_hubbard_model import FermiHubbardModel
#from custom_vqe import *
from kagome import *

# https://qiskit.org/ecosystem/nature/tutorials/10_lattice_models.html

def hamiltonian_fermihubbard(lattice, weight = -1.0):
    t = weight  # the interaction parameter
    v = 0.0  # the onsite potential
    u = 5.0  # the interaction parameter U

    # https://qiskit.org/ecosystem/nature/locale/bn_BN/stubs/qiskit_nature.second_q.hamiltonians.FermiHubbardModel.html
    """
    fhm = FermiHubbardModel.uniform_parameters(
        lattice=lattice,
        uniform_interaction=t,  # same spin-spin interaction weight as used in graph
        uniform_onsite_potential=v,  # No singe site external field
        onsite_interaction=u
    )
    """
    fhm = FermiHubbardModel(
        lattice.uniform_parameters(
            uniform_interaction=t,
            uniform_onsite_potential=v,
        ),
        onsite_interaction=u,
    )
    
    # https://qiskit.org/ecosystem/nature/tutorials/06_qubit_mappers.html
    mapper = JordanWignerMapper()

    ham = mapper.map(fhm.second_q_op().simplify())
    return ham, fhm
 
def hamiltonian_heisenberg(lattice, weight = -1.0):
    # Build Hamiltonian from graph edges
    heis = HeisenbergModel.uniform_parameters(
        lattice=lattice,
        uniform_interaction=weight,  # same spin-spin interaction weight as used in graph
        uniform_onsite_potential=0.0,  # No singe site external field
    )
    
    # The Lattice needs an explicit mapping to the qubit states.
    # We map 1 qubit for 1 spin-1/2 particle using the LogarithmicMapper
    log_mapper = LogarithmicMapper()

    # Multiply by factor of 4 to account for (1/2)^2 terms from spin operators in the HeisenbergModel
    ham = 4 * log_mapper.map(heis.second_q_ops().simplify())
    return ham, heis
 
def ground_state_fermihubbard(lattice, weight = -1.0):
    ham, fhm        = hamiltonian_fermihubbard(lattice, weight)
    lmp             = LatticeModelProblem(fhm)
    numpy_solver    = NumPyMinimumEigensolver()
    qubit_mapper    = JordanWignerMapper()

    #print (square_lattice.num_nodes)
    calc            = GroundStateEigensolver(qubit_mapper, numpy_solver)

    t0              = time()
    res             = calc.solve(lmp)
    t1              = time()

    #print("Square lattice(t=%.1f,v=%.2f, U=%.2f) rows=%d  cols=%d %s in %.3f(s)" 
    #    % (t,v,u,rows, cols,  str(res), (t1-t0)) )
    return res, (t1-t0), ham

    
def ground_state_heisenberg(lattice, weight = 1.0):
    ham, heis    = hamiltonian_heisenberg(lattice, weight)
    
    # find the first three (k=3) eigenvalues
    exact_solver = NumPyEigensolver(k=3)
    t0           = time()
    exact_result = exact_solver.compute_eigenvalues(ham)
    t1           = time()
    #print(exact_result.eigenvalues)

    ############ Compute ground state energy
    gs_energy   = np.round(exact_result.eigenvalues[0], 4)
    return gs_energy, (t1-t0), ham

def ground_state_square_lattice_fermihubbard(rows = 2, cols = 2, weight = 1.0):
    lattice      = SquareLattice(rows=rows, cols=cols) 
    return ground_state_fermihubbard(lattice, weight)
    
def ground_state_square_lattice_heisenberg(rows = 2, cols = 2, weight = 1.0):
    lattice      = SquareLattice(rows=rows, cols=cols) 
    return ground_state_heisenberg(lattice, weight)
 
def get_hamiltonian(model, lattice, weight):
    if model == 'heisenberg' :
        return hamiltonian_heisenberg(lattice, weight)
    elif model == 'fermihubbard' :
        return hamiltonian_fermihubbard(lattice, weight)
    else :
        raise Exception('Invalid Ising model' + str(model))
        
def get_args_parser() :
    parser  = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ibm-q-community/ibmquantumawards/open-science-22
    parser.add_argument('-p', '--provider',      help="Connection Provider: Hub/Group/Project"
        ,  default='ibm-q-ncsu/nc-state/grad-qc-class')
    parser.add_argument('-b', '--runbackend',           help="Run backend",  default='ibmq_guadalupe')
    parser.add_argument('-t', '--transpile_backend',    help="Transpile backend",  default='ibmq_guadalupe')

    parser.add_argument('-q', '--num_qubits',     type=int, help="Run backend # of qubits",  default=16)
    #1,2,3,5,8,11,14,13,12,10,7,4
    parser.add_argument('-ql', '--qubit_layout',  help="qubit-layout",  default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")

    parser.add_argument('-s', '--shots',            type=int, help="Shots",  default=1048)

    parser.add_argument('-a', '--ansatz_type',      help="Ansatz type"
        , default='EfficientSU2'
        , choices=['ExcitationPreserving', 'EfficientSU2','PauliTwoDesign','TwoLocal','RealAmplitudes'])
        
    parser.add_argument('-o', '--optimizer_type',   help="Optimizer type"
        , default='NFT'
        , choices=['SPSA', 'SLSQP','COBYLA','UMDA','GSLS','GradientDescent','L_BFGS_B','NELDER_MEAD','POWELL','NFT'])
    parser.add_argument('-i', '--max_iter',         help="Maximum number of iterations", type=int, default=100)

    parser.add_argument('-ol', '--opt_level',           help="Optimization level", type=int, default=1, choices=range(1, 4))
    parser.add_argument('-ui', '--uniform_interaction', help="HeisenbergModel uniform interaction", type=float)
    parser.add_argument('-up', '--uniform_potential',   help="HeisenbergModel uniform potential", type=float, default=0.0)
    parser.add_argument('-tg', '--target',          help="Target ground energy", default='None')

    parser.add_argument('-lt', '--lattice_type',   help="Lattice type", default='square'
        , choices=['square', 'triangular','line','kagome'])
    parser.add_argument('-lr', '--lattice_rows',   help="Lattice rows", type=int, default=2)
    parser.add_argument('-lc', '--lattice_cols',   help="Lattice cols", type=int, default=2)
    parser.add_argument('-lb', '--lattice_boundary', help="Lattice cols", default='open'
        , choices=['open', 'periodic'])
 
    parser.add_argument('-m', '--ising_model',   help="Ising Model", default='heisenberg'
        , choices=['heisenberg', 'fermihubbard'])
  
    parser.add_argument('-r', '--resilience_type',  help="Resilience type", default='ZNE', choices=['T-REx','ZNE','PEC'])
    parser.add_argument('-w', '--weight',           help="Edge weight", type=float, default=1.6)
    parser.add_argument('-v', '--verbosity',        help="Verbosity level", type=int, default=2, choices=range(1,5))
    return parser

def get_lattice_boundary_condition(cond):
     if cond == 'open':
        return BoundaryCondition.OPEN
     elif cond == 'periodic': 
        return BoundaryCondition.PERIODIC
     else:
        raise Exception("Invalid lattice boundary condition " + cond) 
   
# https://qiskit.org/ecosystem/nature/tutorials/10_lattice_models.html   
def get_lattice (args):
    rows    = args.lattice_rows
    cols    = args.lattice_cols
    type    = args.lattice_type
    boundary_condition = get_lattice_boundary_condition(args.lattice_boundary)
    lattice = ''
    num_nodes = rows * cols
    
    if type == 'kagome':
        qubit_layout = [eval(i) for i in args.qubit_layout.split(",")]
        lattice     = kn_get_helix (qubit_layout, qubits=args.num_qubits)
        num_nodes   = args.num_qubits
    elif type == 'square':
        lattice = SquareLattice(rows=rows, cols=cols, boundary_condition=boundary_condition)
    elif type == 'triangular':
        lattice = TriangularLattice(rows=rows, cols=cols, boundary_condition=boundary_condition)
    elif type == 'line':
        lattice = LineLattice(num_nodes=num_nodes, boundary_condition=boundary_condition)
    else:
        raise Exception("Invalid lattice type " + type) 
    return lattice, type + '-' + str(num_nodes) # 'lattice-' +
