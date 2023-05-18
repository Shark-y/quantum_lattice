from qiskit import *
from lattice_tools import *
from lattice_runner import *
from kagome import *
import argparse, math
import matplotlib.pyplot as plt

"""
Sample Usage:

== SQUARE
python runner.py -lt square -lc 3 -lb periodic -v 4 -q 6 -ql 0,1,2,3,4,5 -b simulator_statevector -t simulator_statevector -tg compute
python runner.py -lt square -lc 3 -lb periodic -v 4 -q 6 -ql 0,1,2,3,4,5 -b FakeGuadalupe -t None -tg compute
python runner.py -lt square -lc 4 -lr 4 -lb periodic -v 4 -q 16 -ql 1,2,3,5,8,11,14,13,12,10,7,4,0,9,6,15 -b ibmq_guadalupe -t ibmq_guadalupe -tg compute -i 100

== TRIANGULAR
python runner.py -lt triangular -lc 3 -lb periodic -v 4 -q 6 -ql 0,1,2,3,4,5 -b simulator_statevector -t simulator_statevector -tg compute
python runner.py -lt triangular -lc 3 -lb periodic -v 4 -q 6 -ql 0,1,2,3,4,5 -b FakeGuadalupe -t None -tg compute

== LINE
python runner.py -lt line -lc 3 -lb periodic -v 4 -q 6 -ql 0,1,2,3,4,5 -b simulator_statevector -t simulator_statevector -tg compute
python runner.py -lt line -lc 3 -lb periodic -v 4 -q 6 -ql 0,1,2,3,4,5 -b FakeGuadalupe -t None -tg compute

== KAGOME
python runner.py -lt kagome -v 4 -q 12 -ql 0,1,2,3,4,5,6,7,8,9,10,11 -b simulator_statevector -t simulator_statevector -tg compute
python runner.py -lt kagome -v 4 -q 12 -ql 0,1,2,3,4,5,6,7,8,9,10,11 -b FakeGuadalupe -t None -tg compute
python runner.py -lt kagome -v 4 -q 16 -ql 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 -b FakeGuadalupe -t FakeGuadalupe -tg compute
python runner.py -lt kagome -lc 4 -lr 3 -v 4 -q 16 -ql 0,1,2,3,4,5,6,7,8,9,10,11 -b ibmq_guadalupe -t ibmq_guadalupe -tg -18 -i 100
"""

 
parser  = get_args_parser()
"""
input   = ['-p','ibm-q-ncsu/nc-state/grad-qc-class', '-b','simulator_statevector', '-t','simulator_statevector'
    , '-i', '20', '-v', '2', '-ql', '0,1,2,3,4,5', '-q', '6', '-tg','compute'] #  
"""    
args            = parser.parse_args()
cn              = args.provider.split("/")     # Connection string hub/group/project

qubit_layout    = [eval(i) for i in args.qubit_layout.split(",")]

# build lattice
lattice, l_name = get_lattice (args) 
num_qubits      = len(qubit_layout) 

lattice.draw(style={'with_labels':True, 'font_color':'white', 'node_color':'purple'}) 
#plt.show()

plt.savefig('lattice-' + l_name + '.png')

# Expected ground energy
if args.target == 'compute':
    gs_energy, _time, _   = ground_state_heisenberg(lattice)  
    print("Computed ground state energy: %.8f in %d (s)"  % (gs_energy, _time))
elif args.target != 'None' :
    gs_energy   = float(args.target)
    print("Using ground state energy: %.2f" % gs_energy)
else:
    gs_energy   = None

params  = { 
    'hub'                   : cn[0],    # args.hub,
    'group'                 : cn[1],    # args.group, 
    'project'               : cn[2],    # args.project,
    'provider'              : args.provider,
    'transpile_backend'     : args.transpile_backend,
    'run_backend'           : args.runbackend,
    'num_qubits'            : args.num_qubits,
    'qubit_layout'          : qubit_layout,
    'shots'                 : args.shots,
    'opt_level'             : args.opt_level,
    'uniform_interaction'   : args.uniform_interaction if args.uniform_interaction != None else args.weight,
    'uniform_potential'     : args.uniform_potential,
    'verbosity'             : args.verbosity,
    'ising_model'           : args.ising_model
}

#qubits=args.num_qubits
runner = VQELatticeRunner(qubits=len(qubit_layout), weight=args.weight, ansatz_type=args.ansatz_type
                , optimizer_type=args.optimizer_type
                , optimizer_maxiter=args.max_iter
                , resilience_type=args.resilience_type)

runner.expected_energy = gs_energy

computed_gse, intermediate_info_real_backend = runner.run(lattice, params)

def rel_err(target, measured):
    return abs((target - measured) / target)

err_prob    = 100 * rel_err(gs_energy, computed_gse) if gs_energy != None else -1
    
print(f'Computed ground state energy: {computed_gse:.8f}')
if gs_energy != None :
    print(f'Expected ground state energy: {gs_energy:.8f}') 
    print(f'Relative error: {100 * rel_err(gs_energy, computed_gse):.8f} %') 

# Let's plot the energy convergence data the callback function acquired.
backend         = args.runbackend
ans_name        = args.ansatz_type
optimizer_name  = args.optimizer_type
resilience      = args.resilience_type
shots           = args.shots
t               = args.weight
plot_name       = "plot-%s-%s-%s-%s-s(%d)-w(%.4f)-e%.2f.png" % (backend, ans_name, optimizer_name, resilience, shots, t, err_prob)
plt.figure(figsize=(10, 5))
plt.plot(intermediate_info_real_backend, color='purple', lw=2, label='VQE: ' + str(computed_gse))
plt.ylabel('Energy')
plt.xlabel('Iterations')
plt.title(backend + "/" + ans_name + "/" + optimizer_name + "/" + resilience 
        + "/" + str(shots) + " weight: %.4f Error: %.2f%% Lattice: %s" % (t, err_prob, l_name)) 
# Exact ground state energy value
if gs_energy != None:
    plt.axhline(y=gs_energy, color="tab:red", ls="--", lw=2, label="Target: " + str(gs_energy))
plt.legend()
plt.grid()
plt.savefig(plot_name)
