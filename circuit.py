from qiskit import QuantumCircuit, transpile, Aer, execute
import qiskit_aer.noise as noise
import numpy as np


def make_chsh_circuit(theta_vec):
    chsh_circuits = []
    for theta in theta_vec:
        obs_vec = ['00', '01', '10', '11']
        for el in obs_vec:
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.ry(theta, 0)
            for a in range(2):
                if el[a] == '1':
                    qc.h(a)
            qc.measure(range(2), range(2))
            chsh_circuits.append(qc)
    return chsh_circuits


def compute_chsh_witness(counts):
    # Order is ZZ,ZX,XZ,XX
    CHSH1 = []
    CHSH2 = []
    # Divide the list of dictionaries in sets of 4
    for i in range(0, len(counts), 4):
        theta_dict = counts[i:i + 4]
        zz = theta_dict[0]
        zx = theta_dict[1]
        xz = theta_dict[2]
        xx = theta_dict[3]
        no_shots = sum(xx[y] for y in xx)
        chsh1 = 0
        chsh2 = 0
        for element in zz:
            parity = (-1)**(int(element[0])+int(element[1]))
            chsh1 += parity*zz[element]
            chsh2 += parity*zz[element]
        for element in zx:
            parity = (-1)**(int(element[0])+int(element[1]))
            chsh1 += parity*zx[element]
            chsh2 -= parity*zx[element]
        for element in xz:
            parity = (-1)**(int(element[0])+int(element[1]))
            chsh1 -= parity*xz[element]
            chsh2 += parity*xz[element]
        for element in xx:
            parity = (-1)**(int(element[0])+int(element[1]))
            chsh1 += parity*xx[element]
            chsh2 += parity*xx[element]
        CHSH1.append(chsh1/no_shots)
        CHSH2.append(chsh2/no_shots)
    return CHSH1, CHSH2


def run_circuits(my_chsh_circuits):
    # Error probabilities
    prob_1 = np.random.uniform()  # 1-qubit gate
    prob_2 = np.random.uniform()  # 2-qubit gate
    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)
    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates
    # Perform a noise simulation
    result = execute(my_chsh_circuits, Aer.get_backend('aer_simulator'),
                     basis_gates=basis_gates,
                     noise_model=noise_model).result()
    return result, prob_1, prob_2


def get_chsh(number_of_thetas=15):
    theta_vec = np.linspace(0, 2*np.pi, number_of_thetas)
    my_chsh_circuits = make_chsh_circuit(theta_vec)
    # Execute and get counts
    result_ideal, prob_1, prob_2 = run_circuits(my_chsh_circuits)
    _, CHSH2_ideal = compute_chsh_witness(result_ideal.get_counts())
    return theta_vec, [prob_1, prob_2], CHSH2_ideal
