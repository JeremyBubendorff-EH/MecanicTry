import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# Material and cross-sectional properties
E = 210e9  # Young's modulus in Pa
rho = 7800  # Density in kg/m^3
nu = 0.3    # Poisson's ratio
D_outer = 0.15  # Outer diameter in meters
t = 0.005  # Wall thickness in meters
D_inner = D_outer - (2 * t)
A = np.pi * (D_outer**2 - D_inner**2) / 4  # Cross-sectional area
I = np.pi * (D_outer**4 - D_inner**4) / 64  # Moment of inertia
G = E / (2 * (1 + nu)) #Shear Modulus
J = (np.pi / 32) * (D_outer**4 - D_inner**4)

# Geometry of the truss model
heights = [5, 3, 1]  # Heights of main supports in meters
spacing_x = 4  # Horizontal spacing between supports in meters
width_y = 2  # Width of structure in meters
bar_spacing = 1  # Spacing of upper transverse bars in meters
num_supporters = 51  # Total number of supporters
weight_per_supporter = 80  # Weight per supporter in kg

# Lumped masses at extremities
mass_supporters = num_supporters * weight_per_supporter
mass_per_node = mass_supporters / 18  # Uniform distribution over 18 nodes

#Simulation parameters
num_elements = 57
num_nodes = num_elements + 1
constrained_dofs = [0, 1, 2, 3, 4, 5]  # Clamped supports

def element_matrices(L, A, I, E, rho):
    """Returns stiffness and mass matrices for a 3D beam element."""
    k = np.zeros((12, 12))
    m = np.zeros((12, 12))
    # Axial stiffness
    k_axial = E * A / L
    # Flexural terms
    k_flexion = 12 * E * I / L ** 3
    k_rot_flexion = 6 * E * I / L ** 2
    k_rotational = 4 * E * I / L
    k_cross = 2 * E * I / L
    # Shear and torsion
    k_torsion = G * J / L

    # Populate stiffness matrix
    k[0, 0] = k[6, 6] = k_axial
    k[0, 6] = k[6, 0] = -k_axial
    k[1, 1] = k[7, 7] = k_flexion
    k[5, 5] = k[11, 11] = k_rotational
    k[3, 3] = k[9, 9] = k_torsion
    k[3, 9] = k[9, 3] = -k_torsion
    k[1, 7] = k[7, 1] = -k_flexion
    k[5, 11] = k[11, 5] = -k_rotational
    k[1, 5] = k[5, 1] = k_rot_flexion
    k[1, 11] = k[11, 1] = -k_rot_flexion
    k[5, 7] = k[7, 5] = -k_rot_flexion
    k[7, 11] = k[11, 7] = k_rot_flexion

    # Flexion dans l'autre direction
    k[2, 2] = k[8, 8] = k_flexion
    k[4, 4] = k[10, 10] = k_rotational
    k[2, 8] = k[8, 2] = -k_flexion
    k[4, 10] = k[10, 4] = -k_rotational
    k[2, 4] = k[4, 2] = -k_rot_flexion
    k[2, 10] = k[10, 2] = k_rot_flexion
    k[4, 8] = k[8, 4] = k_rot_flexion
    k[8, 10] = k[10, 8] = -k_rot_flexion

    # Matrice de masse pour la poutre (matrice de consistance de masse pour un élément en 3D)
    m[0, 0] = m[6, 6] = rho * A * L / 3
    m[0, 6] = m[6, 0] = rho * A * L / 6
    m[1, 1] = m[7, 7] = 13 * rho * A * L / 35
    m[1, 7] = m[7, 1] = 9 * rho * A * L / 70
    m[2, 2] = m[8, 8] = 13 * rho * A * L / 35
    m[2, 8] = m[8, 2] = 9 * rho * A * L / 70
    m[4, 4] = m[10, 10] = rho * I * L / 3
    m[5, 5] = m[11, 11] = rho * I * L / 3

    return k, m

k, m = element_matrices(spacing_x, A, I, E, rho)
assert np.all(np.isfinite(k)), "La matrice de rigidité locale contient des NaN ou des infinities."
assert np.all(np.isfinite(m)), "La matrice de masse locale contient des NaN ou des infinities."
print("Matrice de rigidité d'un élément:\n", k)
print("Matrice de masse d'un élément:\n", m)

def assemble_global_matrices(num_elements, dof_per_node=6):
    """Assemble global stiffness and mass matrices."""
    num_nodes = num_elements + 1
    total_dofs = num_nodes * dof_per_node
    K_global = np.zeros((total_dofs, total_dofs))
    M_global = np.zeros((total_dofs, total_dofs))

    for element in range(num_elements):
        L = spacing_x / num_elements  # Element length
        k_local, m_local = element_matrices(L, A, I, E, rho)
        start_dof = element * dof_per_node
        end_dof = start_dof + 2 * dof_per_node
        K_global[start_dof:end_dof, start_dof:end_dof] += k_local
        M_global[start_dof:end_dof, start_dof:end_dof] += m_local

    return K_global, M_global

    # Adjusted index range to cover 12 DOFs per element (6 DOFs per node for 2 nodes)
    start_dof = element * dof_per_node
    end_dof = (element + 2) * dof_per_node
    index_range = slice(start_dof, end_dof)

    # Assemble local matrices into the global matrices
    K_global[index_range, index_range] += k_local
    M_global[index_range, index_range] += m_local

    # Rendre les matrices symétriques
    K_global = (K_global + K_global.T) / 2
    M_global = (M_global + M_global.T) / 2

    # Ajouter une diagonale dominante pour éviter la singularité
    np.fill_diagonal(K_global, np.abs(K_global).sum(axis=1) + 10)
    np.fill_diagonal(M_global, np.abs(M_global).sum(axis=1) + 1)

    return K_global, M_global

K_global, M_global = assemble_global_matrices(10)
print("Matrice de rigidité globale:\n", K_global)
print("Matrice de masse globale:\n", M_global)

def apply_boundary_conditions(K_global, M_global, constrained_dofs):
    """Apply boundary conditions to global matrices."""
    for dof in constrained_dofs:
        K_global[dof, :] = 0
        K_global[:, dof] = 0
        K_global[dof, dof] = 1e12
        M_global[dof, :] = 0
        M_global[:, dof] = 0
        M_global[dof, dof] = 1e-6
    return K_global, M_global

def compute_natural_frequencies(num_elements, constrained_dofs):
    """Compute natural frequencies and mode shapes."""
    K_global, M_global = assemble_global_matrices(num_elements)
    K_global, M_global = apply_boundary_conditions(K_global, M_global, constrained_dofs)
    K_sparse = csr_matrix(K_global)
    M_sparse = csr_matrix(M_global)
    eigenvalues, eigenvectors = eigsh(K_sparse, 6, M_sparse, which='SM', tol=1e-5)
    frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)  # Convert to Hz
    return frequencies, eigenvector

    # # Calcul des fréquences naturelles
    # frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)  # Conversion en Hz
    # # Extract the first six natural frequencies and their modes
    # first_six_frequencies = natural_frequencies[:6]
    # first_six_modes = eigenvectors[:, :6]

    # return first_six_frequencies, first_six_modes
    # return frequencies, eigenvectors

def convergence_study(min_elements=1, max_elements=20):
    """Perform a convergence study on natural frequencies as a function of element count."""
    first_frequency_values = []  # Store the first natural frequency for each element count
    element_counts = range(min_elements, max_elements + 1)  # Range of element counts to test

    for num_elements in element_counts:
        first_six_frequencies, first_six_modes  = compute_natural_frequencies (num_elements, constrained_dofs)  # Compute natural frequencies
        first_frequency_values.append(first_six_frequencies[0])  # Record the first natural frequency

    # Plot the convergence of the first natural frequency
    plt.figure(figsize=(10, 6))
    plt.plot(element_counts, first_frequency_values, marker='o', linestyle='-')
    plt.xlabel("Number of Elements per Beam")
    plt.ylabel("First Natural Frequency (Hz)")
    plt.title("Convergence of the First Natural Frequency")
    plt.grid(True)
    plt.show()

convergence_study(1, 15)

def plot_mode_shapes(modes, num_nodes, dof_per_node=6):
    """Plot the first six mode shapes."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(6):
        ax = axs[i // 3, i % 3]
        mode = modes[:, i]
        x_positions = np.linspace(0, num_nodes - 1, num_nodes)
        displacement = mode[0::dof_per_node]  # Simplified x-displacement
        ax.plot(x_positions, displacement, marker='o', linestyle='-', color='b')
        ax.set_title(f"Mode {i + 1}")
        ax.set_xlabel("Node Position")
        ax.set_ylabel("Displacement")
    plt.tight_layout()
    plt.show()

def compute_element_mass(rho, A, L):
    """Calculate mass of a single beam element."""
    return rho * A * L

def compute_total_mass(num_elements, heights, spacing_x, mass_supporters):
    """Compute the total mass of the structure and supporters."""
    structure_mass = 0
    for height in heights:
        L = height / num_elements
        structure_mass += num_elements * rho * A * L
    for _ in range(len(heights) - 1):
        L = spacing_x / num_elements
        structure_mass += 2 * num_elements * rho * A * L  # Two bars per level
    total_mass = structure_mass + mass_supporters
    return total_mass

frequencies, modes = compute_natural_frequencies(num_elements, constrained_dofs)
print("First six natural frequencies (Hz):\n", frequencies)

plot_mode_shapes(modes, num_nodes)
print("Premières six formes modales:\n", modes)

total_mass = compute_total_mass(num_elements, heights, spacing_x, mass_supporters)
print("Total mass of the stadium (kg):", total_mass)