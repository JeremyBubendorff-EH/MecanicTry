import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BeamElement3D:
    def __init__(self, E, G, A, Iy, Iz, J, L, rho):

        self.E = E
        self.G = G
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.L = L
        self.rho = rho

    def get_stiffness_matrix(self):
        """Calculate element stiffness matrix"""
        E, G, A, Iy, Iz, J, L = self.E, self.G, self.A, self.Iy, self.Iz, self.J, self.L

        # Initialize 12x12 matrix
        K = np.zeros((12, 12))

        # Axial terms
        K[0, 0] = K[6, 6] = E * A / L
        K[0, 6] = K[6, 0] = -E * A / L

        # Bending terms (y-z plane)
        EIz = E * Iz
        K[1, 1] = K[7, 7] = 12 * EIz / (L ** 3)
        K[1, 7] = K[7, 1] = -12 * EIz / (L ** 3)
        K[1, 5] = K[5, 1] = K[1, 11] = K[11, 1] = 6 * EIz / (L ** 2)
        K[5, 7] = K[7, 5] = -6 * EIz / (L ** 2)
        K[5, 5] = K[11, 11] = 4 * EIz / L
        K[5, 11] = K[11, 5] = 2 * EIz / L

        # Bending terms (x-z plane)
        EIy = E * Iy
        K[2, 2] = K[8, 8] = 12 * EIy / (L ** 3)
        K[2, 8] = K[8, 2] = -12 * EIy / (L ** 3)
        K[2, 4] = K[4, 2] = K[2, 10] = K[10, 2] = -6 * EIy / (L ** 2)
        K[4, 8] = K[8, 4] = 6 * EIy / (L ** 2)
        K[4, 4] = K[10, 10] = 4 * EIy / L
        K[4, 10] = K[10, 4] = 2 * EIy / L

        # Torsional terms
        GJ = G * J
        K[3, 3] = K[9, 9] = GJ / L
        K[3, 9] = K[9, 3] = -GJ / L

        return K

    def get_mass_matrix(self):
        """Calculate element consistent mass matrix"""
        rho, A, L = self.rho, self.A, self.L

        # Initialize 12x12 matrix
        M = np.zeros((12, 12))

        # Lumped mass approach for simplicity
        m = rho * A * L

        # Translational mass terms
        for i in [0, 1, 2, 6, 7, 8]:
            M[i, i] = m / 2

        # Rotational mass terms (simplified)
        for i in [3, 4, 5, 9, 10, 11]:
            M[i, i] = m * L ** 2 / 24

        return M

class StadiumTruss:
    def __init__(self):
        # Material properties
        self.E = 210e9  # Young's modulus [Pa]
        self.rho = 7800  # Density [kg/m³]
        self.nu = 0.3  # Poisson's ratio
        self.G = self.E / (2 * (1 + self.nu))  # Shear modulus

        # Cross-section properties (circular hollow section)
        self.D_outer = 0.150  # Outer diameter [m]
        self.t = 0.005  # Wall thickness [m]
        self.D_inner = self.D_outer - 2 * self.t
        self.A = np.pi / 4 * (self.D_outer ** 2 - self.D_inner ** 2)
        self.I = np.pi / 64 * (self.D_outer ** 4 - self.D_inner ** 4)
        self.J = 2 * self.I

        # Structure dimensions
        self.width = 2  # Width [m]
        self.heights = [5, 3, 1]  # Support heights [m]
        self.spacing = 4  # Support spacing [m]
        self.transverse_spacing = 1  # Transverse bar spacing [m]

        # Supporter mass properties
        self.supporter_mass = 80  # Mass per supporter [kg]
        self.num_supporters = 51
        self.num_mass_nodes = 18

        # Initialize truss members
        self.truss_lengths = self.calculate_truss_lengths()

    def generate_mesh(self, elements_per_beam):
        """Générer les noeuds et éléments pour la structure"""
        nodes = []
        elements = []
        current_node_id = 0

        # Génération des nœuds pour les colonnes
        for col_index, height in enumerate(self.heights):
            base_x = col_index * self.spacing
            for i in range(elements_per_beam + 1):
                z = i * height / elements_per_beam
                nodes.append([base_x, 0, z])
                if i > 0:
                    elements.append((current_node_id - 1, current_node_id))
                current_node_id += 1

        # Génération des nœuds pour les poutres horizontales
        for level in range(1, elements_per_beam + 1):
            for i in range(3):
                base_x = i * self.spacing
                base_z = self.heights[i] * level / elements_per_beam
                nodes.append([base_x, self.width, base_z])
                if i > 0:
                    elements.append((current_node_id - 1, current_node_id))
                current_node_id += 1

        return np.array(nodes), np.array(elements)

    def calculate_truss_lengths(self):
        # Extract dimensions
        height_diff = np.diff(self.heights)  # Differences in support heights
        num_spans = len(self.heights) - 1  # Number of spans between supports

        # Horizontal span lengths
        horizontal_length = self.spacing  # Fixed for each span

        # Diagonal member lengths (between supports)
        diagonal_lengths = [
        np.sqrt(horizontal_length ** 2 + h_diff ** 2)
        for h_diff in height_diff
        ]

        # Vertical bar lengths (support heights)
        vertical_lengths = self.heights

        # Transverse bar lengths (across width)
        transverse_length = self.width

        # Combine lengths (example order: diagonals, verticals, transverse bars)
        member_lengths = diagonal_lengths + vertical_lengths + [transverse_length]
        return member_lengths

    def calculate_member_masses(self):
        """
        Calculate mass for each truss member based on its length.
        """
        masses = [self.rho * self.A * L for L in self.truss_lengths]
        return masses

    def calculate_total_truss_mass(self):
        """
        Calculate the total mass of the truss structure.
        """
        member_masses = self.calculate_member_masses()
        return sum(member_masses)

    def calculate_mass_per_node(self):
        """
        Calculate mass per node.
        """
        total_mass = self.calculate_total_truss_mass()
        return total_mass / self.num_mass_nodes

    def calculate_total_mass(self):
        """Calculate total mass using rigid body mode"""
        # The total mass is the sum of translational DOFs along the diagonal
        translational_dofs = np.arange(0, M.shape[0], 6)
        total_mass = np.sum(np.diag(M)[translational_dofs])
        return total_mass

    print("Member masses (kg):", member_masses)
    print("Total truss mass (kg):", total_mass)
    print("Mass per node (kg):", mass_per_node)
    print("Total mass (kg) :"), total_mass

    def assemble_global_matrices(self, elements_per_beam):
        """Assemble global stiffness and mass matrices"""
        # Generate mesh
        nodes, elements = self.generate_mesh(elements_per_beam)
        num_nodes = len(nodes)
        num_dof = 6 * num_nodes

        # Initialize global matrices
        K_global = np.zeros((num_dof, num_dof))
        M_global = np.zeros((num_dof, num_dof))

        # Assemble matrices for each element
        for elem in elements:
            # Get element nodes
            node1, node2 = nodes[elem[1]], nodes[elem[2]]

            # Calculate element length and orientation
            dx = node2[1] - node1[1]
            dy = node2[2] - node1[2]
            dz = node2[3] - node1[3]
            L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # Create element
            beam = BeamElement3D(
                self.E, self.G, self.A, self.I, self.I, self.J, L, self.rho
            )

            # Get element matrices
            K_elem = beam.get_stiffness_matrix()
            M_elem = beam.get_mass_matrix()

            # Calculate transformation matrix (simplified - assumes small rotations)
            T = np.eye(12)  # For now, using identity transformation

            # Transform element matrices
            K_elem = T.T @ K_elem @ T
            M_elem = T.T @ M_elem @ T

            # Get global DOF indices
            dof1 = 6 * elem[1]
            dof2 = 6 * elem[2]
            dofs = np.concatenate([np.arange(dof1, dof1 + 6), np.arange(dof2, dof2 + 6)])

            # Assemble into global matrices
            for i in range(12):
                for j in range(12):
                    K_global[dofs[i], dofs[j]] += K_elem[i, j]
                    M_global[dofs[i], dofs[j]] += M_elem[i, j]

        # Add lumped masses for supporters
        mass_per_node = self.mass_per_node
        for node in range(num_nodes):
            # Add mass to translational DOFs only
            for dof in range(3):
                M_global[6 * node + dof, 6 * node + dof] += mass_per_node

        # Apply boundary conditions (fixed supports)
        # Identify fixed nodes (at z=0)
        fixed_nodes = nodes[nodes[:, 3] == 0]
        fixed_dofs = []
        for node in fixed_nodes[:, 0]:
            fixed_dofs.extend([6 * int(node) + i for i in range(6)])

        # Remove fixed DOFs
        free_dofs = np.setdiff1d(np.arange(num_dof), fixed_dofs)
        K_global = K_global[np.ix_(free_dofs, free_dofs)]
        M_global = M_global[np.ix_(free_dofs, free_dofs)]

        return K_global, M_global, nodes, elements

    def solve_eigenvalue_problem(self, K, M):
        """Résoudre le problème aux valeurs propres"""
        eigenvalues, eigenvectors = linalg.eigh(K, M)
        natural_frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
        sorted_indices = np.argsort(natural_frequencies)
        natural_frequencies = natural_frequencies[sorted_indices]
        mode_shapes = eigenvectors[:, sorted_indices]
        return natural_frequencies, mode_shapes

    def plot_mode_shape(self, nodes, elements, mode_vector, mode_number, frequency, scale_factor=1.0):
        """Plot the mode shape for a given mode number"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Get original and deformed node coordinates
        original_coords = nodes[:, 1:]  # Skip node ID

        # Reconstruct full mode shape vector (including fixed DOFs)
        full_mode_vector = np.zeros(len(nodes) * 6)
        free_dof_idx = 0
        fixed_nodes = nodes[nodes[:, 3] == 0]
        fixed_dofs = []
        for node in fixed_nodes[:, 0]:
            fixed_dofs.extend([6 * int(node) + i for i in range(6)])

        for i in range(len(full_mode_vector)):
            if i not in fixed_dofs:
                full_mode_vector[i] = mode_vector[free_dof_idx]
                free_dof_idx += 1

        # Extract translational components
        deformed_coords = np.zeros_like(original_coords)
        for i in range(len(nodes)):
            for j in range(3):  # x, y, z coordinates
                deformed_coords[i, j] = original_coords[i, j] + scale_factor * full_mode_vector[6 * i + j]

        # Plot original structure (gray)
        for elem in elements:
            node1, node2 = int(elem[1]), int(elem[2])
            x = [original_coords[node1, 0], original_coords[node2, 0]]
            y = [original_coords[node1, 1], original_coords[node2, 1]]
            z = [original_coords[node1, 2], original_coords[node2, 2]]
            ax.plot(x, y, z, 'gray', alpha=0.3, linewidth=1)

        # Plot deformed structure (blue)
        for elem in elements:
            node1, node2 = int(elem[1]), int(elem[2])
            x = [deformed_coords[node1, 0], deformed_coords[node2, 0]]
            y = [deformed_coords[node1, 1], deformed_coords[node2, 1]]
            z = [deformed_coords[node1, 2], deformed_coords[node2, 2]]
            ax.plot(x, y, z, 'b', linewidth=2)

        # Plot nodes
        ax.scatter(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2],
                   c='gray', alpha=0.3, s=30)
        ax.scatter(deformed_coords[:, 0], deformed_coords[:, 1], deformed_coords[:, 2],
                   c='blue', s=30)

        # Highlight fixed nodes
        fixed_nodes_coords = original_coords[nodes[:, 3] == 0]
        ax.scatter(fixed_nodes_coords[:, 0], fixed_nodes_coords[:, 1], fixed_nodes_coords[:, 2],
                   c='red', s=100, marker='s', label='Fixed supports')

        # Set labels and title
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f'Mode {mode_number + 1}: {frequency:.2f} Hz')

        # Add legend
        ax.legend(['Original structure', 'Deformed shape', 'Nodes', 'Fixed supports'])

        # Set equal aspect ratio
        ax.set_box_aspect([np.ptp(original_coords[:, 0]),
                           np.ptp(original_coords[:, 1]),
                           np.ptp(original_coords[:, 2])])

        plt.tight_layout()
        return fig

    def convergence_study(self, max_elements=10):
        frequencies = []
        element_counts = list(range(2, max_elements + 1))

        for n_elements in element_counts:
            # Assemble global matrices for current refinement
            K, M, nodes, elements = self.assemble_global_matrices(n_elements)
            natural_frequencies, _ = self.solve_eigenvalue_problem(K, M)

            # Store the first few frequencies
            frequencies.append(natural_frequencies[:6])  # Adjust number as needed

        return element_counts, np.array(frequencies)

    def plot_convergence_study(self, element_counts, frequencies):

        plt.figure(figsize=(10, 6))
        for i in range(frequencies.shape[1]):
            plt.plot(element_counts, frequencies[:, i], marker='o', label=f'Mode {i + 1}')
        plt.xlabel('Number of Elements per Beam')
        plt.ylabel('Natural Frequency (Hz)')
        plt.title('Convergence Study')
        plt.legend()
        plt.grid()
        plt.show()

# Create stadium model
stadium = StadiumTruss()

# Number of elements per beam
elements_per_beam = 5

# Generate mesh and get nodes and elements
nodes, elements = stadium.generate_mesh(elements_per_beam)

# Perform analysis
K, M = stadium.assemble_global_matrices(elements_per_beam)
# Get natural frequencies and mode shapes
frequencies, mode_shapes = stadium.solve_eigenvalue_problem(K, M)

# Print first 6 natural frequencies
print("\nFirst 6 natural frequencies (Hz):")
for i, freq in enumerate(frequencies[:6]):
    print(f"Mode {i + 1}: {freq:.2f} Hz")

# Plot first 6 mode shapes
scale_factor = 0.5  # Adjust this to make deformations more visible
for i in range(6):
    fig = stadium.plot_mode_shape(nodes, elements, mode_shapes[:, i], i, frequencies[i], scale_factor)
    plt.show()


