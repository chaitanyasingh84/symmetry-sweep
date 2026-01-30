# Symmetry Sweep Analysis for Molecular Hamiltonians

This code performs **additive symmetry analysis** of truncated Jordan-Wigner Hamiltonians for H4 linear chains at various bond distances. It identifies emergent symmetries as epsilon-truncation thresholds are varied.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Mathematical Background](#mathematical-background)
4. [Algorithm Workflow](#algorithm-workflow)
5. [Function Reference](#function-reference)
6. [Configuration](#configuration)
7. [Output Format](#output-format)
8. [Citation](#citation)

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy` — Numerical operations
- `pandas` — Data organization
- `openfermion` — Fermionic/qubit operator algebra
- `openfermionpyscf` — Interface to PySCF
- `pyscf` — Quantum chemistry calculations

**Note:** PySCF may require system libraries:
```bash
sudo apt-get install libopenblas-dev  # Ubuntu/Debian
```

---

## Quick Start

```bash
python symmetry_sweep_h4.py
```

Or import as a module:
```python
from symmetry_sweep_h4 import main, build_jw_hamiltonian_for_R

df_results, basis_data = main()
```

---

## Mathematical Background

### 1. Molecular Hamiltonian

The electronic Hamiltonian in second quantization:

$$\hat{H} = \sum_{pq} h_{pq} \hat{a}_p^\dagger \hat{a}_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} \hat{a}_p^\dagger \hat{a}_q^\dagger \hat{a}_r \hat{a}_s$$

where $\hat{a}_p^\dagger$ and $\hat{a}_p$ are fermionic creation/annihilation operators.

### 2. Jordan-Wigner Transformation

Maps fermionic operators to qubit (Pauli) operators:

$$\hat{a}_j^\dagger \rightarrow \frac{1}{2}(X_j - iY_j) \otimes Z_{j-1} \otimes Z_{j-2} \otimes \cdots \otimes Z_0$$

The resulting qubit Hamiltonian is a sum of Pauli strings:

$$H = \sum_k c_k P_k, \quad P_k \in \{I, X, Y, Z\}^{\otimes n}$$

### 3. Symplectic Representation of Paulis

Any $n$-qubit Pauli string $P$ (ignoring phase) can be encoded as a binary vector of length $2n$:

$$P \leftrightarrow (\mathbf{a}_x | \mathbf{a}_z) \in \mathbb{F}_2^{2n}$$

where for each qubit $q$:

| Pauli | $a_x[q]$ | $a_z[q]$ |
|-------|----------|----------|
| $I$   | 0        | 0        |
| $X$   | 1        | 0        |
| $Z$   | 0        | 1        |
| $Y$   | 1        | 1        |

### 4. Commutation via Symplectic Inner Product

Two Paulis $P$ and $Q$ with symplectic vectors $\mathbf{g} = (g_x | g_z)$ and $\mathbf{s} = (s_x | s_z)$ satisfy:

$$[P, Q] = 0 \iff \langle \mathbf{g}, \mathbf{s} \rangle_{\text{symp}} = 0 \pmod 2$$

where the **symplectic inner product** is:

$$\langle \mathbf{g}, \mathbf{s} \rangle_{\text{symp}} = g_x \cdot s_z + g_z \cdot s_x = \sum_q (g_x[q] \cdot s_z[q] + g_z[q] \cdot s_x[q])$$

### 5. Finding Symmetries

A Pauli $S$ is a **symmetry** of $H$ if $[S, H] = 0$, i.e., $S$ commutes with every term in $H$.

Given Hamiltonian terms encoded as rows of matrix $G$, we seek all $\mathbf{s}$ such that:

$$A \cdot \mathbf{s} = \mathbf{0} \pmod 2$$

where $A$ is constructed by swapping $x$ and $z$ blocks: if $G = (G_x | G_z)$, then $A = (G_z | G_x)$.

**Symmetries = Nullspace of $A$ over $\mathbb{F}_2$ (GF(2))**

### 6. Epsilon Truncation

For threshold $\varepsilon$, define truncated Hamiltonian:

$$H_\varepsilon = \sum_{k: |c_k| \geq \varepsilon} c_k P_k$$

As $\varepsilon$ increases, terms are dropped, and the symmetry space typically **grows** (fewer constraints).

### 7. Additive Basis

We maintain a **monotonically growing** basis across all $\varepsilon$ values. New symmetries discovered at higher $\varepsilon$ are added only if they are linearly independent (over GF(2)) from existing ones.

---

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN WORKFLOW                                      │
└─────────────────────────────────────────────────────────────────────────────┘

For each bond distance R in R_VALUES:
│
├──► [1] BUILD MOLECULE
│        │
│        ├── build_geometry(R)           → Define H4 atomic positions
│        │
│        └── build_jw_hamiltonian_for_R(R)
│                │
│                ├── MolecularData()     → Create OpenFermion molecule
│                ├── run_pyscf()         → Compute integrals (SCF + FCI)
│                ├── get_fermion_operator() → Get fermionic Hamiltonian
│                └── jordan_wigner()     → Transform to qubit Hamiltonian H
│
├──► [2] SYMMETRY SWEEP (symmetry_sweep_additive)
│        │
│        │   Initialize: additive_basis = ∅
│        │
│        │   For each ε in linspace(0, ε_max, NUM_EPS_DIVISIONS):
│        │   │
│        │   ├── [2a] TRUNCATE
│        │   │        truncate_qubitop(H, ε) → H_ε (drop terms with |c| < ε)
│        │   │
│        │   ├── [2b] BUILD SYMPLECTIC MATRIX
│        │   │        │
│        │   │        ├── qubitop_to_G_matrix(H_ε)
│        │   │        │       For each Pauli term P_k:
│        │   │        │         pauli_term_to_ax_az() → (a_x, a_z)
│        │   │        │       Stack into G = (G_x | G_z)
│        │   │        │
│        │   │        └── build_commutation_constraints_A(G)
│        │   │                A = (G_z | G_x)  [swap x,z blocks]
│        │   │
│        │   ├── [2c] FIND SYMMETRIES (GF(2) Linear Algebra)
│        │   │        │
│        │   │        ├── gf2_nullspace(A)
│        │   │        │       └── gf2_rref(A) → Row-reduce over GF(2)
│        │   │        │       └── Solve for free variables
│        │   │        │       → basis vectors {s : A·s = 0 mod 2}
│        │   │        │
│        │   │        └── gf2_rref(basis) → Clean RREF form
│        │   │
│        │   ├── [2d] EXTEND ADDITIVE BASIS
│        │   │        gf2_extend_basis_additive(additive_basis, new_basis)
│        │   │            For each candidate vector v:
│        │   │              if gf2_rank([additive_basis; v]) > current_rank:
│        │   │                additive_basis ← additive_basis ∪ {v}
│        │   │
│        │   ├── [2e] VALIDATE
│        │   │        gf2_check_in_nullspace(A, additive_basis)
│        │   │            Verify A · additive_basis^T = 0 mod 2
│        │   │
│        │   └── [2f] RECORD RESULTS
│        │            Convert to Pauli strings: symplectic_to_pauli_string()
│        │            Store in DataFrame row
│        │
│        └── Return: df, basis_additive_list, basis_rref_list
│
└──► [3] AGGREGATE RESULTS
         Concatenate DataFrames for all R values
         Return combined df_R_eps_additive
```

---

## Function Reference

### Molecule Construction

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `build_geometry(R)` | Define H4 linear chain coordinates | `R`: bond distance (Å) | List of `(atom, (x,y,z))` tuples |
| `build_jw_hamiltonian_for_R(R)` | Full pipeline: geometry → JW Hamiltonian | `R`: bond distance | `(H, mol)`: QubitOperator, MolecularData |

### Qubit Operator Utilities

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `infer_n_qubits_from_qubitop(H)` | Count qubits from operator | QubitOperator | `int`: number of qubits |
| `truncate_qubitop(H, eps)` | Drop small terms | `H`, threshold `eps` | Truncated QubitOperator |

### Symplectic Representation

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `pauli_term_to_ax_az(term, n)` | Convert single Pauli term to binary vectors | Term tuple, `n` qubits | `(ax, az)`: uint8 arrays |
| `qubitop_to_G_matrix(H, n)` | Build symplectic matrix for all terms | QubitOperator, optional `n` | `(G, coeffs, labels, n)` |
| `symplectic_to_pauli_string(s, n)` | Convert binary vector to readable string | Vector `s`, `n` qubits | String like `"X0 Z2 Y5"` |

### Commutation Constraints

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `build_commutation_constraints_A(G, n)` | Build constraint matrix (swap x↔z) | Symplectic matrix `G`, `n` qubits | Matrix `A` of shape `(m, 2n)` |

### GF(2) Linear Algebra

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `gf2_rref(A)` | Row-reduce over GF(2) | Binary matrix | `(R, pivots)`: RREF matrix, pivot columns |
| `gf2_rank(M)` | Compute rank over GF(2) | Binary matrix | `int`: rank |
| `gf2_nullspace(A)` | Find nullspace basis | Constraint matrix | `(k, 2n)` basis array |
| `gf2_check_in_nullspace(A, S)` | Verify vectors satisfy constraints | `A`, candidate matrix `S` | `bool`: all in nullspace? |
| `gf2_extend_basis_additive(B, candidates)` | Extend basis with independent vectors | Current basis, new candidates | `(B_new, added_vectors)` |

### Main Sweep

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `symmetry_sweep_additive(H, ...)` | Full epsilon sweep for one Hamiltonian | `H`, `num_intervals`, `eps_max`, flags | `(df, basis_list, rref_list)` |
| `main()` | Run sweep over all R values | — | `(df_combined, basis_by_R)` |

---

## Configuration

Edit constants at the top of `symmetry_sweep_h4.py`:

```python
# Epsilon discretization
NUM_EPS_DIVISIONS = 1000      # Number of ε points in [0, ε_max]

# Bond distances to scan (Angstroms)
R_VALUES = [1.0, 1.2, 1.4, ..., 3.0]

# Quantum chemistry basis set
BASIS_NAME = "sto-3g"         # Options: "sto-3g", "6-31g", "cc-pvdz", ...
MULTIPLICITY = 1              # 2S+1 spin multiplicity
CHARGE = 0                    # Molecular charge

# Epsilon max policy
EPS_POLICY = "per_R_max"      # "per_R_max" or "global_max"

# Output verbosity
VERBOSE_PER_R = True          # Print progress per R
PRINT_HEAD_ROWS = 3           # DataFrame rows to preview
```

### Customizing the Molecule

To study a different molecule, modify `build_geometry(R)`:

```python
def build_geometry(R):
    # Example: H2O with variable O-H distance
    angle = 104.5 * np.pi / 180
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (R, 0.0, 0.0)),
        ("H", (R * np.cos(angle), R * np.sin(angle), 0.0)),
    ]
```

---

## Output Format

### DataFrame Columns

| Column | Type | Description |
|--------|------|-------------|
| `R_Ang` | float | Bond distance (Å) |
| `eps_idx` | int | Index in epsilon grid |
| `epsilon` | float | Truncation threshold |
| `terms_left` | int | Pauli terms remaining after truncation |
| `k_nullspace` | int | Dimension of symmetry space at this ε |
| `k_additive` | int | Dimension of cumulative additive basis |
| `new_added` | int | Number of new symmetries found at this ε |
| `new_added_syms` | str | New symmetry operators (e.g., `"Z0 Z1 \| X2 X3"`) |
| `additive_basis_syms` | str | Full additive basis as Pauli strings |
| `additive_in_nullspace?` | bool | Sanity check: basis valid for current ε |
| `jw_terms_full` | int | Total JW terms before truncation |
| `basis` | str | Basis set name |
| `geometry` | str | Geometry description |
| `fci_energy` | float | FCI ground state energy (Hartree) |

### Example Output

```
==================== R = 1.0 Å ====================
JW terms = 185, eps_max = 0.674523

[eps_idx=   0] ε=0         terms_left= 185  k_null= 0  k_additive= 0  (+0)  ok_null=True
[eps_idx= 142] ε=0.0959... terms_left=  89  k_null= 1  k_additive= 1  (+1)  ok_null=True
   + Z0 Z1 Z2 Z3 Z4 Z5 Z6 Z7
[eps_idx= 287] ε=0.1937... terms_left=  45  k_null= 3  k_additive= 3  (+2)  ok_null=True
   + Z0 Z2 Z4 Z6
   + Z1 Z3 Z5 Z7
```

---

## Citation

```bibtex
@software{symmetry_sweep_h4,
  title   = {Symmetry Sweep Analysis for Molecular Hamiltonians},
  author  = {[Your Name]},
  year    = {2025},
  url     = {https://github.com/[username]/symmetry-sweep}
}
```

---

## License

MIT License — See `LICENSE` file for details.
