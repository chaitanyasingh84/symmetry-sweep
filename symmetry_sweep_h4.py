#!/usr/bin/env python3
"""
Symmetry Sweep Analysis for Molecular Hamiltonians under Jordan-Wigner Transformation.

This script performs additive symmetry analysis of truncated Jordan-Wigner 
Hamiltonians for H4 linear chain at various bond distances. It identifies 
emergent symmetries as epsilon-truncation thresholds are varied.

Dependencies:
    pip install numpy pandas openfermion openfermionpyscf pyscf

Usage:
    python symmetry_sweep_h4.py

Author: [Your Name]
Date: [Date]
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pandas as pd

from openfermion import MolecularData, QubitOperator
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner


# =============================================================================
# CONFIGURATION
# =============================================================================

# Epsilon discretization
NUM_EPS_DIVISIONS = 1000

# Radius sweep [1.0..3.0 step 0.2] in Angstroms
R_VALUES = list(np.round(np.arange(1.0, 3.0, 0.2), 10)) + [3.0]

# Molecular parameters
BASIS_NAME = "sto-3g"
MULTIPLICITY = 1
CHARGE = 0

# Epsilon max policy:
#   "per_R_max": eps_max = 1.000001 * max |coeff| of that R's Hamiltonian
#   "global_max": use the maximum over all R (comparable epsilon scale across R)
EPS_POLICY = "per_R_max"

# Output verbosity
VERBOSE_PER_R = True
PRINT_HEAD_ROWS = 3


# =============================================================================
# MOLECULE CONSTRUCTION
# =============================================================================

def build_geometry(R):
    """
    Build molecular geometry for linear H4 chain.
    
    Parameters
    ----------
    R : float
        Bond distance in Angstroms.
        
    Returns
    -------
    list of tuple
        List of (atom_symbol, (x, y, z)) tuples.
    """
    return [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.0 * R)),
        ("H", (0.0, 0.0, 2.0 * R)),
        ("H", (0.0, 0.0, 3.0 * R)),
    ]


def build_jw_hamiltonian_for_R(R):
    """
    Build Jordan-Wigner transformed Hamiltonian for H4 at distance R.
    
    Parameters
    ----------
    R : float
        Bond distance in Angstroms.
        
    Returns
    -------
    H : QubitOperator
        Jordan-Wigner transformed Hamiltonian.
    mol : MolecularData
        OpenFermion molecule object with computed properties.
    """
    geom = build_geometry(R)
    mol = MolecularData(geom, BASIS_NAME, MULTIPLICITY, CHARGE)
    mol = run_pyscf(mol, run_scf=1, run_fci=1)
    H_mol = mol.get_molecular_hamiltonian()
    H_ferm = get_fermion_operator(H_mol)
    H = jordan_wigner(H_ferm)
    return H, mol


# =============================================================================
# QUBIT OPERATOR UTILITIES
# =============================================================================

def infer_n_qubits_from_qubitop(qubit_op):
    """
    Infer number of qubits as 1 + max index appearing in any term.
    
    Parameters
    ----------
    qubit_op : QubitOperator
        The qubit operator.
        
    Returns
    -------
    int
        Number of qubits (0 if operator is empty).
    """
    max_q = -1
    for term, _coeff in qubit_op.terms.items():
        for (q, _p) in term:
            if q > max_q:
                max_q = q
    return max_q + 1 if max_q >= 0 else 0


def truncate_qubitop(H, eps):
    """
    Truncate qubit operator by dropping terms with |coeff| < eps.
    
    Parameters
    ----------
    H : QubitOperator
        Input Hamiltonian.
    eps : float
        Truncation threshold.
        
    Returns
    -------
    QubitOperator
        Truncated Hamiltonian.
    """
    out = QubitOperator()
    for term, coeff in H.terms.items():
        if abs(coeff) >= eps:
            if abs(coeff.imag) < 1e-12:
                coeff = coeff.real
            out += QubitOperator(term, coeff)
    return out


# =============================================================================
# SYMPLECTIC / PAULI REPRESENTATION
# =============================================================================

def pauli_term_to_ax_az(term, n):
    """
    Convert OpenFermion term to symplectic (ax, az) representation.
    
    Parameters
    ----------
    term : tuple
        OpenFermion term key, e.g. ((0,'X'), (3,'Y')) or ().
    n : int
        Number of qubits.
        
    Returns
    -------
    ax : ndarray
        X-component binary vector of length n.
    az : ndarray
        Z-component binary vector of length n.
    """
    ax = np.zeros(n, dtype=np.uint8)
    az = np.zeros(n, dtype=np.uint8)

    for q, p in term:
        if p == 'X':
            ax[q] = 1
        elif p == 'Z':
            az[q] = 1
        elif p == 'Y':
            # Y = iXZ, both bits set (phase ignored for commutation)
            ax[q] = 1
            az[q] = 1
        else:
            raise ValueError(f"Unknown Pauli {p} in term {term}")

    return ax, az


def qubitop_to_G_matrix(qubit_op, n=None):
    """
    Build symplectic matrix G = (Gx | Gz) encoding Pauli terms.
    
    Parameters
    ----------
    qubit_op : QubitOperator
        Input qubit operator.
    n : int, optional
        Number of qubits. If None, inferred from operator.
        
    Returns
    -------
    G : ndarray
        Shape (m, 2n) symplectic matrix.
    coeffs : ndarray
        Complex coefficients of each term.
    labels : list
        String labels for each term.
    n : int
        Number of qubits used.
    """
    if n is None:
        n = infer_n_qubits_from_qubitop(qubit_op)

    items = list(qubit_op.terms.items())
    m = len(items)

    G = np.zeros((m, 2 * n), dtype=np.uint8)
    coeffs = np.zeros(m, dtype=np.complex128)
    labels = []

    for i, (term, coeff) in enumerate(items):
        ax, az = pauli_term_to_ax_az(term, n)
        G[i, :n] = ax
        G[i, n:] = az
        coeffs[i] = coeff

        if len(term) == 0:
            labels.append("I")
        else:
            labels.append(" ".join([f"{p}{q}" for (q, p) in term]))

    return G, coeffs, labels, n


def symplectic_to_pauli_string(s, n):
    """
    Convert symplectic vector to Pauli string representation.
    
    Parameters
    ----------
    s : ndarray
        Symplectic vector of length 2n (s_x | s_z).
    n : int
        Number of qubits.
        
    Returns
    -------
    str
        Pauli string like "Z0 Z2 X5" or "I" for identity.
    """
    sx = s[:n]
    sz = s[n:]

    ops = []
    for q in range(n):
        x = int(sx[q])
        z = int(sz[q])
        if x == 0 and z == 0:
            continue
        elif x == 1 and z == 0:
            ops.append(f"X{q}")
        elif x == 0 and z == 1:
            ops.append(f"Z{q}")
        else:  # x==1 and z==1
            ops.append(f"Y{q}")
    return " ".join(ops) if ops else "I"


# =============================================================================
# COMMUTATION CONSTRAINTS
# =============================================================================

def build_commutation_constraints_A(G, n):
    """
    Build constraint matrix A for commutation conditions.
    
    For each row g_i in G, the constraint row is (g_z,i | g_x,i) so that:
        (g_z,i | g_x,i) · (s_x | s_z) = 0 (mod 2)
    
    Parameters
    ----------
    G : ndarray
        Symplectic matrix (Gx | Gz) of shape (m, 2n).
    n : int
        Number of qubits.
        
    Returns
    -------
    A : ndarray
        Constraint matrix of shape (m, 2n).
    """
    Gx = G[:, :n]
    Gz = G[:, n:]
    A = np.concatenate([Gz, Gx], axis=1).astype(np.uint8)
    return A


# =============================================================================
# GF(2) LINEAR ALGEBRA
# =============================================================================

def gf2_rref(A):
    """
    Compute row-reduced echelon form over GF(2).
    
    Parameters
    ----------
    A : ndarray
        Input binary matrix.
        
    Returns
    -------
    R : ndarray
        RREF of A over GF(2).
    pivots : list
        List of pivot column indices.
    """
    R = (A.copy() & 1).astype(np.uint8)
    m, n = R.shape
    pivots = []
    r = 0

    for c in range(n):
        # Find pivot row
        pivot = None
        for rr in range(r, m):
            if R[rr, c] == 1:
                pivot = rr
                break
        if pivot is None:
            continue

        # Swap into row r
        if pivot != r:
            R[[r, pivot]] = R[[pivot, r]]

        pivots.append(c)

        # Eliminate all other 1s in column c
        for rr in range(m):
            if rr != r and R[rr, c] == 1:
                R[rr, :] ^= R[r, :]

        r += 1
        if r == m:
            break

    return R, pivots


def gf2_rank(M):
    """
    Compute rank of matrix M over GF(2).
    
    Parameters
    ----------
    M : ndarray
        Input binary matrix.
        
    Returns
    -------
    int
        Rank over GF(2).
    """
    R, piv = gf2_rref(M.astype(np.uint8))
    return len(piv)


def gf2_nullspace(A):
    """
    Compute basis for nullspace of A over GF(2): {x : Ax = 0 mod 2}.
    
    Parameters
    ----------
    A : ndarray
        Input binary matrix of shape (m, n).
        
    Returns
    -------
    basis : ndarray
        Shape (k, n) basis vectors (each row is a solution vector).
    """
    R, pivots = gf2_rref(A)
    m, n = R.shape
    pivset = set(pivots)
    free_cols = [c for c in range(n) if c not in pivset]

    basis = []
    for f in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1

        for i, p in enumerate(pivots):
            s = 0
            row = R[i, :]
            for j in range(n):
                if j != p and row[j] == 1 and x[j] == 1:
                    s ^= 1
            x[p] = s

        basis.append(x)

    if len(basis) == 0:
        return np.zeros((0, n), dtype=np.uint8)

    return np.stack(basis, axis=0)


def gf2_check_in_nullspace(A, S):
    """
    Check if all rows of S lie in nullspace of A over GF(2).
    
    Parameters
    ----------
    A : ndarray
        Constraint matrix of shape (m, 2n).
    S : ndarray
        Matrix of candidate vectors of shape (k, 2n).
        
    Returns
    -------
    bool
        True if A @ S^T == 0 mod 2 for all rows of S.
    """
    if S is None or S.size == 0:
        return True
    prod = (A.astype(np.uint8) @ S.T.astype(np.uint8)) & 1
    return np.all(prod == 0)


def gf2_extend_basis_additive(B_current, candidates):
    """
    Extend additive basis with linearly independent candidates.
    
    Given current additive basis B_current (k, 2n) and candidate vectors
    from the current nullspace, extend B_current by adding candidates
    that increase the GF(2) rank.
    
    Parameters
    ----------
    B_current : ndarray or None
        Current basis of shape (k, 2n).
    candidates : ndarray
        Candidate vectors of shape (c, 2n).
        
    Returns
    -------
    B_new : ndarray
        Extended basis.
    added_vectors : ndarray
        Vectors that were added.
    """
    if B_current is None or B_current.size == 0:
        B = np.zeros((0, candidates.shape[1]), dtype=np.uint8)
    else:
        B = B_current.copy().astype(np.uint8)

    added = []

    # Deterministic ordering via lexicographic sort
    cand = candidates.copy().astype(np.uint8)
    if cand.shape[0] > 0:
        order = np.lexsort(cand.T[::-1])
        cand = cand[order]

    r0 = gf2_rank(B) if B.shape[0] else 0
    for v in cand:
        if B.shape[0] == 0:
            B = v.reshape(1, -1).astype(np.uint8)
            added.append(v.copy())
            r0 = 1
            continue

        r1 = gf2_rank(np.vstack([B, v]).astype(np.uint8))
        if r1 > r0:
            B = np.vstack([B, v]).astype(np.uint8)
            added.append(v.copy())
            r0 = r1

    return B, np.array(added, dtype=np.uint8)


# =============================================================================
# MAIN SYMMETRY SWEEP
# =============================================================================

def symmetry_sweep_additive(H, num_intervals=100, eps_max=None, verbose=True, print_new=True):
    """
    Perform additive symmetry sweep over epsilon truncation thresholds.
    
    For each epsilon in grid, truncates H, builds constraint matrix A,
    finds nullspace basis, and maintains an additive (nested) basis
    across all epsilon values.
    
    Parameters
    ----------
    H : QubitOperator
        Full Hamiltonian.
    num_intervals : int
        Number of epsilon discretization points.
    eps_max : float, optional
        Maximum epsilon. If None, uses 1.000001 * max|coeff|.
    verbose : bool
        Print progress messages.
    print_new : bool
        Print when new symmetries are discovered.
        
    Returns
    -------
    df : pd.DataFrame
        Per-epsilon summary statistics.
    basis_additive_list : list
        List of additive bases (one per epsilon).
    basis_rref_list : list
        List of RREF bases (fresh basis per epsilon).
    """
    # Fix n once from original H
    n_fixed = infer_n_qubits_from_qubitop(H)

    # Epsilon grid
    max_abs = max((abs(c) for c in H.terms.values()), default=0.0)
    if eps_max is None:
        eps_max = max_abs * 1.000001

    eps_grid = np.linspace(0.0, eps_max, num_intervals)

    basis_add = np.zeros((0, 2 * n_fixed), dtype=np.uint8)
    basis_additive_list = []
    basis_rref_list = []
    rows = []
    prev_dim = 0

    for idx, eps in enumerate(eps_grid):
        # Truncate H
        Ht = truncate_qubitop(H, float(eps))

        # Build constraint matrix A from truncated H
        Gt, _, _, _ = qubitop_to_G_matrix(Ht, n=n_fixed)
        A = build_commutation_constraints_A(Gt, n_fixed)

        # Find symmetries = nullspace(A)
        basis = gf2_nullspace(A)
        basis_rref, piv = gf2_rref(basis)

        # Additive extension
        basis_add, added = gf2_extend_basis_additive(basis_add, basis_rref)

        # Sanity check: additive basis should lie in current nullspace
        ok_null = gf2_check_in_nullspace(A, basis_add)

        # Convert to strings
        add_strs = [symplectic_to_pauli_string(v, n_fixed) for v in added] if added.size else []
        basis_add_strs = [symplectic_to_pauli_string(v, n_fixed) for v in basis_add] if basis_add.size else []

        # Record results
        rows.append({
            "eps_idx": int(idx),
            "epsilon": float(eps),
            "terms_left": int(len(Ht.terms)),
            "k_nullspace": int(basis.shape[0]),
            "k_additive": int(basis_add.shape[0]),
            "new_added": int(added.shape[0]) if added.size else 0,
            "new_added_syms": " | ".join(add_strs) if add_strs else "",
            "additive_basis_syms": " || ".join(basis_add_strs) if basis_add_strs else "",
            "additive_in_nullspace?": bool(ok_null),
        })

        basis_additive_list.append(basis_add.copy())
        basis_rref_list.append(basis_rref.copy())

        # Print on changes
        if verbose and print_new and basis_add.shape[0] != prev_dim:
            print(f"[eps_idx={idx:4d}] ε={eps:.6g}  terms_left={len(Ht.terms):4d}  "
                  f"k_null={basis.shape[0]:2d}  k_additive={basis_add.shape[0]:2d}  "
                  f"(+{len(add_strs)})  ok_null={ok_null}")
            for s in add_strs:
                print("   +", s)
            prev_dim = basis_add.shape[0]

    df = pd.DataFrame(rows)
    return df, basis_additive_list, basis_rref_list


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run symmetry sweep over all R values."""
    all_rows = []
    basis_additive_by_R = {}

    for R in R_VALUES:
        print(f"\n{'='*20} R = {R} Å {'='*20}")
        
        H_R, mol = build_jw_hamiltonian_for_R(R)
        
        if EPS_POLICY == "per_R_max":
            max_abs = max((abs(c) for c in H_R.terms.values()), default=0.0)
            eps_max = 1.000001 * max_abs
        else:
            eps_max = None

        print(f"JW terms = {len(H_R.terms)}, eps_max = {eps_max:.6g}")

        df_sym, basis_add_list, _basis_rref_list = symmetry_sweep_additive(
            H_R,
            num_intervals=NUM_EPS_DIVISIONS,
            eps_max=eps_max,
            verbose=VERBOSE_PER_R,
            print_new=True,
        )

        # Attach metadata
        df_sym.insert(0, "R_Ang", float(R))
        df_sym["jw_terms_full"] = int(len(H_R.terms))
        df_sym["basis"] = BASIS_NAME
        df_sym["geometry"] = "H4 linear z: [0,R,2R,3R]"
        df_sym["fci_energy"] = float(getattr(mol, "fci_energy", np.nan))

        all_rows.append(df_sym)
        basis_additive_by_R[float(R)] = basis_add_list

        # Quick sanity print
        if PRINT_HEAD_ROWS > 0:
            print("\nHead (sanity):")
            print(df_sym.head(PRINT_HEAD_ROWS)[
                ["R_Ang", "eps_idx", "epsilon", "terms_left", 
                 "k_nullspace", "k_additive", "new_added", "additive_in_nullspace?"]
            ].to_string(index=False))

    # Combined dataframe
    df_R_eps_additive = pd.concat(all_rows, ignore_index=True)

    print(f"\nDONE. Combined df shape: {df_R_eps_additive.shape}")
    
    return df_R_eps_additive, basis_additive_by_R


if __name__ == "__main__":
    df_results, basis_data = main()
