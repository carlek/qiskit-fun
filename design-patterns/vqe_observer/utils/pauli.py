import re
from typing import List, Tuple, Optional, Dict

_DECIMAL = r"(?:\d+(?:\.\d*)?|\.\d+)?"
_TERM_RE = re.compile(
    rf"^(?:(?P<coef>{_DECIMAL})\*?)?(?P<body>(?:[IXYZixyz]\d+)*)$"
)
_FACTOR_RE = re.compile(r"([IXYZixyz])(\d+)")

def parse_pauli_observable(
    obs_str: str,
    *,
    num_qubits: Optional[int] = None, # set it explicitly else it assumes the highest specified qubit
    endianness: str = "qiskit",       # 'qiskit': H->L =  L->R. 'little': H:L = R->L
    ) -> List[Tuple[str, float]]:
    """
    Parse a compact algebraic Pauli observable string into Qiskit's SparsePauliOp-compatible
    (label, coefficient) tuples.

    Grammar (whitespace-insensitive):
        pauli_obs := { signed_term }
        signed_term := [ '+' | '-' ] term
        term := [ coeff ] [ '*' ] { factor }
        coeff := decimal
        factor := [I|X|Y|Z][0-9]+   (case-insensitive)

    Args:
        obs_str (str): The observable string to parse.
        num_qubits (Optional[int]): Optional fixed register width.
                                    If None, inferred from the max qubit index.
        endianness (str): 'qiskit' (default) - leftmost = highest qubit index;
                          'little' - leftmost = qubit 0.
        drop_tol (float): Coefficients below this threshold are discarded.

    Returns:
        List[Tuple[str, float]]: List of (Pauli string, coefficient) pairs for use with
                                 SparsePauliOp.from_list().
    Raises:
        ValueError: If the string is malformed or num_qubits cannot be inferred.

    Examples:
        Input  -> "2 I0 I1 - 2 X0 X1 + 3 Y0 Y1 - 3 Z0 Z1"
        Stripped -> "2I0I1-2X0X1+3Y0Y1-3Z0Z1"
        Output -> [("II", 2.0), ("XX", -2.0), ("YY", 3.0), ("ZZ", -3.0)]

    """
    
    # Normalize: remove whitespace, ensure leading sign
    s = obs_str.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
    if not s:
        raise ValueError("Empty observable string.")
    if s[0] not in "+-":
        s = "+" + s

    # Split into signed chunks at each +/-
    chunks = re.split(r"(?=[+-])", s)
    chunks = [c for c in chunks if c]  # drop empties

    terms: List[Tuple[float, Dict[int, str]]] = []
    max_idx = -1

    for chunk in chunks:
        sign = +1.0 if chunk[0] == "+" else -1.0
        body = chunk[1:]  # rest of the term after sign

        m = _TERM_RE.match(body)
        if not m:
            raise ValueError(f"Malformed term: '{chunk}'")

        coef_str = m.group("coef")
        coef = float(coef_str) if coef_str else 1.0
        coef *= sign

        factors_str = m.group("body")
        factors: Dict[int, str] = {}
        if factors_str:
            for p, q in _FACTOR_RE.findall(factors_str):
                qbit = int(q)
                pauli = p.upper()
                if qbit in factors:
                    raise ValueError(f"Duplicate factor for qubit {qbit} in term '{chunk}'.")
                factors[qbit] = pauli
                max_idx = max(max_idx, qbit)

        terms.append((coef, factors))

    # Determine register width
    N = num_qubits if num_qubits is not None else (max_idx + 1 if max_idx >= 0 else None)
    if N is None:
        raise ValueError("Cannot infer num_qubits from a pure-identity observable. Provide num_qubits.")
    if N <= 0:
        raise ValueError("num_qubits must be positive.")

    # Map qubit index -> label position
    def pos(q: int) -> int:
        return (N - 1 - q) if endianness == "qiskit" else q

    # Build labels and combine like terms
    accum: Dict[str, float] = {}
    for coef, term in terms:
        label = ["I"] * N
        for q, p in term.items():
            qbitpos= pos(q)
            if not (0 <= qbitpos < N):
                raise ValueError(f"Qubit index {q} out of range for num_qubits={N}.")
            if p != "I":
                label[qbitpos] = p
        key = "".join(label)
        accum[key] = coef

    items = [(lab, c) for lab, c in sorted(accum.items())]
    return items
