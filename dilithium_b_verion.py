from simplified_kyber_pke import *
from dilithium_toy_version import *
from hashlib import shake_256
import secrets

n = 256
q = 2**14
lambda_p = 256
k = 4
l = 3
gamma1 = 2**19
gamma2 = 2**18
tau = 60
eta = 2
Beta = tau*eta

def generate_xi():
    return secrets.token_bytes(32)  # 256-bit seed (xi)

def shake_hash(data: bytes, outlen_bits: int) -> bytes:
    """Helper: SHAKE-256 based hashing to produce outlen_bits bits."""
    return shake_256(data).digest(outlen_bits // 8)

def compute_p_p_prime_K(xi: bytes):
    """
    Compute p, p', K from xi:
    H(xi, 1024) -> 1024 bits = 128 bytes total
    p: first 32 bytes (256 bits)
    p_prime: next 64 bytes (512 bits)
    K: last 32 bytes (256 bits)
    """
    full_hash = shake_hash(xi, 1024)
    p = full_hash[0:32]
    p_prime = full_hash[32:96]
    K = full_hash[96:128]
    return p, p_prime, K

def sample_uniform_polynomial(q, n, seed: bytes, nonce: int) -> list:
    """
    Generate a polynomial with coefficients uniformly in [0, q-1] 
    from a seed and a nonce using SHAKE-256.
    """
    # Encode nonce in 4 bytes
    input_data = seed + nonce.to_bytes(4, 'little')
    stream = shake_256(input_data)
    poly = []
    needed_bytes = (n * 2)  # since q = 2^14, we need 14 bits. 2 bytes can hold up to 16 bits.
    rand_bytes = stream.digest(needed_bytes)
    # Extract each coefficient
    # q=2^14 means we just take 14 bits from each 16-bit chunk
    # We'll just mod the 16-bit value by q
    for i in range(n):
        val = (rand_bytes[2*i] << 8) | rand_bytes[2*i+1]
        val = val % q
        poly.append(val)
    return poly

def sample_small_polynomial(n, eta, seed: bytes, nonce: int) -> list:
    """
    Generate a small polynomial with coefficients in [-eta, eta].
    Use SHAKE to get random bytes and map them to the desired range.
    """
    input_data = seed + nonce.to_bytes(4, 'little')
    stream = shake_256(input_data)
    poly = []
    # To get a coefficient in [-eta, eta], we can map bytes to an integer in that range.
    # The range size is (2*eta + 1).
    range_size = 2*eta + 1
    # Each byte can be reduced mod (2*eta+1).
    needed_bytes = n  # one byte per coefficient should be fine since eta small
    rand_bytes = stream.digest(needed_bytes)
    for i in range(n):
        val = rand_bytes[i] % range_size
        val = val - eta
        poly.append(val)
    return poly

def expandA(q, k, l, n, p: bytes):
    """
    Construct A matrix deterministically from seed p.
    A is k x l, each entry is a polynomial of degree n.
    We will use a simple counter for each polynomial.
    """
    A = []
    counter = 0
    for i in range(k):
        row = []
        for j in range(l):
            poly = sample_uniform_polynomial(q, n, p, counter)
            row.append(poly)
            counter += 1
        A.append(row)
    return A

def expandS(k, l, n, eta, p_prime: bytes):
    """
    Expand p_prime into two secret vectors s1 and s2.
    s1: l polynomials, s2: k polynomials.
    """
    s1 = []
    for i in range(l):
        s1.append(sample_small_polynomial(n, eta, p_prime, i))
    s2 = []
    for j in range(k):
        s2.append(sample_small_polynomial(n, eta, p_prime, l + j))
    return s1, s2

def compute_t_vector(A_matrix, s1_vector, s2_vector, q):
    """
    Compute t = A*s1 + s2
    """
    a_times_s1 = multiply_matrix_by_vector(A_matrix, s1_vector, q)
    t = add_vector_to_vector(a_times_s1, s2_vector, q)
    return t

def compute_tr(p_seed: bytes, t_vector, parameter_lambda):
    """
    tr = H(p_seed || t_vector, 2*lambda)
    Serialize t_vector and hash.
    """
    # Serialize t_vector coefficients as unsigned
    t_bytes = b"".join(
        (coeff % q).to_bytes(2, 'big', signed=False)  
        for poly in t_vector for coeff in poly
    )
    concatenated = p_seed + t_bytes
    # produce a 2*lambda = 512 bits hash if lambda=256
    tr = shake_256(concatenated).digest((2 * parameter_lambda) // 8)
    return tr

def compute_mu(tr: bytes, message: bytes):
    """
    mu = H(tr || M, 512 bits)
    """
    concatenated = tr + message
    mu = shake_256(concatenated).digest(64)  # 512 bits
    return mu

def compute_p_prime_prime(K: bytes, rnd: bytes, mu: bytes):
    """
    p'' = H(K || rnd || mu, 512 bits)
    """
    concatenated = K + rnd + mu
    p_prime_prime = shake_256(concatenated).digest(64)  # 512 bits
    return p_prime_prime

def generate_deterministic_rnd():
    return b'\x00' * 32

def expand_mask_with_nonce(seed: bytes, kappa, l, gamma1):
    """
    y vector with length l from seed, nonce=kappa.
    """
    input_data = seed + kappa.to_bytes(4, 'big')
    stream = shake_256(input_data)
    y = []
    for _ in range(l):
        rand_bytes = stream.digest(4)
        rand_int = int.from_bytes(rand_bytes, 'big')
        mapped_int = (rand_int % (2 * gamma1 + 1)) - gamma1 - 1
        y.append(mapped_int)
    return y

def generate_keys(q, k, l, n, lambda_param):
    xi = generate_xi()
    p, p_prime, K = compute_p_p_prime_K(xi)
    A_matrix = expandA(q, k, l, n, p)
    s1_vec, s2_vec = expandS(k, l, n, eta, p_prime)
    t_vector = compute_t_vector(A_matrix, s1_vec, s2_vec, q)
    trace_tr = compute_tr(p, t_vector, lambda_param)
    return p, t_vector, K, trace_tr, s1_vec, s2_vec

def generate_signature(message, q, k, l, n, p, tr, K):
    matrix_A = expandA(q, k, l, n, p)
    mu = compute_mu(tr, message)
    deterministic_rnd = generate_deterministic_rnd()
    rho_second = compute_p_prime_prime(K, deterministic_rnd, mu)
    kappa = 0
    found = False
    while not found:
        y_vector = expand_mask_with_nonce(rho_second, kappa, l, gamma1)
        w = multiply_matrix_by_vector(matrix_A, y_vector, q)
        # Further steps for Dilithium signature generation would go here.
        # For now, we just show that we've fixed the seeding/hashing.
        # Implement the rest of the signature steps as per the Dilithium spec.
        found = True  # Placeholder to prevent infinite loop

    return (y_vector, w)  # Placeholder return for demonstration
