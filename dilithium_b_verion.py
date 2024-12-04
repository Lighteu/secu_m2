from simplified_kyber_pke import *
from dilithium_toy_version import *
from hashlib import shake_128, shake_256
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

# Using secrects because appearently it's more secure and more commonly used for cryptography
def generate_xi():
     return secrets.token_bytes(32) 


def hash(input_data, output_size_bits):
    """Hash function with specified output size in bits."""
    hash_bytes = shake_256(input_data).digest(output_size_bits // 8)
    return int.from_bytes(hash_bytes, byteorder="big")


def split_hash(hash_value, bit_sizes):
    """
    Split a large hash value into multiple parts based on bit sizes.

    Parameters:
    - hash_value: The large hash value as an integer.
    - bit_sizes: A list of sizes (in bits) for each part.

    Returns:
    - A tuple of integers, each representing a portion of the hash.
    """
    parts = []
    shift = sum(bit_sizes)  # Total size of the hash
    for size in bit_sizes:
        shift -= size
        part = (hash_value >> shift) & ((1 << size) - 1)
        parts.append(part)
    return tuple(parts)

def compute_p_p_prime_K(xi):
    """
    Compute (p, p', K) = H(ξ, 1024) and split the result.

    Parameters:
    - xi: The 256-bit seed (as bytes).

    Returns:
    - p: 256-bit integer
    - p_prime: 512-bit integer
    - K: 256-bit integer
    """
    # Step 1: Compute H(ξ, 1024)
    hash_value = hash(xi, 1024)  # Produces a 1024-bit integer
    
    # Step 2: Split the hash into (p, p', K)
    p, p_prime, K = split_hash(hash_value, [256, 512, 256])
    
    return p, p_prime, K


def generate_random_polynomial(q, n, seed):
    """Generate a random polynomial modulo q using a seed."""
    random.seed(seed)  # Setting the seed for reproducibility
    return [random.randint(0, q-1) for _ in range(n)]


def generate_random_small_polynomial(n, eta, seed):
    """Generate a random small polynomial between -eta and eta using a seed."""
    random.seed(seed)  # Setting the seed for reproducibility
    return [ randint(-eta, eta) for _ in range(n) ]
    
def generate_A(q, k, n):
    return [[generate_random_polynomial(q,n) for _ in range(k)] for _ in range(k)]
    

def expandA(q, k, l, n, p):
    """Generates the A matrix composed of random polynomial modulo q using a seed p."""
    return [[generate_random_polynomial(q,n,p) for _ in range(l)] for _ in range(k)]


def generate_small_polynomial_vector(k, n, eta, seed):
    """
    Generate a vector of k small polynomials, each of degree n, with coefficients in [-eta, eta].
    
    Parameters:
    - k: Number of polynomials in the vector.
    - n: Degree of each polynomial.
    - eta: Range for polynomial coefficients.
    - seed: Seed to ensure deterministic generation.
    
    Returns:
    - List of k polynomials.
    """
    vector = []
    for i in range(k):
        # Derive a unique seed for each polynomial in the vector
        polynomial_seed = hash((seed, i)) & 0xFFFFFFFF
        vector.append(generate_random_small_polynomial(n, eta, polynomial_seed))
    return vector


def expandS(k, l, n, eta, p_prime):
    """
    Expand p_prime into two secret vectors s1 and s2.
    
    Parameters:
    - k: Dimension of s2 (vector length).
    - l: Dimension of s1 (vector length).
    - n: Degree of polynomials in the vectors.
    - eta: Range for polynomial coefficients.
    - p_prime: Seed to derive s1 and s2.
    
    Returns:
    - s1: Vector of l small polynomials.
    - s2: Vector of k small polynomials.
    """
    # Derive two seeds from p_prime for generating s1 and s2
    seed_s1 = hash((p_prime, "s1")) & 0xFFFFFFFF
    seed_s2 = hash((p_prime, "s2")) & 0xFFFFFFFF
    
    # Generate s1 and s2
    s1 = generate_small_polynomial_vector(l, n, eta, seed_s1)
    s2 = generate_small_polynomial_vector(k, n, eta, seed_s2)
    return s1, s2


def compute_t_vector(A_matrix, s1_vector, s2_vector, q):
    """
    Compute the public vector t.
    
    Parameters:
    - A_matrix: The generated A matrix of polynomials.
    - s1_vector: s1 vector of small polynomials.
    - s2_vector: s2 vector of small polynomials.
    - q: Range for polynomial coefficients.
    
    Returns:
    - t: Public vector t.
    """
    a_times_s1 = multiply_matrix_by_vector(A_matrix, s1_vector, q)
    t = add_vector_to_vector(a_times_s1, s2_vector, q)
    return t

def compute_mu(tr, message):
    """
    Compute μ = H(tr || M, 512), where H is SHAKE-256.
    
    Parameters:
    - tr: Trace value (bytes).
    - message: Message to be signed (bytes).
    
    Returns:
    - μ: 512-bit hash output as bytes.
    """
    # Concatenate the trace and the message
    concatenated = tr + message
    
    # Hash using SHAKE-256 to produce a 512-bit output
    mu = shake_256(concatenated).digest(512 // 8)  # 512 bits = 64 bytes
    
    return mu

def compute_tr(p_seed, t_vector, parameter_lambda):
    """
    Compute the trace vector tr, composed of the concatenation of p_seed (rho) and the t_vector, producing a trace of length 2 times lambda.
    
    Parameters:
    - p_seed: The generated rho seed.
    - t_vector: Public vector t.
    - parameter_lambda: Lambda value used to produce tr of length 2*lambda.
    
    Returns:
    - tr: Concatenation of rho and the t vector of length 2*lambda.
    """
    # Step 1: Serialize the t_vector into a byte string
    t_bytes = b"".join(int.to_bytes(coeff, length=4, byteorder="big", signed=True) 
                       for poly in t_vector for coeff in poly)
    
    # Step 2: Concatenate p_seed and the serialized t_vector bytes
    concatenated = p_seed + t_bytes
    
    # Step 3: Hash the concatenated value to derive tr
    hash_output = shake_256(concatenated).digest((2 * parameter_lambda) // 8)
    
    return hash_output

def compute_p_prime_prime(K, rnd, mu):
    """
    Compute p'' = H(K || rnd || μ, 512) deterministically using SHAKE-256.

    Parameters:
    - K: Key (bytes).
    - rnd: Random value (bytes, set to 256 zero bits for deterministic signing).
    - mu: Message digest μ (bytes).

    Returns:
    - p'': 512-bit hash output as bytes.
    """
    # Concatenate K, rnd, and μ
    concatenated = K + rnd + mu
    
    # Hash using SHAKE-256 to produce a 512-bit output
    p_prime_prime = shake_256(concatenated).digest(512 // 8)  # 512 bits = 64 bytes
    
    return p_prime_prime


def generate_deterministic_rnd():
    """Generate a deterministic rnd value for signing."""
    return b'\x00' * 32  # 256 bits (32 bytes) of zeros

def expand_mask_with_nonce(seed, kappa, l, gamma1):
    """
    Generate a vector y with length l using the seed and nonce.
    
    Parameters:
    - seed: The seed (rho_0) as bytes.
    - kappa: Nonce to ensure distinct randomness.
    - l: Length of the vector y.
    - q: Modulus to apply to each component of y.
    
    Returns:
    - A list of l components polynomials of the vector y.
    """
    # Combine seed and nonce to make the input unique for each run
    input_data = seed + kappa.to_bytes(4, 'big')  # Assuming kappa fits in 4 bytes
    
    # Use SHAKE-256 for flexible-length output
    shake = shake_256(input_data)
    
    y = []
    for _ in range(l):
        # Generate an integer value for each component of y
        rand_bytes = shake.digest(4)  # 4 bytes (32 bits) for each value
        rand_int = int.from_bytes(rand_bytes, byteorder='big')
        mapped_int = (rand_int % (2 * gamma1 + 1)) - gamma1 - 1
        
        # Append the value to the vector y
        y.append(mapped_int)
    
    return y

def generate_keys(q, k, l, n, p, lambda_param):
    xi = generate_xi()
    p, p_prime, K = compute_p_p_prime_K(xi)
    A_matrix = expandA(q,k,l,n,p)
    s1_vec, s2_vec = expandS(k,l,n,eta,p_prime)
    t_vector = compute_t_vector(A_matrix,s1_vec,s2_vec,q)
    trace_tr = compute_tr(p,t_vector,lambda_param)
    return p, t_vector, K, trace_tr, s1_vec, s2_vec


def generate_signature(message, q, k, l, n, p, tr, K):
    matrix_A = expandA(q, k, l, n, p)
    mu = compute_mu(tr, message)
    deterministic_rnd = generate_deterministic_rnd()
    rho_second = compute_p_prime_prime(K, deterministic_rnd, mu)
    kappa = 0
    found = False
    while(not found):
        y_vector = expand_mask_with_nonce(rho_second,kappa,l,gamma1)
        w =  multiply_matrix_by_vector(matrix_A,y_vector, q)
        