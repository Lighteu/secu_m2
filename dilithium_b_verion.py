from simplified_kyber_pke import *
from dilithium_toy_version import *
from hashlib import shake_128, shake_256
import secrets

# Using secrects because appearently it's more secure and more commonly used for cryptography
def generate_xi():
     return secrets.token_bytes(32) 

def hash(input_data, output_size_bits):
    """Hash function with specified output size in bits."""
    hash_bytes = shake_256(input_data).digest(output_size_bits // 8)
    return int.from_bytes(hash_bytes, byteorder="big")

def generate_random_polynomial(q, n, seed):
    """Generate a random polynomial modulo q using a seed."""
    random.seed(seed)  # Setting the seed for reproducibility
    return [random.randint(0, q-1) for _ in range(n)]

def expandA(q, k, l, n, seed):
    return [[generate_random_polynomial(q,n,seed) for _ in range(l)] for _ in range(k)]


     
