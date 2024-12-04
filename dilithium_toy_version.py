from hashlib import shake_128, shake_256
from random import randint
from simplified_kyber_pke import generate_random_polynomial, visualize_polynomial_matrix, generate_small_polynomial_vector, add_vector_to_vector, multiply_matrix_by_vector, string_to_binary_vector, polynomial_subtraction, polynomial_subtraction
import random
import numpy as np


n = 256, q = 2**14, eta=2
def generate_A(q, k, l, n):
    return [[generate_random_polynomial(q,n) for _ in range(l)] for _ in range(k)]


def generate_random_polynomial_with_bound(n, bound):
    """
    Generates a polynomial of degree n with coefficients in [-bound excluded, bound.
    """
    return [randint(-bound +1, bound) for _ in range(n)]


def generate_y_vector(gamma1, l, n):
    """
    Generates a vector y of length l, where each entry is a polynomial 
    with coefficients in [-gamma1+1, gamma1].
    """
    return [generate_random_polynomial_with_bound(n, gamma1) for _ in range(l)]

def high_bits(w, gamma2, q):
    """
    Extract the high bits of the coefficients of vector w.

    Parameters:
    - w: Vector of polynomials (list of lists of coefficients).
    - tau: Determines the bit shift (high bits are extracted based on 2^tau).
    - q: Modulus.

    Returns:
    - Vector of polynomials with high bits extracted.
    """
      # Compute 2^tau
    high_bits_vector = []

    for poly in w:  # Iterate through each polynomial in the vector
        high_bits_poly = [(coef % q) // gamma2 for coef in poly]  # Compute high bits for each coefficient
        high_bits_vector.append(high_bits_poly)

    return high_bits_vector

def low_bits(vector, gamma_2):
    """
    Extracts the low bits of a vector of lists.
    
    Args:
        vector (list of lists): The input vector of coefficients, where each element is a list.
        gamma_2 (int): The modulus for extracting low bits.
    
    Returns:
        list of lists: A vector with the low bits of each coefficient in each sublist.
    """
    return [[x % gamma_2 for x in inner_list] for inner_list in vector]


def flatten_vector(v):
    """
    Flatten a nested list structure into a single list.
    This is used for w1 if it contains polynomials or nested lists.
    """
    return [item for sublist in v for item in sublist]

def concatenate_vectors(M, w1):
    """
    Concatenate a binary message vector M with the high bits vector w1.
    
    Parameters:
    - M: Binary vector representing the message.
    - w1: Vector of high bits (may need flattening if nested).
    
    Returns:
    - Concatenated vector.
    """
    # Flatten w1 if it contains nested lists (e.g., polynomials)
    flattened_w1 = flatten_vector(w1)
    
    # Concatenate the two vectors
    return M + flattened_w1



def subtract_vectors(vector1, vector2, q):
    # Ensure both vectors have the same length before subtracting
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length")
    result = []
    for i in range(len(vector1)):
            result.append(
                polynomial_subtraction(vector1[i], vector2[i], q)
            )
    return result




def generate_challenge_polynomial(hash_output, n, tau):
    """
    Generate a ternary polynomial c with tau non-zero coefficients.

    Args:
        hash_output (bytes): The output of a hash function (e.g., shake_128).
        n (int): Total number of coefficients in the polynomial.
        tau (int): Number of non-zero coefficients.

    Returns:
        list: A ternary polynomial with n coefficients.
    """
    # Initialize all coefficients to 0
    c = [0] * n
    positions = set()  # To avoid duplicate positions

    # Convert hash output to a deterministic random bit stream
    random.seed(int.from_bytes(hash_output, byteorder="big"))

    # Assign non-zero values to tau random positions
    while len(positions) < tau:
        pos = random.randint(0, n - 1)  # Random position in the range [0, n-1]
        if pos not in positions:
            positions.add(pos)
            # Assign +1 or -1 randomly
            c[pos] = random.choice([-1, 1])

    return c

    
def multiply_vector_by_scalar(vector, scalar, modulus):
    if isinstance(vector[0], list):  # Check if it's a list of lists
        return [[(x * scalar) % modulus for x in inner_vector] for inner_vector in vector]
    else:  # Handle the case where it's a flat list
        return [(x * scalar) % modulus for x in vector]


def are_low_bits_sufficiently_small(low_bits, threshold):
    """
    Checks if all elements in the low bits vector are below a specified threshold.
    Assumes low_bits is a list of lists.
    
    Args:
        low_bits (list of lists): The vector of low bits (e.g., results of LowBits function).
        threshold (int): The maximum allowed value for the low bits.
    
    Returns:
        bool: True if all elements are below the threshold, False otherwise.
    """
    return all(x < threshold for inner_list in low_bits for x in inner_list)



def multiply_polynomial_with_vector(c, s1, q):
    """
    Multiply a ternary polynomial c with a vector of polynomials s1.

    Args:
        c (list): A single polynomial of size n (ternary coefficients: -1, 0, or +1).
        s1 (list of lists): A vector of polynomials (each of size n).
        q (int): Modulus for the operations.

    Returns:
        list of lists: A new vector of polynomials resulting from the multiplication.
    """
    result = []
    for poly in s1:
        convolved = np.convolve(c, poly)
        # Reduce modulo q and truncate to original size
        truncated = np.mod(convolved[:len(c)], q)
        # Convert to pure Python list
        result.append(truncated.tolist())
    return result

def compute_w_minus_cs2(w, cs2_product, q):
    # Assuming w and cs2 are lists of lists of coefficients
    result = []
    for i in range(len(w)):  # Loop over the polynomials (or parts of the polynomial)
        w_poly = w[i]       # Polynomial from w
        cs2_poly = cs2_product[i]   # Corresponding polynomial from cs2
        # Perform polynomial subtraction for each pair of polynomials
        result.append(polynomial_subtraction(w_poly, cs2_poly, q))
    return result
    


def generate_keys():
    # Creating our A matrix, containing polynomials that are randomly created 
    k = 4
    l = 3
    
    n = 256
    q = 2**14
    eta = 2
    # gamma2 = (int)(q - 1) / 32
    A_matrix = generate_A(q, k, l, n)
    # visualize_polynomial_matrix(A_matrix)

    s1_vector = generate_small_polynomial_vector(l,n,eta)
    s2_vector = generate_small_polynomial_vector(k,n,eta)

    t = add_vector_to_vector(multiply_matrix_by_vector(A_matrix, s1_vector, q), s2_vector, q)
    public_key = (A_matrix,t)
    private_key = (s1_vector, s2_vector)
    return public_key, private_key

def generate_signature(message, A_matrix, s1, s2, t, gamma):
    found = False
    bin_message = string_to_binary_vector(message)
    while(not found):
        y_vector = generate_y_vector(gamma,3,n)
        w = multiply_matrix_by_vector(A_matrix,y_vector, q)
        w1 = high_bits(w, gamma, q)


        hash_input = bytes(concatenate_vectors(bin_message, w1))
        hash_output = shake_128(hash_input).digest(32)  # Produces 256 bits of output
        c = generate_challenge_polynomial(hash_output, n=256, tau=60)
        # c = shake_128(bytes(concatenate_vectors(bin_message,w1))).digest(32) # Digest gives us an output of 32 bytes, equivalent to 256 bits
        # c_int = int.from_bytes(c, byteorder="big")
        # challenge_c = c_int % q
        c_times_s1 = multiply_polynomial_with_vector(c,s1,q)
        z = add_vector_to_vector(y_vector, c_times_s1,q)
        # print(f"S1 is {s1}\n\n")
        print(f"C is {c}\n\n")
        # print(f"C X S1 is {c_times_s1}\n\n")
        c_times_s2 = multiply_polynomial_with_vector(c,s2,q)
        # print(f"C X S2 is {c_times_s2}\n\n")
        # print(f"W IS {w}\n\n")
        w_minus_cs2 = compute_w_minus_cs2(w, c_times_s2, q)
        # print(f"w_minus_cs2 IS {w_minus_cs2}\n\n")
        # print(f"Size of challenge : {len(w)}")
        # print(f"Size of s2: {len(s2)}")
        # print(f"Size of cs2: {len(cs2)}")
        # print(f"Size of wminuscs2: {len(w_minus_cs2)}")
        # print(w_minus_cs2)
        verif = high_bits(w_minus_cs2, gamma, q)
        print(f"VERIF OF w-cs2 IIIIIS {verif}\n\n\n")
        low_bits_vector = low_bits(w_minus_cs2,gamma)
        # print(f"low_bits_vector is {low_bits_vector}\n\n")
        if(are_low_bits_sufficiently_small(low_bits_vector, gamma)):
            found = True
    signature = (c,z)
    # Az = multiply_matrix_by_vector(A_matrix, z, q)
    # c_times_t = multiply_vector_by_scalar(t, challenge_c, q)
    # Az_ct = subtract_vectors(Az, c_times_t, q)
    # w1_prime = high_bits(Az_ct, gamma, q)
    # c2 = shake_128(bytes(concatenate_vectors(bin_message,w1_prime))).digest(32) # Digest gives us an output of 32 bytes, equivalent to 256 bits
    # c2_int = int.from_bytes(c2, byteorder="big")
    # challenge2_c = c2_int % q
    # print(f"THE CHALLENGE 2 C in GENERATE IS {challenge2_c}")
    return signature


def verify_signature(message, signature, A_matrix, t, q, bit_count=1026):
    """
    Verifies a Dilithium signature by computing w1' = HighBits(Az - ct).
    
    Args:
        A_matrix (list): Public matrix A.
        z (list): The vector z from the signature.
        c (int): The challenge value.
        t (list): The vector t from the signature.
        q (int): The modulus for the operations.
        bit_count (i10): The number of high bits to retain.
    
    Returns:
        list: The high bits of the verification vector w1'.
    """
    # Step 1: Compute Az - c * t
    c,z = signature

    Az = multiply_matrix_by_vector(A_matrix, z, q)
    c_times_t = multiply_polynomial_with_vector(c, t, q)
    Az_ct = subtract_vectors(Az, c_times_t, q)
    w1_prime = high_bits(Az_ct, bit_count, q)
    print(f"AND W1 PRIME IIIIIS {w1_prime}\n\n")
    bin_message = string_to_binary_vector(message)

    # Step 2: Apply HighBits to the resulting vector
    hash_input = bytes(concatenate_vectors(bin_message, w1_prime))
    hash_output = shake_128(hash_input).digest(32)  # Produces 256 bits of output
    c2 = generate_challenge_polynomial(hash_output, n=256, tau=60)
    
    print(f"IN VERIF, C IIIIS {c}\n\n\n")
    print(f"IN VERIF, C2 IIIIS {c2}\n\n\n")
    


if __name__ == "__main__":
    public_k,private_k = generate_keys()
    message_signature = generate_signature("hi", public_k[0], private_k[0], private_k[1], public_k[1], gamma=1026)
    signature_verif = verify_signature("hi", message_signature, public_k[0], public_k[1], q)