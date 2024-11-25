import hashlib
import random 

def shake128(input_data, output_length):
    shake = hashlib.shake_128()
    shake.update(input_data)
    return list(shake.digest(output_length))  # Convert the output to a list of integers

def generate_random_polynomial(seed, row, col, degree, q):
    # Unique input for each polynomial based on seed, row, and column
    input_data = (seed + f"{row},{col}").encode('utf-8')
    # Generate enough pseudo-random bytes for the coefficients
    hash_output = shake128(input_data, (degree + 1) * 2)
    # Convert hash output into coefficients (2 bytes per coefficient)
    coefficients = []
    for i in range(degree + 1):
        value = (hash_output[2 * i] << 8) + hash_output[2 * i + 1]  # Combine 2 bytes
        coefficients.append(value % q)  # Reduce mod q
    return coefficients

def generate_matrix_A(n, degree, q, seed):
    matrix = []
    for row in range(n):
        matrix_row = []
        for col in range(n):
            polynomial = generate_random_polynomial(seed, row, col, degree, q)
            matrix_row.append(polynomial)
        matrix.append(matrix_row)
    return matrix

def compress_coefficients(coefficients, q, k):
    compressed = []
    for coeff in coefficients:
        # Scale down the coefficient to the range [0, 2^k)
        compressed_coeff = round((coeff * (2**k)) / q) % (2**k)
        compressed.append(compressed_coeff)
    return compressed


def decompress_coefficients(compressed, q, k):
    decompressed = []
    for compressed_coeff in compressed:
        # Scale back the coefficient to the range [0, q)
        decompressed_coeff = round((compressed_coeff * q) / (2**k)) % q
        decompressed.append(decompressed_coeff)
    return decompressed

def generate_random_small_polynomial(n, eta):
    def cbd_sample(eta):
        b = [random.randint(0, 1) for _ in range(eta)]  # Random bits for sum b
        b_prime = [random.randint(0, 1) for _ in range(eta)]  # Random bits for sum b'
        return sum(b) - sum(b_prime)  # CBD value is the difference of sums

    # Generate n coefficients using CBD(eta)
    return [cbd_sample(eta) for _ in range(n)]


n = 10   # Number of coefficients in the polynomial
eta = 2  # CBD parameter
random_small_polynomial = generate_random_small_polynomial(n, eta)
print("Random small polynomial:", random_small_polynomial)

exit()
# Example usage
coefficients = [1000, 2000, 3000, 4000, 5000, 6000]  # Example coefficients
q = 3329  # Kyber modulus
k = 10    # Target bit-depth (e.g., compress to 10 bits)

# Compress the coefficients
compressed_coeffs = compress_coefficients(coefficients, q, k)
print("Original coefficients:", coefficients)
print("Compressed coefficients:", compressed_coeffs)

# Decompress the coefficients
decompressed_coeffs = decompress_coefficients(compressed_coeffs, q, k)
print("Decompressed coefficients:", decompressed_coeffs)
exit()
# Example usage
n = 2          # Matrix dimension (2x2 for simplicity)
degree = 2     # Degree of the polynomials (e.g., 2 -> 3 coefficients per polynomial)
q = 3329       # Modulus for coefficients
seed = "example_seed"  # Seed for randomness

matrix_A = generate_matrix_A(n, degree, q, seed)

# Print the matrix
print("Matrix A (polynomials as coefficient vectors):")
for row in matrix_A:
    print(row)
