from random import randint

def format_polynomial(coefficients):
    """
    Format a polynomial from a list of coefficients into a readable string.
    For example, [3, 2, 1] -> '3 + 2x + x^2'.
    
    Parameters:
        coefficients (list): List of polynomial coefficients.
    
    Returns:
        str: String representation of the polynomial.
    """
    terms = []
    for i, coeff in enumerate(coefficients):
        if coeff != 0:
            if i == 0:
                terms.append(f"{coeff}")
            elif i == 1:
                terms.append(f"{coeff}x" if coeff != 1 else "x")
            else:
                terms.append(f"{coeff}x^{i}" if coeff != 1 else f"x^{i}")
    return " + ".join(terms) if terms else "0"

def visualize_polynomial_matrix(matrix):
    """
    Visualize a matrix of polynomials by formatting each polynomial as a string.
    
    Parameters:
        matrix (list of list of list): A matrix where each element is a list of polynomial coefficients.
    
    Returns:
        None: Prints the formatted matrix.
    """
    formatted_matrix = []
    for row in matrix:
        formatted_row = [format_polynomial(poly) for poly in row]
        formatted_matrix.append(formatted_row)
    
    # Display the matrix
    for row in formatted_matrix:
        print(" | ".join(row))

def string_to_binary_vector(s):
    """
    Transforms a string into a binary vector of 1s and 0s, supporting Unicode characters.

    :param s: The input string
    :return: A list of 1s and 0s representing the binary encoding of the string
    """
    binary_vector = []
    for char in s:
        # Convert each character to its UTF-8 binary representation
        binary_char = ''.join(format(byte, '08b') for byte in char.encode('utf-8'))
        binary_vector.extend([int(bit) for bit in binary_char])
    return binary_vector


def binary_vector_to_string(binary_vector):
    """
    Transforms a binary vector of 1s and 0s back into a string, supporting Unicode characters.
    If a character cannot be decoded, it is replaced with '*'.

    :param binary_vector: The input binary vector (list of 1s and 0s)
    :return: The reconstructed string
    """
    if len(binary_vector) % 8 != 0:
        raise ValueError("Binary vector length must be a multiple of 8.")

    result = []
    for i in range(0, len(binary_vector), 8):
        try:
            byte = int(''.join(map(str, binary_vector[i:i+8])), 2)
            result.append(chr(byte))
        except Exception:
            result.append('*')
    
    return ''.join(result)

def transpose_matrix(matrix):
    """
    Computes the transpose of a given matrix.

    Args:
        matrix (list of lists): The matrix to transpose.

    Returns:
        list of lists: The transposed matrix.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def transpose_vector_of_polynomials(vector):
    """
    Computes the transpose of a vector of polynomials.

    Args:
        vector (list): The vector of polynomials, represented as a row (1D list) or a column (2D list).

    Returns:
        list: The transposed vector, switching between row and column representation.
    """
    # Row vector (1D list) -> Column vector (2D list)
    if isinstance(vector[0], list) and isinstance(vector[0][0], (int, float)):
        return [[v] for v in vector]
    # Column vector (2D list) -> Row vector (1D list)
    else:
        return [v[0] for v in vector]



def polynomial_multiplication(P, Q, q):
    n = len(P)
    result = [0] * n
    for i in range(n):
        for j in range(n):
            index = (i + j) % n
            sign = -1 if (i + j) >= n else 1
            result[index] = (result[index] + sign * P[i] * Q[j]) % q
    return result

def polynomial_addition(P, Q, q):
    result = []
    for i in range(len(P)):
        result.append((P[i] + Q[i]) % q)
    return result

def polynomial_subtraction(P, Q, q):
    """
    Perform the subtraction of two polynomial vectors P and Q with modular reduction.
    Handles polynomials of different sizes by padding the shorter one with zeros.

    Parameters:
        P (list): Coefficients of the first polynomial.
        Q (list): Coefficients of the second polynomial.
        q (int): Modulus for the computation.

    Returns:
        list: Coefficients of the resulting polynomial (P - Q) mod q.
    """
    # Determine the maximum length
    max_len = max(len(P), len(Q))

    # Extend both polynomials to the same length by padding with zeros
    P = P + [0] * (max_len - len(P))
    Q = Q + [0] * (max_len - len(Q))

    # Perform modular subtraction
    result = [(P[i] - Q[i]) % q for i in range(max_len)]
    
    return result

def multiply_matrix_by_vector(A, V, q):
    # Initialize result as a vector of zero polynomials
    result = [[0] * len(A[0][0]) for _ in range(len(A))]
    for i in range(len(A)):  # Iterate over rows of A
        for k in range(len(V)):  # Iterate over entries in V
            # Multiply the polynomial in A[i][k] with the polynomial in V[k]
            product = polynomial_multiplication(A[i][k], V[k], q)
            # Add the result to the corresponding entry in the result vector
            result[i] = polynomial_addition(result[i], product, q)
    
    return result

def multiply_matrix_by_matrix(A, B, q):
    # Get dimensions of A and B
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    # Check if matrix multiplication is valid
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must match the number of rows in B.")

    # Initialize result as a zero matrix of the appropriate size
    result = [[[0] * len(A[0][0]) for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):  # Iterate over rows of A
        for j in range(cols_B):  # Iterate over columns of B
            for k in range(cols_A):  # Iterate over entries in the row of A and column of B
                # Multiply the polynomial in A[i][k] with the polynomial in B[k][j]
                product = polynomial_multiplication(A[i][k], B[k][j], q)
                # Add the product to the corresponding entry in the result matrix
                result[i][j] = polynomial_addition(result[i][j], product, q)
    
    return result

def multiply_vector_by_vector(U, V, q):
    """
    Computes the dot product of two vectors of polynomials.

    Args:
        U (list): A vector of polynomials (each polynomial is a list of coefficients).
        V (list): Another vector of polynomials (same size as U).
        q (int): Modulo value for coefficient reduction.

    Returns:
        list: A single polynomial (dot product of U and V), reduced modulo q.
    """
    result = [0] * len(U[0][0])
    for i in range(len(U)) :
        product = polynomial_multiplication(U[i][0], V[i], q)
        result = polynomial_addition(product, result, q)
    return result

def add_vector_to_vector(V1, V2, q) :
    result = []
    for i in range(len(V1)):
            result.append(
                polynomial_addition(V1[i], V2[i], q)
            )
    return result
    
def mods(q, x):
    r = x % q
    if r > (q - 1) // 2:
        r -= q
    return r
        

def size_of_integer(q, x):
    return abs(mods(q=q, x=x))

def size_of_polynomial(q, V):
    result = -q-1
    for v in V:
        temp = size_of_integer(q=q, x=v)
        if temp > result :
            result = temp
            
    return result

def roundq(q, x):
    x = x % q
    return 1 if x > q // 4 and x < 3 * q // 4 else 0
    
def generate_random_polynomial(q, n):
    return [ randint(0, q-1) for _ in range(n) ]

def generate_random_small_polynomial(n, eta):
    return [ randint(-eta, eta) for _ in range(n) ]
    
def generate_A(q, k, n):
    return [[generate_random_polynomial(q,n) for _ in range(k)] for _ in range(k)]
    

def generate_small_polynomial_vector(k, n, eta1):
    return [ generate_random_small_polynomial(n=n, eta=eta1) for _ in range(k) ]

def gen_keys():
    q = 3329
    n = 4
    k = 2
    eta1 = 2
    eta2 = 2
    A = generate_A(q=q,k=k, n=n)
    s = [ generate_random_small_polynomial(n=n, eta=eta1) for _ in range(k) ]
    e = [ generate_random_small_polynomial(n=n, eta=eta2) for _ in range(k) ]
    t = add_vector_to_vector(multiply_matrix_by_vector(A, s, q), e, q)
    return ((A,t), s)

def encrypt(m, pub):
    q = 3329
    n = 4
    k = 2
    eta1 = 2
    eta2 = 2
    A = pub[0]
    t = pub[1]
    
    r = generate_small_polynomial_vector(k,n,eta1)
    e1 = generate_small_polynomial_vector(k,n,eta2)
    e2 = generate_random_small_polynomial(n,eta2)
    u = add_vector_to_vector(multiply_matrix_by_vector(transpose_matrix(A), r,q) ,e1, q)
    t_transpose = transpose_vector_of_polynomials(t)
    t_transpose_times_r = multiply_vector_by_vector(t_transpose, r, q)
    t_transpose_times_r_plus_e2 = polynomial_addition(t_transpose_times_r, e2, q)
    v = compute_v(t_transpose_times_r_plus_e2, m, q)
    print(v)
    return (u,v)

def decrypt(cipher, prv) :
    q = 3329
    u, v = cipher
    s = prv
    s_transpose = transpose_vector_of_polynomials(s)
    s_transpose_times_u = multiply_vector_by_vector(s_transpose, u, q)
    v_minus_s_transpose_times_u = polynomial_subtraction(v, s_transpose_times_u, q)
    result = []
    for element in v_minus_s_transpose_times_u:
        result.append(roundq(q, element))
        
    return result


def compute_v(t_r_e2, message, q):
    """
    Compute v = tT⋅r + e2 + message * round(q/2) mod q
    
    Parameters:
        t_r_e2 (list): Coefficients of tT⋅r + e2 as a polynomial vector.
        message (list): Coefficients of the message as a polynomial vector.
        q (int): Modulus for the computation.
    
    Returns:
        list: Coefficients of the polynomial v mod q.
    """
    # Compute the scaling factor
    scale = round(q/2)

    # Extend the shorter list to match the longer one
    max_len = max(len(t_r_e2), len(message))
    t_r_e2 = t_r_e2 + [0] * (max_len - len(t_r_e2))
    message = message + [0] * (max_len - len(message))

    # Compute scaled message
    scaled_message = [(coeff * scale) % q for coeff in message]

    # Compute v as the sum of t_r_e2 and scaled_message, reduced mod q
    v = [(t_r_e2[i] + scaled_message[i]) % q for i in range(max_len)]
    
    return v

def demonstration():
    pub, prv = gen_keys()
    message = "H"
    binary_message = string_to_binary_vector(message)
    cipher = encrypt(binary_message, pub)
    decrypted_binary = decrypt(cipher, prv)
    decrypted_message = binary_vector_to_string(decrypted_binary)
    print(message)
    print(decrypted_message)


if __name__ == "__main__":
    demonstration()