# Assume the provided code with gen_keys, encrypt, decrypt, etc. is available.
# You might do something like:
# from kyber_code import gen_keys, encrypt, decrypt, string_to_binary_vector
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
    q = 137
    n = 4
    k = 2
    eta1 = 2
    eta2 = 2
    A = generate_A(q=q,k=k, n=n)
    s = [ generate_random_small_polynomial(n=n, eta=eta1) for _ in range(k) ]
    e = [ generate_random_small_polynomial(n=n, eta=eta2) for _ in range(k) ]
    t = add_vector_to_vector(multiply_matrix_by_vector(A, s, q), e, q)
    return A, t, s, e

def encrypt(m, pub):
    q = 137
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
    
    return u, v, r, e1, e2

def decrypt(cipher, prv) :
    q = 137
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
    message = "Hello world!"
    binary_message = string_to_binary_vector(message)
    cipher = encrypt(binary_message, pub)
    decrypted_binary = decrypt(cipher, prv)
    decrypted_message = binary_vector_to_string(decrypted_binary)
    print(message)
    print(decrypted_message)


from manim import *
import numpy as np

# Assuming the kyber code (gen_keys, encrypt, decrypt, etc.) is already defined.

def poly_to_tex(poly):
    terms = []
    for i, coeff in enumerate(poly):
        if coeff != 0:
            if i == 0:
                terms.append(str(coeff))
            else:
                terms.append(str(coeff) + "x^{" + str(i) + "}")
    return " + ".join(terms) if terms else "0"

class ShowKyberProcess(Scene):
    def construct(self):
        # Generate keys and related data
        A_demo, t_demo, s_demo, e_demo = gen_keys()
        q = 137
        n = 4
        k = 2
        eta1 = 2
        eta2 = 2

        # Encryption data
        message = "i"
        m_bin = string_to_binary_vector(message)
        u, v, r, e1, e2 = encrypt(m_bin, (A_demo, t_demo))

        # Decryption data
        decrypted_bin = decrypt((u, v), s_demo)

        # Title
        title = Text("Kyber Key Generation, Encryption & Decryption", font_size=36)
        self.play(FadeIn(title))
        self.wait(2)
        self.play(FadeOut(title))

        # ---------------------------------
        # Key Generation
        # ---------------------------------
        keygen_title = Text("Key Generation", font_size=32)
        self.play(FadeIn(keygen_title))
        self.wait(2)
        self.play(keygen_title.animate.to_edge(UP))

        # Show parameters and keep them until keygen step done
        param_text = MathTex(
            r"q=", str(q), r", n=", str(n), r", k=", str(k), r", \eta_1=", str(eta1), r", \eta_2=", str(eta2)
        ).scale(0.5)
        self.play(FadeIn(param_text))
        self.wait(2)
        
        self.play(param_text.animate.move_to([0, 2.5, 0]))
        self.wait(2)

        matrix_str = r"A = \begin{pmatrix}"
        for i, row in enumerate(A_demo):
            row_str = " & ".join(poly_to_tex(entry) for entry in row)
            matrix_str += row_str
            if i < len(A_demo) - 1:
                matrix_str += r" \\ "
        matrix_str += r"\end{pmatrix}"

        A_mat = MathTex(matrix_str).scale(0.4)



        # Position it below param_text
        A_mat.next_to(param_text, DOWN, buff=1)

        # Fade in the matrix
        self.play(FadeIn(A_mat))
        self.wait(2)

        # Animate it moving to the left by 3.5 units
        self.play(A_mat.animate.shift(LEFT * 3.5))
        self.wait(2)


        # Show s as a MathTex column vector
        s_str = r"\mathbf{s} = \begin{pmatrix}"
        s_str += r" \\ ".join(poly_to_tex(sp) for sp in s_demo)
        s_str += r"\end{pmatrix}"
        s_group = MathTex(s_str).scale(0.4)
        s_group.next_to(A_mat, RIGHT, buff=1)
        self.play(FadeIn(s_group))
        self.wait(2)

        # Show e as a MathTex column vector
        e_str = r"\mathbf{e} = \begin{pmatrix}"
        e_str += r" \\ ".join(poly_to_tex(ep) for ep in e_demo)
        e_str += r"\end{pmatrix}"
        e_group = MathTex(e_str).scale(0.4)
        e_group.next_to(s_group, RIGHT, buff=1)
        self.play(FadeIn(e_group))
        self.wait(2)

        # Show t = A s + e
        t_expression = MathTex(r"\mathbf{t} = A\mathbf{s} + \mathbf{e}").scale(1)
        t_expression.next_to(A_mat, DOWN, buff=1)
        self.play(FadeIn(t_expression))
        self.wait(2)

        # Show t as a MathTex column vector
        t_str = r"\mathbf{t} = \begin{pmatrix}"
        t_str += r" \\ ".join(poly_to_tex(tp) for tp in t_demo)
        t_str += r"\end{pmatrix}"
        t_group = MathTex(t_str).scale(1)
        t_group.next_to(t_expression, RIGHT, buff=0.5)
        self.play(FadeIn(t_group))
        self.wait(2)


        # Now that key generation step is complete, remove all keygen elements
        self.play(
            FadeOut(param_text),
            FadeOut(A_mat),
            FadeOut(s_group),
            FadeOut(e_group),
            FadeOut(t_expression),
            FadeOut(t_group),
            FadeOut(keygen_title),
        )
        self.wait(1)

        # ---------------------------------
        # Encryption
        # ---------------------------------
        enc_title = Text("Encryption", font_size=32)
        self.play(FadeIn(enc_title))
        self.wait(2)
        self.play(enc_title.animate.to_edge(UP))
        
        binary_str = "m = " + "".join(str(bit) for bit in m_bin)
        binary_math = MathTex(binary_str).scale(0.55)

        poly_str = "m(x) = " + poly_to_tex(m_bin)
        poly_math = MathTex(poly_str).scale(0.55)

        msg_group = VGroup(binary_math, poly_math).arrange(RIGHT, buff=0.5)
        msg_group.next_to(enc_title, DOWN, buff=1)

        self.play(FadeIn(msg_group))
        self.wait(2)

        # Show r, e1, e2 and keep them until done showing u and v
        r_tex = VGroup(*[MathTex(poly_to_tex(rp)).scale(0.5) for rp in r]).arrange(DOWN)
        r_label = MathTex(r"\mathbf{r} =").scale(0.5)
        r_group = VGroup(r_label, r_tex).arrange(RIGHT)
        r_group.to_edge(LEFT)
        self.play(FadeIn(r_group))
        self.wait(1)

        e1_tex = VGroup(*[MathTex(poly_to_tex(e1p)).scale(0.5) for e1p in e1]).arrange(DOWN)
        e1_label = MathTex(r"\mathbf{e_1} =").scale(0.5)
        e1_group = VGroup(e1_label, e1_tex).arrange(RIGHT)
        e1_group.next_to(r_group, RIGHT, buff=1)
        self.play(FadeIn(e1_group))
        self.wait(1)

        e2_str = MathTex(r"\mathbf{e_2} = " + poly_to_tex(e2)).scale(0.5)
        e2_str.next_to(e1_group, RIGHT, buff=1)
        self.play(FadeIn(e2_str))
        self.wait(2)

        # Show u = A^T r + e1
        u_expression = MathTex(r"\mathbf{u} = A^T \mathbf{r} + \mathbf{e_1}").scale(0.5)
        u_expression.next_to(r_group, DOWN, buff=1)
        self.play(FadeIn(u_expression))
        self.wait(2)

        u_tex = VGroup(*[MathTex(poly_to_tex(up)).scale(0.5) for up in u]).arrange(DOWN)
        u_label = MathTex(r"\mathbf{u} =").scale(0.5)
        u_group = VGroup(u_label, u_tex).arrange(RIGHT)
        u_group.next_to(u_expression, RIGHT, buff=0.5)
        self.play(FadeIn(u_group))
        self.wait(2)

        # Show v = t^T r + e2 + (q/2)*m
        v_expression = MathTex(r"\mathbf{v} = \mathbf{t}^T\mathbf{r} + \mathbf{e_2} + \lceil q/2 \rfloor \mathbf{m}").scale(0.5)
        v_expression.next_to(u_expression, DOWN, buff=1)
        self.play(FadeIn(v_expression))
        self.wait(2)

        v_tex = MathTex(poly_to_tex(v)).scale(0.5)
        v_label = MathTex(r"\mathbf{v} =").scale(0.5)
        v_group = VGroup(v_label, v_tex).arrange(RIGHT)
        v_group.next_to(v_expression, RIGHT, buff=0.5)
        self.play(FadeIn(v_group))
        self.wait(2)

        # Now remove all encryption elements
        self.play(
            FadeOut(r_group),
            FadeOut(e1_group),
            FadeOut(e2_str),
            FadeOut(u_expression),
            FadeOut(u_group),
            FadeOut(v_expression),
            FadeOut(v_group),
            FadeOut(enc_title),
            FadeOut(msg_group)
        )
        self.wait(1)

        # ---------------------------------
        # Decryption
        # ---------------------------------
        dec_title = Text("Decryption", font_size=32)
        self.play(FadeIn(dec_title))
        self.wait(2)
        self.play(dec_title.animate.to_edge(UP))

        # Show the final equation for decryption and final message
        expression = MathTex(r"\mathbf{m} = \text{Round}_q(\mathbf{v} - \mathbf{s}^T \mathbf{u})").scale(0.5)
        self.play(FadeIn(expression))
        self.wait(2)

        bit_str = "".join(str(b) for b in decrypted_bin)
        bits_text = MathTex(r"\mathbf{m} = " + bit_str).scale(0.5)
        bits_text.next_to(expression, DOWN, buff=0.5)
        self.play(FadeIn(bits_text))
        self.wait(2)

        # Remove decryption elements
        self.play(FadeOut(expression), FadeOut(bits_text), FadeOut(dec_title))
        self.wait(1)

        # End
        final_message = Text("Process Complete", font_size=32)
        self.play(FadeIn(final_message))
        self.wait(3)
