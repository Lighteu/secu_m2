import hashlib
import random 


q = 137
n = 4
k = 2
eta1 = 2
eta2 = 2
du = 10
dv = 4


def shake128(input_data, output_length):
    shake = hashlib.shake_128()
    shake.update(input_data)
    return list(shake.digest(output_length))  # Convert the output to a list of integers

def generate_random_polynomial(seed, row, col, degree, q):
    input_data = (seed + f"{row},{col}").encode('utf-8')
    hash_output = shake128(input_data, degree * 2)  # Generate enough bytes for 'degree' coefficients
    coefficients = []
    for i in range(degree):
        value = (hash_output[2 * i] << 8) + hash_output[2 * i + 1]
        coefficients.append(value % q)
    return coefficients


def generate_matrix_A(k, n, q, seed):
    degree = n  # Set degree to n
    matrix = []
    for row in range(k):
        matrix_row = []
        for col in range(k):
            polynomial = generate_random_polynomial(seed, row, col, degree, q)
            matrix_row.append(polynomial)
        matrix.append(matrix_row)
    return matrix


def compress_coefficients(coefficients, q, d):
    compressed = []
    for coeff in coefficients:
        # Scale down the coefficient to the range [0, 2^d)
        compressed_coeff = round(coeff * ((2**d) / q)) % (2**d)
        compressed.append(compressed_coeff)
    return compressed


def decompress_coefficients(compressed, q, d):
    decompressed = []
    for compressed_coeff in compressed:
        # Scale back the coefficient to the range [0, q)
        decompressed_coeff = round(compressed_coeff * (q / (2**d))) % q
        decompressed.append(decompressed_coeff)
    return decompressed

def generate_random_small_polynomial(n, eta):
    def cbd_sample(eta):
        b = [random.randint(0, 1) for _ in range(eta)]  # Random bits for sum b
        b_prime = [random.randint(0, 1) for _ in range(eta)]  # Random bits for sum b'
        return sum(b) - sum(b_prime)  # CBD value is the difference of sums

    # Generate n coefficients using CBD(eta)
    return [cbd_sample(eta) for _ in range(n)]

def format_polynomial(coefficients):
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
    formatted_matrix = []
    for row in matrix:
        formatted_row = [format_polynomial(poly) for poly in row]
        formatted_matrix.append(formatted_row)
    
    # Display the matrix
    for row in formatted_matrix:
        print(" | ".join(row))

def string_to_binary_vector(s):
    binary_vector = []
    for char in s:
        # Convert each character to its UTF-8 binary representation
        binary_char = ''.join(format(byte, '08b') for byte in char.encode('utf-8'))
        binary_vector.extend([int(bit) for bit in binary_char])
    return binary_vector


def binary_vector_to_string(binary_vector):
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
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def transpose_vector_of_polynomials(vector):
    # Row vector (1D list) -> Column vector (2D list)
    if isinstance(vector[0], list) and isinstance(vector[0][0], (int, float)):
        return [[v] for v in vector]
    # Column vector (2D list) -> Row vector (1D list)
    else:
        return [v[0] for v in vector]



def polynomial_multiplication(P, Q, q):
    if len(P) != len(Q):
        raise ValueError("Polynomials P and Q must be of the same length.")
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
    

def generate_small_polynomial_vector(k, n, eta1):
    return [ generate_random_small_polynomial(n=n, eta=eta1) for _ in range(k) ]

def gen_keys():
    global q 
    global n 
    global k 
    global eta1 
    global eta2 
    row = ''.join([str(random.randint(0,1)) for _ in range(n)])
    A = generate_matrix_A(k, n, q, row)
    s = [ generate_random_small_polynomial(n=n, eta=eta1) for _ in range(k) ]
    e = [ generate_random_small_polynomial(n=n, eta=eta2) for _ in range(k) ]
    t = add_vector_to_vector(multiply_matrix_by_vector(A, s, q), e, q)
    return row, A, s, e, t

def encrypt(m, pub):
    global q 
    global n 
    global k 
    global eta1 
    global eta2 
    global du
    global dv
    
    row = pub[0]
    t = pub[1]
    A = generate_matrix_A(k, n, q, row)
    r = generate_small_polynomial_vector(k,n,eta1)
    e1 = generate_small_polynomial_vector(k,n,eta2)
    e2 = generate_random_small_polynomial(n,eta2)
    u = add_vector_to_vector(multiply_matrix_by_vector(transpose_matrix(A), r,q) ,e1, q)
    t_transpose = transpose_vector_of_polynomials(t)
    t_transpose_times_r = multiply_vector_by_vector(t_transpose, r, q)
    t_transpose_times_r_plus_e2 = polynomial_addition(t_transpose_times_r, e2, q)
    v = compute_v(t_transpose_times_r_plus_e2, m, q)
    
    c1 = []
    for i in range(len(u)):
        c1.append(compress_coefficients(u[i], q, du))
    c2 = compress_coefficients(v, q, dv)
    return r, e1, e2, u, v, c1, c2

def decrypt(cipher, prv) :
    global q
    global du
    global dv
    
    c1, c2 = cipher
    u_prime = []
    for i in range(len(c1)):
        u_prime.append(decompress_coefficients(c1[i], q, du))
    v_prime = decompress_coefficients(c2, q, dv)
    s = prv
    s_transpose = transpose_vector_of_polynomials(s)
    s_transpose_times_u = multiply_vector_by_vector(s_transpose, u_prime, q)
    v_minus_s_transpose_times_u = polynomial_subtraction(v_prime, s_transpose_times_u, q)
    result = []
    for element in v_minus_s_transpose_times_u:
        result.append(roundq(q, element))
        
    return u_prime, v_prime, result


def compute_v(t_r_e2, message, q):
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
        row, A_demo, s_demo, e_demo, t_demo = gen_keys()
        q = 137
        n = 4
        k = 2
        eta1 = 2
        eta2 = 2
        du = 10
        dv = 4

        # Encryption data
        message = "i"
        m_bin = string_to_binary_vector(message)
        r, e1, e2, u, v, c1, c2 = encrypt(m_bin, (row, t_demo))

        # Decryption data
        u_prime, v_prime, decrypted_bin = decrypt((c1, c2), s_demo)

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
            r"q=", str(q), r", n=", str(n), r", k=", str(k), r", \eta_1=", str(eta1), r", \eta_2=", str(eta2), r", d_u=", str(du), r", d_v=", str(dv)
        ).scale(0.5)
        self.play(FadeIn(param_text))
        self.wait(2)
        
        self.play(param_text.animate.move_to([0, 2.5, 0]))
        self.wait(2)

        row_text = MathTex(
            r"\rho = ", row
        ).scale(0.5)
        self.play(FadeIn(row_text))
        self.wait(2)
        self.play(row_text.animate.move_to([0,2,0]))
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
            FadeOut(row_text),
            FadeOut(keygen_title),
        )
        self.wait(1)

        public_key_text = MathTex(r"(\rho , t)", font_size=120)
        self.play(FadeIn(public_key_text))
        self.wait(1)
        self.play(FadeOut(public_key_text))
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
        u_group.to_corner(UL)
        v_group.to_corner(UR)
        self.play(FadeIn(u_group))
        self.play(FadeIn(v_group))
        arrow1 = Arrow(start=UP, end=DOWN)
        arrow1.next_to(u_group, DOWN, buff=0.5)
        arrow2 = Arrow(start=UP, end=DOWN)
        arrow2.next_to(v_group, DOWN, buff=0.5)
        self.play(GrowArrow(arrow1))
        
        c1_tex = VGroup(*[MathTex(poly_to_tex(up)).scale(0.5) for up in c1]).arrange(DOWN)
        c1_label = MathTex(r"\mathbf{c_1} =").scale(0.5)
        c1_group = VGroup(c1_label, c1_tex).arrange(RIGHT)
        c1_group.next_to(arrow1, DOWN, buff=0.5)
        self.play(FadeIn(c1_group))
        self.wait(2)
        
        self.play(GrowArrow(arrow2))
        c2_tex = MathTex(poly_to_tex(c2)).scale(0.5)
        c2_label = MathTex(r"\mathbf{c_2} =").scale(0.5)
        c2_group = VGroup(c2_label, c2_tex).arrange(RIGHT)
        c2_group.next_to(arrow2, DOWN, buff=0.5)
        self.play(FadeIn(c2_group))
        self.wait(2)
        
        self.play(
            FadeOut(u_group),
            FadeOut(v_group),
            FadeOut(c2_group),
            FadeOut(c1_group),
            FadeOut(arrow1),
            FadeOut(arrow2)
        )
        
        cipher_text = MathTex(r"(c_1 , c_2)", font_size=120)
        self.play(FadeIn(cipher_text))
        self.wait(1)
        self.play(FadeOut(cipher_text))
        self.wait(1)
        # ---------------------------------
        # Decryption
        # ---------------------------------
        dec_title = Text("Decryption", font_size=32)
        self.play(FadeIn(dec_title))
        self.wait(2)
        self.play(FadeOut(dec_title))
        self.wait(1)
        
        
        c1_group.to_corner(UL)
        c2_group.to_corner(UR)
        self.play(FadeIn(c1_group))
        self.play(FadeIn(c2_group))
        arrow1 = Arrow(start=UP, end=DOWN)
        arrow1.next_to(c1_group, DOWN, buff=0.5)
        arrow2 = Arrow(start=UP, end=DOWN)
        arrow2.next_to(c2_group, DOWN, buff=0.5)
        self.play(GrowArrow(arrow1))
        
        
        u_prime_tex = VGroup(*[MathTex(poly_to_tex(up)).scale(0.5) for up in u_prime]).arrange(DOWN)
        u_prime_label = MathTex(r"\mathbf{u\prime} =").scale(0.5)
        u_prime_group = VGroup(u_prime_label, u_prime_tex).arrange(RIGHT)
        u_prime_group.next_to(arrow1, DOWN, buff=0.5)
        self.play(FadeIn(u_prime_group))
        self.wait(2)

        self.play(GrowArrow(arrow2))
        v_prime_tex = MathTex(poly_to_tex(v_prime)).scale(0.5)
        v_prime_label = MathTex(r"\mathbf{v\prime} =").scale(0.5)
        v_prime_group = VGroup(v_prime_label, v_prime_tex).arrange(RIGHT)
        v_prime_group.next_to(arrow2, DOWN, buff=0.5)
        self.play(FadeIn(v_prime_group))
        self.wait(2)
        
        self.play(
            FadeOut(u_prime_group),
            FadeOut(v_prime_group),
            FadeOut(c2_group),
            FadeOut(c1_group),
            FadeOut(arrow1),
            FadeOut(arrow2)
        )
        
        # Show the final equation for decryption and final message
        expression = MathTex(r"\mathbf{m} = \text{Round}_q(\mathbf{v\prime} - \mathbf{s}^T \mathbf{u\prime})").scale(0.5)
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
