from simplified_kyber_pke import *
from dilithium_toy_version import *
from hashlib import shake_256
import secrets



n = 4
q = 2**14
lambda_p = 256
k = 3
l = 2
gamma1 = 2**10
gamma2 = 513
tau = 4
eta = 10
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


def generate_random_polynomial(q, n, base_seed, i, j):
    # Derive a unique seed for (i, j)
    seed_input = base_seed.to_bytes((base_seed.bit_length() + 7)//8, byteorder='big') + i.to_bytes(2,'big') + j.to_bytes(2,'big')
    derived_seed = int.from_bytes(shake_256(seed_input).digest(32), byteorder='big')
    
    random.seed(derived_seed)
    return [random.randint(0, q-1) for _ in range(n)]

def generate_random_small_polynomial(n, eta, seed):
    """Generate a random small polynomial between -eta and eta using a seed."""
    random.seed(seed)  # Setting the seed for reproducibility
    return [ randint(-eta, eta) for _ in range(n) ]
    
def generate_A(q, k, n):
    return [[generate_random_polynomial(q,n) for _ in range(k)] for _ in range(k)]
    

def expandA(q, k, l, n, p):
    """Generates the A matrix composed of random polynomial modulo q using a seed p."""
    A = []
    for i in range(k):
        row = []
        for j in range(l):
            poly = generate_random_polynomial(q, n, p, i, j)
            row.append(poly)
        A.append(row)
    return A

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
        vector.append(generate_random_small_polynomial(n, eta, seed + i))
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
    # Convert p_prime (integer) to bytes (let's assume 4 bytes for this example)
    p_prime_bytes = p_prime.to_bytes((p_prime.bit_length() + 7) // 8, byteorder='big')

    # Derive two seeds from p_prime for generating s1 and s2
    seed_s1 = hash(p_prime_bytes + "s1".encode('utf-8'),256) & 0xFFFFFFFF
    seed_s2 = hash(p_prime_bytes + "s2".encode('utf-8'),512) & 0xFFFFFFFF
    
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
    byte_message = message.encode('utf-8')
    concatenated = tr + byte_message
    
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
    
    # Converting p_seed (integer) to bytes (let's assume 4 bytes for this example)
    # Step 2: Concatenate p_seed and the serialized t_vector bytes
    p_bytes = p_seed.to_bytes((p_seed.bit_length() + 7) // 8, byteorder='big')

    concatenated = p_bytes + t_bytes
    
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
    K_bytes = K.to_bytes(256 // 8, byteorder='big')
    concatenated = K_bytes + rnd + mu
    
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
        polynomial = []
        for _ in range(n):
            rand_bytes = shake.digest(4)  # 4 bytes (32 bits) for each value
            shake.update(rand_bytes)  # Update with new randomness for the next coefficient
            
            rand_int = int.from_bytes(rand_bytes, byteorder='big')
            mapped_int = (rand_int % (2 * gamma1 + 1)) - gamma1 - 1  # Mapping to range [-gamma1, gamma1]
            
            polynomial.append(mapped_int)
        
        y.append(polynomial)
    
    return y

def decompose(r, a, q):
    """
    Decompose r into (r1, r0) such that r = r1 * a + r0,
    with -a/2 < r0 ≤ a/2 and 0 ≤ r1 ≤ m.

    Parameters:
        r (int): Input integer, 0 <= r < q.
        a (int): Even divisor of q - 1.
        q (int): Modulus.

    Returns:
        tuple: (r1, r0) decomposition.
    """
    # Step 1: Compute r0 as the remainder of r modulo a, adjusted to [-a/2, a/2]
    r0 = r % a
    if r0 > a // 2:
        r0 -= a  # Center r0 around 0 (-a/2 < r0 <= a/2)

    # Step 2: Compute r1 as the quotient
    r1 = (r - r0) // a
    # Step 3: Return the decomposition
    return r1, r0

def compute_w1(w):
    result = [[0] * len(w[0]) for _ in range(len(w))]
    for i in range(len(w)):
        for j in range(len(w[0])):
            w_highbits,_ = decompose(w[i][j], 2*gamma2,q)
            result[i][j] = w_highbits
    return result

def compute_r0(w_minus_cs2):
    result = [[0] * len(w_minus_cs2[0]) for _ in range(len(w_minus_cs2))]
    for i in range(len(w_minus_cs2)):
        for j in range(len(w_minus_cs2[0])):
            _,lowbits = decompose(w_minus_cs2[i][j], 2*gamma2,q)
            result[i][j] = lowbits
    return result


def flatten_matrix(m):
    flattened_list = []
    for sublist in m:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list


def compute_challenge(mu, w1, lamda_param, hash_function):
    """
    Computes the challenge c_hat = H(mu || w1, 2*lamda_param).
    
    Parameters:
        mu (bytes): The hash of the message and key trace.
        w1 (bytes): The coarse approximation of w.
        lamda_param (int): The length of c_hat (2 * lamda_param).
        hash_function: The hash function to use (default: SHA-512).
    
    Returns:
        bytes: The challenge hash c_hat.
    """
    # Combine mu and w1 as byte strings
    flat_w1 = flatten_matrix(w1)
    w1_bytes = bytes(flat_w1)
    input_data = mu + w1_bytes
    
    # Apply the hash function
    c_hat = hash_function(input_data).digest(32)
    
    # Truncate to the desired domain size 2*lamda_param
    max_bits = 2*lamda_param
    c_hat_int = int.from_bytes(c_hat, byteorder="big") % max_bits
    return c_hat_int.to_bytes((max_bits.bit_length() + 7) // 8, byteorder="big")


def sample_in_ball(c_hat, n, tau):
    """
    Samples a polynomial c from the domain of c_hat with exactly tau non-zero coefficients.
    
    Parameters:
        c_hat (bytes): The input challenge hash.
        n (int): The degree of the polynomial (number of coefficients).
        tau (int): The number of non-zero coefficients in the polynomial.
    
    Returns:
        list[int]: The sampled polynomial with coefficients in {-1, 0, 1}.
    """
    random.seed(c_hat)  # Use c_hat as a seed for reproducibility
    
    # Initialize a polynomial of degree n with all coefficients set to 0
    c = [0] * n
    
    # Randomly choose tau unique positions for non-zero coefficients
    positions = random.sample(range(n), tau)
    
    # Assign either 1 or -1 to the selected positions
    for pos in positions:
        c[pos] = random.choice([-1, 1])
    
    return c


def infinity_norm(vec):
    """
    Computes the infinity norm of a vector.
    
    Parameters:
        vec (list[int]): The vector to compute the norm for.
    
    Returns:
        int: The infinity norm (maximum absolute value of the coefficients).
    """
    res = []
    for x in vec:
        res.append(size_of_polynomial(q,x))
    return max(abs(x) for x in res)


def polynomial_addition_mods(P, Q, q):
    result = []
    for i in range(len(P)):
        result.append(mods(q, (P[i] + Q[i])))
        # print(f"P + Q = {(P[i] + Q[i])}\n\n")
        # print(f"P + Q % q = {(P[i] + Q[i]) % q}\n\n")
    return result

def add_vector_to_vector_mods(V1, V2, q) :
    result = []
    for i in range(len(V1)):
            result.append(
                polynomial_addition_mods(V1[i], V2[i], q)
            )
    return result

def compute_w1_prime(mu, w1_prime, output_bits):
    """
    Compute c_tilde = H(mu || w1_prime, 2 * lambda).

    Parameters:
    - mu: The message hash (bytes).
    - w1_prime: The vector w1' (list of integers or bytes).
    - output_bits: The desired output size in bits (2 * lambda).

    Returns:
    - c_tilde: The hash output as an integer truncated to output_bits.
    """
    flat_w1_prime = flatten_matrix(w1_prime)
    w1_prime_bytes = bytes(flat_w1_prime)

     # Concatenate mu and w1_prime
    concatenated = mu + w1_prime_bytes
    
    # Hash using SHAKE-256 and truncate to the desired number of bits
    shake = shake_256(concatenated)
    c_tilde_bytes = shake.digest(output_bits // 8)  # Truncate to output_bits in bytes
    
    # Convert to an integer for further use
    c_tilde = int.from_bytes(c_tilde_bytes, byteorder='big')
    
    return c_tilde    


    

def generate_keys(q, k, l, n, lambda_param):
    xi = generate_xi()
    rho, p_prime, K = compute_p_p_prime_K(xi)
    A_matrix = expandA(q,k,l,n,rho)
    s1_vec, s2_vec = expandS(k,l,n,eta,p_prime)
    t_vector = compute_t_vector(A_matrix,s1_vec,s2_vec,q)
    trace_tr = compute_tr(rho,t_vector,lambda_param)
    return rho, A_matrix, t_vector, K, trace_tr, s1_vec, s2_vec



def generate_signature(message, q, k, l, n, p, tr, K, s1_vector, s2_vector):
    matrix_A = expandA(q, k, l, n, p)
    print("A Matrix:", matrix_A)
    mu = compute_mu(tr, message) # string byte
    deterministic_rnd = generate_deterministic_rnd()
    rho_second = compute_p_prime_prime(K, deterministic_rnd, mu)
    kappa = 0
    found = False
    while(not found):
        print("Attempt kappa:", kappa)
        y_vector = expand_mask_with_nonce(rho_second,kappa,l,gamma1)
        w =  multiply_matrix_by_vector(matrix_A,y_vector, q)
        w1 = compute_w1(w)
        c_hat = compute_challenge(mu,w1,lambda_p,shake_256)
        c = sample_in_ball(c_hat, n, tau)
        c_times_s1 = multiply_polynomial_with_vector(c,s1_vector,q)
        z = add_vector_to_vector(y_vector, c_times_s1,q)
        c_times_s2 = multiply_polynomial_with_vector(c,s2_vector,q)
        w_minus_cs2 = compute_w_minus_cs2(w, c_times_s2, q)
        print("w_minus_cs2 (Signing):", w_minus_cs2)
        r_zero = compute_r0(w_minus_cs2)
        z_bound = gamma1 - Beta
        r1_bound = gamma2 - Beta
        z_norm = infinity_norm(z)
        r0_norm = infinity_norm(r_zero)
        is_z_valid = z_norm < z_bound
        is_r0_valid = r0_norm < r1_bound
        if(is_z_valid and is_r0_valid):
            found = True
        kappa+=l
    return c_hat,z, w, w1, c, z_bound, w_minus_cs2, r_zero


def verify_signature(message, rho, z, t_vector, c_hat):
    z_norm = infinity_norm(z)
    z_bound = gamma1 - Beta
    is_z_valid = z_norm < z_bound
    if(not is_z_valid):
        print("Signature is not valid")
        return None
    matrix_A = expandA(q, k, l, n, rho)
    print("A Matrix in verify:", matrix_A)
    tr = compute_tr(rho,t_vector,lambda_p)
    mu = compute_mu(tr,message)
    c = sample_in_ball(c_hat,n,tau)
    
    A_times_z = multiply_matrix_by_vector(matrix_A,z,q)
    c_times_t = multiply_polynomial_with_vector(c,t_vector,q)

    Az_ct = compute_w_minus_cs2(A_times_z,c_times_t,q)
    print("Az - ct (Verification):", Az_ct)
    w1_prime = compute_w1(Az_ct)
    c_tilda_cmp = compute_challenge(mu,w1_prime,lambda_p,shake_256)
    if(c_tilda_cmp == c_hat):
        print("Valid signature")
    else:
        print("Unvalid signature")
    # print(f"c_tilda_cmp value is {type(c_tilda_cmp)}\n\n")
    # c_hat_as_int = int.from_bytes(c_hat, byteorder='big')
    return Az_ct,c,w1_prime  




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

class ShowDiliProcess(Scene):
    def construct(self):
        # Generate keys and related data
        n = 4
        q = 2**7
        lambda_p = 16
        k = 3
        l = 2
        gamma1 = 2**10
        gamma2 = 513
        tau = 4
        eta = 10
        Beta = tau*eta

        rho, A_matrix, t_vector, K, trace_tr, s1_vec, s2_vec = generate_keys(q,k,l,n,lambda_p)
        c_hat,z, w, w1, c, z_bound, w_minus_cs2, r_zero = generate_signature("h",q,k,l,n,rho,trace_tr,K,s1_vec,s2_vec)
        Az_ct,c,w1_prime = verify_signature("h",rho,z,t_vector,c_hat)
        
        
        # ANIMATION START 
        
        title = Text("Kyber Key Generation, Encryption & Decryption", font_size=36)
        self.play(FadeIn(title))
        self.wait(2)
        self.play(FadeOut(title))
        
        
        # KEY GEN 
        
        keygen_title = Text("Key Generation", font_size=32)
        self.play(FadeIn(keygen_title))
        self.wait(2)
        self.play(keygen_title.animate.to_edge(UP))

        # Show parameters and keep them until keygen step done
        param_text = MathTex(
            r"n=", str(n), r", q=2^{7}, \lambda_p=", str(lambda_p), r", k=", str(k), r", l=", str(l), r", \gamma_1=", str(gamma1), r", \gamma_2=", str(gamma2), r", \tau=", str(tau), r", \beta=", str(Beta)
        ).scale(0.5)
        self.play(FadeIn(param_text))
        self.wait(2)
        
        self.play(param_text.animate.move_to([0, 2.5, 0]))
        self.wait(2)
        
        row_text = MathTex(
            r"\rho = ", rho
        ).scale(0.5)
        self.play(FadeIn(row_text))
        self.wait(2)
        self.play(row_text.animate.move_to([0,2,0]))
        self.wait(2)
        
        
        matrix_str = r"A = \begin{pmatrix}"
        for i, row in enumerate(A_matrix):
            row_str = " & ".join(poly_to_tex(entry) for entry in row)
            matrix_str += row_str
            if i < len(A_matrix) - 1:
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
        
        
        s1_str = r"\mathbf{s_1} = \begin{pmatrix}"
        s1_str += r" \\ ".join(poly_to_tex(sp) for sp in s1_vec)
        s1_str += r"\end{pmatrix}"
        s1_group = MathTex(s1_str).scale(0.4)
        s1_group.next_to(A_mat, RIGHT, buff=1)
        self.play(FadeIn(s1_group))
        self.wait(2)

        # Show e as a MathTex column vector
        s2_str = r"\mathbf{s_2} = \begin{pmatrix}"
        s2_str += r" \\ ".join(poly_to_tex(ep) for ep in s2_vec)
        s2_str += r"\end{pmatrix}"
        s2_group = MathTex(s2_str).scale(0.4)
        s2_group.next_to(s1_group, RIGHT, buff=1)
        self.play(FadeIn(s2_group))
        self.wait(2)
        
        tr_text = MathTex(
            r"t_r = ", '"' + ','.join([hex(x).split('x')[-1] for x in trace_tr]) + '"'
        ).scale(0.5)
        tr_text.next_to(A_mat, DOWN, buff=1)
        self.play(FadeIn(tr_text))
        self.wait(2)
        
        t_vec_str = r"\mathbf{t} = \begin{pmatrix}"
        t_vec_str += r" \\ ".join(poly_to_tex(ep) for ep in t_vector)
        t_vec_str += r"\end{pmatrix}"
        t_vec_group = MathTex(t_vec_str).scale(0.4)
        t_vec_group.next_to(tr_text, RIGHT, buff=1)
        self.play(FadeIn(t_vec_group))
        self.wait(2)
        
        K_text = MathTex(
            r"K = ", K
        ).scale(0.5)
        K_text.to_edge(DOWN)
        self.play(FadeIn(K_text))
        self.wait(2)
        
        self.play(
            FadeOut(keygen_title),
            FadeOut(param_text),
            FadeOut(row_text),
            FadeOut(A_mat),
            FadeOut(s1_group),
            FadeOut(s2_group),
            FadeOut(tr_text),
            FadeOut(t_vec_group),
            FadeOut(K_text)
        )
        
        
        # GEN SIGNATURE
        siggen_title = Text("Sig Generation", font_size=32)
        self.play(FadeIn(siggen_title))
        self.wait(2)
        self.play(siggen_title.animate.to_edge(UP))
        
        
        c_hatt_text = MathTex(
            r"\hat{c} = ", '"' + ','.join([hex(x).split('x')[-1] for x in c_hat]) + '"'
        ).scale(0.5)
        self.play(FadeIn(c_hatt_text))
        self.wait(2)
        
        self.play(c_hatt_text.animate.move_to([0, 2.5, 0]))
        self.wait(2)
        
        
        # Show s as a MathTex column vector
        z_str = r"\mathbf{s} = \begin{pmatrix}"
        z_str += r" \\ ".join(poly_to_tex(sp) for sp in z)
        z_str += r"\end{pmatrix}"
        z_group = MathTex(z_str).scale(0.4)
        
        z_group.next_to(c_hatt_text, DOWN, buff=1)

        # Fade in the matrix
        self.play(FadeIn(z_group))
        self.wait(1)

        # Animate it moving to the left by 3.5 units
        self.play(z_group.animate.shift(LEFT * 3.5))
        self.wait(1)

        # Show e as a MathTex column vector
        w1_str = r"\mathbf{w_1} = \begin{pmatrix}"
        w1_str += r" \\ ".join(poly_to_tex(ep) for ep in w)
        w1_str += r"\end{pmatrix}"
        w1_group = MathTex(w1_str).scale(0.4)
        w1_group.next_to(z_group, RIGHT, buff=1)
        self.play(FadeIn(w1_group))
        self.wait(1)
        
        """ matrix_str = r"w = \begin{pmatrix}"
        for i, row in enumerate(w):
            row_str = " & ".join(poly_to_tex(entry) for entry in row)
            matrix_str += row_str
            if i < len(w) - 1:
                matrix_str += r" \\ "
        matrix_str += r"\end{pmatrix}"

        w_mat = MathTex(matrix_str).scale(0.4)
        w_mat.next_to(z_group, RIGHT, buff=1)
        self.play(FadeIn(w_group))
        self.wait(1) """

        

        
