from simplified_kyber_pke import *
from dilithium_toy_version import *
from hashlib import shake_256
from dilithium_b_verion import *
import secrets
from kyber_pke import *

n = 4
q = 2**18
lambda_p = 256
k = 3
l = 2
gamma1 = 2**18
gamma2 = 513
tau = 4
eta = 10
Beta = tau*eta
d = 13
omega = 20


def power2round(r, d):
    """
    Power2Round(r, d)
    Input: r in [0, q-1], d in [1, log2(q)].
    Output: (r1, r0) such that r = r1 * 2^d + r0,
    with r0 in [0, 2^d - 1] and r1 = (r - r0) / 2^d.
    """
    # Compute 2^d using pow
    two_to_d = 2**d

    # Step 1: Compute r0 as r mod 2^d
    r0 = r % two_to_d

    # Step 2: Compute r1 as (r - r0) / 2^d (integer division)
    r1 = (r - r0) // two_to_d

    return r1, r0

def decompose(r, alpha, q):
    """
    Decompose(r, alpha) [modified]
    Input: r in [0, q-1], alpha even, q-1 = m * alpha.
    Output: (r1, r0) such that:
        r = r1*alpha + r0 (mod q)
    with -alpha/2 <= r0 <= alpha/2 and 0 <= r1 < m.
    """
    # Step 1: Compute r0 = r mod alpha
    r0 = r % alpha

    # Step 2: Check if (r - r0) = q - 1
    # If so, set r1 = 0 and r0 = r0 - 1
    # Else, r1 = (r - r0) / alpha
    if (r - r0) == (q - 1):
        r1 = 0
        r0 = r0 - 1
    else:
        r1 = (r - r0) // alpha

    # Ensure r0 is in [-alpha/2, alpha/2]
    half_alpha = alpha // 2
    if r0 > half_alpha:
        r0 -= alpha
    return (r1, r0)

def HighBits(r, alpha, q):
    r1, _ = decompose(r, alpha, q)
    return r1

def LowBits(r, alpha, q):
    _, r0 = decompose(r, alpha, q)
    return r0

def MakeHint(r, z, alpha, q):
    """
    MakeHint(r, z, alpha)
    Input: r in [0, q-1], z in [-alpha/2, alpha/2], alpha even.
    Output: h in {0,1}.

    h = 1 if HighBits(r+z, α) ≠ HighBits(r, α)
      = 0 otherwise
    """
    high_r = HighBits(r, alpha, q)
    high_rz = HighBits((r + z) % q, alpha, q)
    h = 1 if high_rz != high_r else 0
    return h

# def UseHint(h, r, alpha, q):
#     """
#     UseHint(h, r, alpha)
#     Input: h in {0,1}, r in [0, q-1].
#     Output: HighBits(r+z, α).

#     Steps:
#     1. (r1, r0) = Decompose(r, α)
#     2. If h=1 and r0>0, r1 = r1+1 mod m
#     3. If h=1 and r0≤0, r1 = r1-1 mod m
#     4. Return r1 (which corresponds to HighBits(r+z, α))
#     """
#     r1, r0 = decompose(r, alpha, q)
#     m = (q - 1) // alpha  # since q-1 = m * α

#     if h == 1:
#         if r0 > 0:
#             r1 = (r1 + 1) % m
#         else:
#             r1 = (r1 - 1) % m

#     return r1  # HighBits(r+z, α)


def UseHint_matrix(h_matrix, r_matrix, alpha, q):
    """
    Adjusts the HighBits (r1) of each coefficient in r_matrix based on the hints in h_matrix.

    Args:
        h_matrix (list of lists): Hint bits for each coefficient in each polynomial.
        r_matrix (list of lists): Polynomials represented as lists of coefficients.
        alpha (int): Decomposition parameter.
        q (int): Modulus.

    Returns:
        list of lists: Adjusted HighBits (r1) for each coefficient in each polynomial.
    """
    m = (q - 1) // alpha  # Number of possible HighBits

    adjusted_highbits = []  # To store the adjusted r1 values

    for poly_idx, (h_poly, r_poly) in enumerate(zip(h_matrix, r_matrix)):
        adjusted_poly = []
        for coeff_idx, (h_bit, r_coeff) in enumerate(zip(h_poly, r_poly)):
            r1, r0 = decompose(r_coeff, alpha, q)

            if h_bit == 1:
                if r0 > 0:
                    r1 = (r1 + 1) % m
                else:
                    r1 = (r1 - 1) % m

            adjusted_poly.append(r1)
        adjusted_highbits.append(adjusted_poly)

    return adjusted_highbits




def power2Round_matrix(t, d):
    """
    Apply power2round to each coefficient of a list of lists t.
    t is a list of lists, for example:
    [[6050, 13869, 10344, 12983],
     [15244, 9568, 13212, 5984],
     [9709, 8653, 323, 16027]]

    Returns two new lists of lists (t1, t0) of the same shape as t,
    where each coefficient has been split using power2round.
    """
    t1 = []
    t0 = []
    for poly in t:
        t1_poly = []
        t0_poly = []
        for coeff in poly:
            r1, r0 = power2round(coeff, d)
            t1_poly.append(r1)
            t0_poly.append(r0)
        t1.append(t1_poly)
        t0.append(t0_poly)
    return t1, t0

def compute_r0(w_minus_cs2):
    result = [[0] * len(w_minus_cs2[0]) for _ in range(len(w_minus_cs2))]
    for i in range(len(w_minus_cs2)):
        for j in range(len(w_minus_cs2[0])):
            _,lowbits = decompose(w_minus_cs2[i][j], 2*gamma2,q)
            result[i][j] = lowbits
    return result


def negate_polynomial(c, q):
    """
    Given a polynomial c (list of coefficients) and a modulus q,
    compute the polynomial -c mod q.
    """
    return [(q - coeff) % q for coeff in c]


def count_ones(h):
    sum = 0
    for i in range(len(h)):
        for j in range (len(h[0])):
            if(h[i][j] == 1):
                sum+=1
    return sum

def HighBits_coeff(coeff, alpha, q):
    r1, _ = decompose(coeff, alpha, q)
    return r1

def decompose(r, alpha, q):
    """
    Decompose(r, alpha) [modified]
    Input: r in [0, q-1], alpha even, q-1 = m * alpha.
    Output: (r1, r0) such that:
        r = r1*alpha + r0 (mod q)
    with -alpha/2 <= r0 <= alpha/2 and 0 <= r1 < m.
    """
    # Compute r0 as r mod alpha
    r0 = r % alpha

    # Check if (r - r0) = q - 1
    if (r - r0) == (q - 1):
        r1 = 0
        r0 = r0 - 1
    else:
        r1 = (r - r0) // alpha

    # Ensure r0 is within [-alpha/2, alpha/2]
    half_alpha = alpha // 2
    if r0 > half_alpha:
        r0 -= alpha

    return (r1, r0)

def HighBits_coeff(coeff, alpha, q):
    """
    Extract HighBits from a single coefficient.
    """
    r1, _ = decompose(coeff, alpha, q)
    return r1

def MakeHint_poly(r, z, alpha, q):
    """
    Compute the hint matrix h for all coefficients of polynomials r and z.
    r and z are lists of polynomials (list of lists).
    For each coefficient pair (r_coeff, z_coeff), compute:
    h[i][j] = 1 if HighBits(r_coeff + z_coeff, alpha) != HighBits(r_coeff, alpha)
              else 0
    """
    h_matrix = []
    # print("In MakeHint_poly")
    # print("r:", r)
    # print("z:", z)
    # print()

    for poly_idx, (poly_r, poly_z) in enumerate(zip(r, z)):
        h_poly = []
        # print(f"Processing Polynomial {poly_idx}:")
        # print("poly_r:", poly_r)
        # print("poly_z:", poly_z)
        for coeff_idx, (rc, zc) in enumerate(zip(poly_r, poly_z)):
            # print(f"  Coefficient {coeff_idx}: rc = {rc}, zc = {zc}")
            # Compute (rc + zc) mod q
            rc_plus_zc = (rc + zc) % q

            # Extract HighBits
            high_r = HighBits_coeff(rc, alpha, q)
            high_rz = HighBits_coeff(rc_plus_zc, alpha, q)

            # Determine hint bit
            h_bit = 1 if high_rz != high_r else 0
            h_poly.append(h_bit)
            # print(f"    high_r = {high_r}, high_rz = {high_rz}, h_bit = {h_bit}")
        h_matrix.append(h_poly)
        # print(f"  h_poly: {h_poly}\n")

    return h_matrix


def generate_keys(q, k, l, n, lambda_param, d):
    xi = generate_xi()
    p, p_prime, K = compute_p_p_prime_K(xi)
    A_matrix = expandA(q,k,l,n,p)
    s1_vec, s2_vec = expandS(k,l,n,eta,p_prime)
    t_vector = compute_t_vector(A_matrix,s1_vec,s2_vec,q)
    t1,t0 = power2Round_matrix(t_vector,d)
    trace_tr = compute_tr(p,t1,lambda_param)
    return p, t_vector, K, trace_tr, s1_vec, s2_vec, t1, t0

def generate_signature(message, q, k, l, n, p, tr, K, s1_vector, s2_vector, t_zero):
    matrix_A = expandA(q, k, l, n, p)
    mu = compute_mu(tr, message)
    deterministic_rnd = generate_deterministic_rnd()
    rho_second = compute_p_prime_prime(K, deterministic_rnd, mu)
    kappa = 0
    found = False
    attempt = 1
    while(not found):
        print(f"[generate_signature] --- Attempt {attempt} ---")
        print("[generate_signature] kappa:", kappa)
        y_vector = expand_mask_with_nonce(rho_second,kappa,l,gamma1)
        print("[generate_signature] y_vector:", y_vector)
        w =  multiply_matrix_by_vector(matrix_A,y_vector, q)
        print("[generate_signature] w:", w)
        w1 = compute_w1(w)
        print("[generate_signature] w1:", w1)
        c_hat = compute_challenge(mu,w1,lambda_p,shake_256)
        print("[generate_signature] c_hat:", c_hat)
        c = sample_in_ball(c_hat, n, tau)
        print("[generate_signature] c (positions):", c)
        c_times_s1 = multiply_polynomial_with_vector(c,s1_vector,q)
        print("[generate_signature] c_times_s1:", c_times_s1)
        z = add_vector_to_vector(y_vector, c_times_s1,q)
        print("[generate_signature] z:", z)
        c_times_s2 = multiply_polynomial_with_vector(c,s2_vector,q)
        print("[generate_signature] c_times_s2:", c_times_s2)
        w_minus_cs2 = compute_w_minus_cs2(w, c_times_s2, q)
        print("[generate_signature] w_minus_cs2 (Signing):", w_minus_cs2)

        r_zero = compute_r0(w_minus_cs2)
        print("[generate_signature] r_zero:", r_zero)
        z_bound = gamma1 - Beta
        r1_bound = gamma2 - Beta
        print("[generate_signature] z_bound:", z_bound, "r1_bound:", r1_bound)
        z_norm = infinity_norm(z)
        r0_norm = infinity_norm(r_zero)
        print("[generate_signature] z_norm:", z_norm, "r0_norm:", r0_norm)
        is_z_valid = z_norm < z_bound
        is_r0_valid = r0_norm < r1_bound
        print("[generate_signature] is_z_valid:", is_z_valid, "is_r0_valid:", is_r0_valid)

        if(is_z_valid and is_r0_valid):
            print("[generate_signature] Valid z found.")
            negative_c = negate_polynomial(c,q)
            print("[generate_signature] negative_c:", negative_c)

            # IMPORTANT: Check that you use t0, not s1_vector, when computing negative_c_times_tzero!
            # According to Dilithium spec:
            # h = MakeHint(..., w - c s2 + c t0)
            # Replace s1_vector with t_zero if needed:
            negative_c_times_tzero = multiply_polynomial_with_vector(negative_c,t_zero,q)
            print("[generate_signature] negative_c_times_t0:", negative_c_times_tzero)
            negative_c_times_tzero_norm = infinity_norm(negative_c_times_tzero)
            is_negative_c_valid = negative_c_times_tzero_norm < gamma2
            print("[c_negative norm] :", negative_c_times_tzero_norm)
            print("[gamma2] :", gamma2)
            if(is_negative_c_valid):
                c_times_tzero = multiply_polynomial_with_vector(c,t_zero,q)
                print("[generate_signature] c_times_t0:", c_times_tzero)

                w_minus_cs2_plus_ctzero = add_vector_to_vector(w_minus_cs2,c_times_tzero,q)
                print("[generate_signature] w_minus_cs2_plus_ctzero:", w_minus_cs2_plus_ctzero)


                h = MakeHint_poly(negative_c_times_tzero,w_minus_cs2_plus_ctzero,2*gamma2,q)
                num_ones = count_ones(h)
                print("[generate_signature] h:", h)
                print("[generate_signature] num_ones in h:", num_ones)
                if num_ones <= omega:
                    print("[generate_signature] Condition met: number of 1's in h is at most ω.")
                    found = True
                else:
                    print("[generate_signature] Condition not met: too many 1's in h.")
        else:
            print("[generate_signature] Conditions not met, continuing attempts.")

        kappa+=l
        attempt+=1

    print("[generate_signature] w1 =", w1)
    print("[generate_signature] c_hat (final):", c_hat)
    print("[generate_signature] z (final):", z)
    print("[generate_signature] h (final):", h)
    print("=== [generate_signature] End ===\n")
    return c_hat,z,h


def verify_signature(message, rho, z, t1_vector, c_hat, h):
    print("[verify_signature] c_hat (from signature):", c_hat)
    print("[verify_signature] h:", h)
    z_norm = infinity_norm(z)
    z_bound = gamma1 - Beta
    is_z_valid = z_norm < z_bound
    print("[verify_signature] z:", z)
    print("[verify_signature] z_norm:", z_norm, "z_bound:", z_bound, "is_z_valid:", is_z_valid)

    num_ones = count_ones(h)
    print("[verify_signature] number of 1's in h:", num_ones, "omega:", omega)
    is_h_valid = num_ones <= omega
    print("[verify_signature] is_h_valid:", is_h_valid)

    if(not is_z_valid or not is_h_valid):
        print("[verify_signature] Signature is not valid (norm or hint check failed).")
        print("=== [verify_signature] End ===\n")
        return None

    matrix_A = expandA(q, k, l, n, rho)
    tr = compute_tr(rho,t1_vector,lambda_p)
    mu = compute_mu(tr,message)

    c = sample_in_ball(c_hat,n,tau)
    print("[verify_signature] c (recomputed from c_hat):", c)

    A_times_z = multiply_matrix_by_vector(matrix_A,z,q)
    print("[verify_signature] A_times_z:", A_times_z)

    c_times_t = multiply_polynomial_with_vector(c,t1_vector,q)
    print("[verify_signature] c_times_t:", c_times_t)
    Az_ct = compute_w_minus_cs2(A_times_z,c_times_t,q)
    # According to Dilithium spec: w1' = UseHint(h, (A z - c t_1) 2^(-d), 2γ2)
    # But the code uses (A z - c t_1 * 2^d) to feed UseHint:
    ct_times_2power_d = multiply_vector_by_scalar(Az_ct,2**d,q)
    print("[verify_signature] ct_times_2power_d:", ct_times_2power_d)
    Az_ct2power_d = compute_w_minus_cs2(A_times_z,ct_times_2power_d,q)
    print("[verify_signature] Az - ct:", Az_ct2power_d)

    w1_prime = UseHint_matrix(h,Az_ct2power_d,2*gamma2,q)
    print("[verify_signature] w1_prime after UseHint:", w1_prime)

    c_tilda_cmp = compute_challenge(mu,w1_prime,lambda_p,shake_256)
    print("[verify_signature] c_tilda_cmp:", c_tilda_cmp)
    print("[verify_signature] c_hat:", c_hat)

    if(c_tilda_cmp == c_hat):
        print("[verify_signature] Valid signature")
    else:
        print("[verify_signature] Invalid signature: c_tilda_cmp does not match c_hat")

    print("=== [verify_signature] End ===\n")





if __name__ == "__main__":
    rho,t_vector,K,tr,s1,s2,t1, t0 = generate_keys(q,k,l,n,lambda_p,d)
    c_tilda,z,h = generate_signature("hi",q,k,l,n,rho,tr,K,s1,s2,t0)
    # verify_signature("hi",rho,z,t1,c_tilda,h)
    # c_tilda, z_vector = generate_signature("hi, i really like python",q,k,l,n,rho,tr,K,s1,s2)
    # r1,r2 = decompose(5566,2*gamma2,q)
    # print(f"Highbits = {r1}")
    # print(f"Lowbits = {r2}")
    # verify_signature("hi, i really like python",rho,z_vector,t_vector,c_tilda)


