import numpy as np
from scipy.linalg import eigh, inv
from scipy.stats import truncnorm, norm, beta
import math

def K_builder(dist, l):
    n = dist.shape[0]
    K = np.exp(-dist * dist / (2.0 * l * l))
    eigval, eigvec = eigh(K)
    eigval = np.maximum(eigval, 1e-8)
    K = eigvec @ np.diag(eigval) @ inv(eigvec)
    return K

def which_stop(x):
    idx = np.where(x == -1)[0]
    if len(idx) > 0:
        return x[:idx[0]]
    return x

def r_truncnorm(mu, sigma, lower, upper):
    assert sigma > 0
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

def boost(Y, dist, nei, s, iter, burn, init_b_sigma, init_h, update_prop):
    # Read data information
    n, p = Y.shape
    
    # Calculate the minimum and maximum distances so as to set up prior setting
    dist_temp = np.copy(dist)
    np.fill_diagonal(dist_temp, np.max(dist))
    t_min = max(np.min(dist_temp), 1e-8) # Avoid zero distance
    t_max = np.max(dist_temp)
    
    # Set hyperparameters
    a_pi, b_pi = 1, 1
    a_phi, b_phi = 0.001, 0.001
    omega = 0.05
    h = init_h
    a_sigma = 3.0
    b_sigma = init_b_sigma
    a_l = t_min / 2
    b_l = 2 * t_max
    
    # Set algorithm settings
    nb, zi = True, True
    E = max(1, int(p * update_prop))
    F = max(1, int(n * update_prop))
    tau_log_phi, tau_log_lambda, tau_log_l = 1.0, 0.1, 0.1
    
    # Set temporary variables
    n_eff = np.zeros(p, dtype=int)
    H_sum_temp = np.zeros(n, dtype=int)
    K_inv_sum = np.zeros(p)
    log_K_det = np.zeros(p)
    G1, G2 = np.zeros(p), np.zeros(p)
    loglambda_null = np.zeros(p)
    
    H = np.zeros((n, p), dtype=int)
    pi = np.full(n, 0.5)
    phi = np.full(p, 10.0)
    logLambda = np.full((n, p), np.log(np.mean(Y)))
    gamma = np.zeros(p, dtype=int)
    l = np.full(p, t_min / 2)
    
    # MCMC storage
    H_ppi = np.zeros((n, p))
    H_sum = np.zeros(iter)
    gamma_ppi = np.zeros(p)
    gamma_sum = np.zeros(iter)
    logBF = np.zeros((iter, p))
    pi_store = np.zeros((iter, n))
    phi_store = np.zeros((iter, p))
    omega_store = np.zeros(iter)
    gamma_store = np.zeros((iter, p))
    l_store = np.zeros((iter, p))
    
    # Initialization
    gamma_sum_temp = 0
    for j in range(p):
        for i in range(n):
            if Y[i, j] == 0:
                H[i, j] = 0
                H_sum_temp[i] += H[i, j]
        n_eff[j] = np.sum(H[:, j] == 0)
        if gamma[j] == 1:
            gamma_sum_temp += 1
            K = K_builder(dist, l[j])
            K = K[np.ix_(H[:, j] == 0, H[:, j] == 0)]
            eigval, eigvec = eigh(K)
            log_K_det[j] = np.sum(np.log(eigval))
            temp = inv(K) @ np.ones((n_eff[j], 1))
            K_inv_sum[j] = np.sum(temp)
            temp = inv(K) @ logLambda[np.ix_(H[:, j] == 0, [j])]
            G1[j] = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ temp)
            G2[j] = np.sum(temp.T @ np.ones((n_eff[j], n_eff[j])) @ temp / (K_inv_sum[j] + 1.0 / h))
        G1_temp = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ logLambda[np.ix_(H[:, j] == 0, [j])])
        G2_temp = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ np.ones((n_eff[j], n_eff[j])) @ logLambda[np.ix_(H[:, j] == 0, [j])] / (n_eff[j] + 1.0 / h))
        loglambda_null[j] = -np.log(n_eff[j] + 1.0 / h) / 2.0 - (a_sigma + n_eff[j] / 2.0) * np.log(b_sigma + G1_temp / 2.0 - G2_temp / 2.0)
    
    # MCMC
    accept_phi, accept_lambda, accept_gamma, accept_l = 0, 0, 0, 0
    try_phi, try_lambda, try_gamma, try_l = 0, 0, 0, 0
    count = 0

    for it in range(iter):
        # Update H
        if zi:
            for j in range(p):
                count_temp = 0
                for i in range(n):
                    if Y[i, j] == 0:
                        if nb:
                            prob_temp = np.array([phi[j] * (np.log(phi[j]) - np.log(s[i] * np.exp(logLambda[i, j]) + phi[j])) + np.log(1 - pi[i]), np.log(pi[i])])
                        else:
                            prob_temp = np.array([-s[i] * np.exp(logLambda[i, j]) + np.log(1 - pi[i]), np.log(pi[i])])
                        max_temp = np.max(prob_temp)
                        prob_temp = np.exp(prob_temp - max_temp)
                        prob_temp /= np.sum(prob_temp)
                        H_temp = H[i, j]
                        H[i, j] = np.random.binomial(1, prob_temp[1])
                        H_sum_temp[i] = H_sum_temp[i] - H_temp + H[i, j]
                        if H[i, j] != H_temp:
                            count_temp += 1
                if count_temp > 0:
                    n_eff[j] = np.sum(H[:, j] == 0)
                    if gamma[j] == 1:
                        K = K_builder(dist, l[j])
                        K = K[np.ix_(H[:, j] == 0, H[:, j] == 0)]
                        eigval, eigvec = eigh(K)
                        log_K_det[j] = np.sum(np.log(eigval))
                        temp = inv(K) @ np.ones((n_eff[j], 1))
                        K_inv_sum[j] = np.sum(temp)
                        temp = inv(K) @ logLambda[np.ix_(H[:, j] == 0, [j])]
                        G1[j] = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ temp)
                        G2[j] = np.sum(temp.T @ np.ones((n_eff[j], n_eff[j])) @ temp / (K_inv_sum[j] + 1.0 / h))
                    G1_temp = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ logLambda[np.ix_(H[:, j] == 0, [j])])
                    G2_temp = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ np.ones((n_eff[j], n_eff[j])) @ logLambda[np.ix_(H[:, j] == 0, [j])] / (n_eff[j] + 1.0 / h))
                    loglambda_null[j] = -np.log(n_eff[j] + 1.0 / h) / 2.0 - (a_sigma + n_eff[j] / 2.0) * np.log(b_sigma + G1_temp / 2.0 - G2_temp / 2.0)

        # Update phi
        if nb:
            for j in range(p):
                phi_temp = np.exp(r_truncnorm(np.log(phi[j]), tau_log_phi, np.log(1), np.log(100)))
                hastings = 0
                for i in range(n):
                    if H[i, j] == 0:
                        hastings += phi_temp * np.log(phi_temp) - math.lgamma(phi_temp) + math.lgamma(phi_temp + Y[i, j]) - (phi_temp + Y[i, j]) * np.log(phi_temp + s[i] * np.exp(logLambda[i, j]))
                        hastings -= phi[j] * np.log(phi[j]) - math.lgamma(phi[j]) + math.lgamma(phi[j] + Y[i, j]) - (phi[j] + Y[i, j]) * np.log(phi[j] + s[i] * np.exp(logLambda[i, j]))
                hastings += (a_phi - 1) * np.log(phi_temp) - b_phi * phi_temp
                hastings -= (a_phi - 1) * np.log(phi[j]) - b_phi * phi[j]
                if it > burn:
                    try_phi += 1
                if hastings >= np.log(np.random.rand()):
                    phi[j] = phi_temp
                    if it > burn:
                        accept_phi += 1

        # Update Lambda
        for j in range(p):
            if gamma[j] == 1:
                K = K_builder(dist, l[j])
            for f in range(F):
                i = np.random.randint(n)
                loglambda_temp = np.random.normal(logLambda[i, j], tau_log_lambda)
                logLambda_temp = np.copy(logLambda[:, j])
                logLambda_temp_0 = logLambda_temp[which_stop(nei[i, :])]
                logLambda_temp[i] = loglambda_temp
                logLambda_temp = logLambda_temp[which_stop(nei[i, :])]
                n_temp = which_stop(nei[i, :]).size
                hastings = 0
                if H[i, j] == 0:
                    if nb:
                        hastings += Y[i, j] * (np.log(s[i]) + loglambda_temp) - (phi[j] + Y[i, j]) * np.log(phi[j] + s[i] * np.exp(loglambda_temp))
                        hastings -= Y[i, j] * (np.log(s[i]) + logLambda[i, j]) - (phi[j] + Y[i, j]) * np.log(phi[j] + s[i] * np.exp(logLambda[i, j]))
                    else:
                        hastings += Y[i, j] * (np.log(s[i]) + loglambda_temp) - s[i] * np.exp(loglambda_temp)
                        hastings -= Y[i, j] * (np.log(s[i]) + logLambda[i, j]) - s[i] * np.exp(logLambda[i, j])
                if gamma[j] == 0:
                    G1_temp = np.sum(logLambda_temp.T @ logLambda_temp)
                    G2_temp = np.sum(logLambda_temp.T @ np.ones((n_temp, n_temp)) @ logLambda_temp / (n_temp + 1.0 / h))
                    loglambda_null_temp = -np.log(n_temp + 1.0 / h) / 2.0 - (a_sigma + n_temp / 2.0) * np.log(b_sigma + G1_temp / 2.0 - G2_temp / 2.0)
                    hastings += loglambda_null_temp
                    G1_temp = np.sum(logLambda_temp_0.T @ logLambda_temp_0)
                    G2_temp = np.sum(logLambda_temp_0.T @ np.ones((n_temp, n_temp)) @ logLambda_temp_0 / (n_temp + 1.0 / h))
                    loglambda_null_temp = -np.log(n_temp + 1.0 / h) / 2.0 - (a_sigma + n_temp / 2.0) * np.log(b_sigma + G1_temp / 2.0 - G2_temp / 2.0)
                    hastings -= loglambda_null_temp
                else:
                    K_temp = K[np.ix_(which_stop(nei[i, :]), which_stop(nei[i, :]))]
                    temp = inv(K_temp) @ np.ones((n_temp, 1))
                    K_inv_sum_temp = np.sum(temp)
                    temp = inv(K_temp) @ logLambda_temp
                    G1_temp = np.sum(logLambda_temp.T @ temp)
                    G2_temp = np.sum(temp.T @ np.ones((n_temp, n_temp)) @ temp / (K_inv_sum_temp + 1.0 / h))
                    hastings += - (a_sigma + n_temp / 2.0) * np.log(b_sigma + G1_temp / 2.0 - G2_temp / 2.0)
                    temp = inv(K_temp) @ logLambda_temp_0
                    G1_temp = np.sum(logLambda_temp_0.T @ temp)
                    G2_temp = np.sum(temp.T @ np.ones((n_temp, n_temp)) @ temp / (K_inv_sum_temp + 1.0 / h))
                    hastings -= - (a_sigma + n_temp / 2.0) * np.log(b_sigma + G1_temp / 2.0 - G2_temp / 2.0)
                if it > burn:
                    try_lambda += 1
                if hastings >= np.log(np.random.rand()):
                    logLambda[i, j] = loglambda_temp
                    if it > burn:
                        accept_lambda += 1
            if gamma[j] == 1:
                K = K[np.ix_(H[:, j] == 0, H[:, j] == 0)]
                temp = inv(K) @ logLambda[np.ix_(H[:, j] == 0, [j])]
                G1[j] = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ temp)
                G2[j] = np.sum(temp.T @ np.ones((n_eff[j], n_eff[j])) @ temp / (K_inv_sum[j] + 1.0 / h))
            G1_temp = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ logLambda[np.ix_(H[:, j] == 0, [j])])
            G2_temp = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ np.ones((n_eff[j], n_eff[j])) @ logLambda[np.ix_(H[:, j] == 0, [j])] / (n_eff[j] + 1.0 / h))
            loglambda_null[j] = -np.log(n_eff[j] + 1.0 / h) / 2.0 - (a_sigma + n_eff[j] / 2.0) * np.log(b_sigma + G1_temp / 2.0 - G2_temp / 2.0)
        
        # Update gamma
        for e in range(E):
            j = np.random.randint(p)
            gamma_temp = 1 - gamma[j]
            if gamma_temp == 0:  # Delete
                hastings = 0
                hastings -= -log_K_det[j] / 2.0 - np.log(K_inv_sum[j] + 1.0 / h) / 2.0 - (a_sigma + n_eff[j] / 2.0) * np.log(b_sigma + G1[j] / 2.0 - G2[j] / 2.0)
                hastings += loglambda_null[j]
                logBF[it, j] = -hastings
                hastings += - (np.log(l[j]) - np.log(t_min / 2)) ** 2 / (2.0 * 10.0 * tau_log_l ** 2)
                hastings -= - np.log(b_l - a_l)
                hastings += np.log(1 - omega) - np.log(omega)
            else:  # Add
                hastings = 0
                hastings -= loglambda_null[j]
                l_temp = np.exp(r_truncnorm(np.log(t_min / 2), 10.0 * tau_log_l, np.log(a_l), np.log(b_l)))
                K = K_builder(dist, l_temp)
                K = K[np.ix_(H[:, j] == 0, H[:, j] == 0)]
                eigval, eigvec = eigh(K)
                log_K_det_temp = np.sum(np.log(eigval))
                temp = inv(K) @ np.ones((n_eff[j], 1))
                K_inv_sum_temp = np.sum(temp)
                temp = inv(K) @ logLambda[np.ix_(H[:, j] == 0, [j])]
                G1_temp = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ temp)
                G2_temp = np.sum(temp.T @ np.ones((n_eff[j], n_eff[j])) @ temp / (K_inv_sum_temp + 1.0 / h))
                hastings += - log_K_det_temp / 2.0 - np.log(K_inv_sum_temp + 1.0 / h) / 2.0 - (a_sigma + n_eff[j] / 2.0) * np.log(b_sigma + G1_temp / 2.0 - G2_temp / 2.0)
                logBF[it, j] = hastings
                hastings -= - (np.log(l_temp) - np.log(t_min / 2)) ** 2 / (2.0 * 10.0 * tau_log_l ** 2)
                hastings += - np.log(b_l - a_l)
                hastings -= np.log(1 - omega) - np.log(omega)
            if it > burn:
                try_gamma += 1
            if hastings >= np.log(np.random.rand()):
                gamma[j] = gamma_temp
                if gamma_temp == 1:
                    gamma_sum_temp += 1
                    l[j] = l_temp
                    log_K_det[j] = log_K_det_temp
                    K_inv_sum[j] = K_inv_sum_temp
                    G1[j] = G1_temp
                    G2[j] = G2_temp
                else:
                    gamma_sum_temp -= 1
                if it > burn:
                    accept_gamma += 1
        
        # Update kernel parameter l
        for j in range(p):
            if gamma[j] == 1:
                hastings = 0
                hastings -= - log_K_det[j] / 2.0 - np.log(K_inv_sum[j] + 1.0 / h) / 2.0 - (a_sigma + n_eff[j] / 2.0) * np.log(b_sigma + G1[j] / 2.0 - G2[j] / 2.0)
                l_temp = np.exp(r_truncnorm(np.log(l[j]), tau_log_l, np.log(a_l), np.log(b_l)))
                K = K_builder(dist, l_temp)
                K = K[np.ix_(H[:, j] == 0, H[:, j] == 0)]
                eigval, eigvec = eigh(K)
                log_K_det_temp = np.sum(np.log(eigval))
                temp = inv(K) @ np.ones((n_eff[j], 1))
                K_inv_sum_temp = np.sum(temp)
                temp = inv(K) @ logLambda[np.ix_(H[:, j] == 0, [j])]
                G1_temp = np.sum(logLambda[np.ix_(H[:, j] == 0, [j])].T @ temp)
                G2_temp = np.sum(temp.T @ np.ones((n_eff[j], n_eff[j])) @ temp / (K_inv_sum_temp + 1.0 / h))
                hastings += - log_K_det_temp / 2.0 - np.log(K_inv_sum_temp + 1.0 / h) / 2.0 - (a_sigma + n_eff[j] / 2.0) * np.log(b_sigma + G1_temp / 2.0 - G2_temp / 2.0)
                if it > burn:
                    try_l += 1
                if hastings >= np.log(np.random.rand()):
                    l[j] = l_temp
                    log_K_det[j] = log_K_det_temp
                    K_inv_sum[j] = K_inv_sum_temp
                    G1[j] = G1_temp
                    G2[j] = G2_temp
                    if it > burn:
                        accept_l += 1
            else:
                l[j] = 0
        
        # Monitor the process
        if (it * 100 / iter) == count:
            print(f"{count}% has been done")
            count += 10
        H_sum[it] = np.sum(H_sum_temp)
        gamma_sum[it] = gamma_sum_temp
        pi_store[it, :] = pi
        phi_store[it, :] = phi
        gamma_store[it, :] = gamma
        omega_store[it] = omega
        l_store[it, :] = l
        if it >= burn:
            gamma_ppi += gamma
            H_ppi += H
    
    gamma_ppi /= (iter - burn)
    H_ppi /= (iter - burn)
    accept_phi /= try_phi
    accept_lambda /= try_lambda
    accept_gamma /= try_gamma
    accept_l /= try_l
    
    return {
        "logBF": logBF,
        "H_sum": H_sum,
        "H_ppi": H_ppi,
        "pi": pi_store,
        "phi": phi_store,
        "logLambda": logLambda,
        "gamma_sum": gamma_sum,
        "gamma_ppi": gamma_ppi,
        "gamma": gamma_store,
        "omega": omega_store,
        "l": l_store,
        "accept_lambda": accept_lambda,
        "accept_gamma": accept_gamma,
        "accept_l": accept_l
    }

