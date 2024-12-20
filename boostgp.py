#########################fucntions.R
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm, truncnorm, cauchy
from numpy.linalg import inv, cholesky, LinAlgError
import matplotlib.pyplot as plt
import seaborn as sns

from .boost import boost

def quality_controller(count, loc, cutoff_sample=10, cutoff_feature=0.1):
    n, p = count.shape
    index = np.where(np.sum(count, axis=1) >= cutoff_sample)[0]
    count = count[index, :]
    loc = loc[index, :]
    index = np.where(np.sum(count != 0, axis=0) >= count.shape[0] * cutoff_feature)[0]
    count = count[:, index]
    return {'count': count, 'loc': loc}

def distance(loc, cutoff):
    dist = squareform(pdist(loc, metric='euclidean'))
    n = dist.shape[0]
    neighbors = np.zeros((n, n), dtype=int)
    for i in range(n):
        temp = np.where(dist[i, :] <= cutoff)[0]
        neighbors[i, :len(temp)] = temp
    neighbors = neighbors[:, np.sum(neighbors, axis=0) != 0]
    return {'dist': dist, 'nei': neighbors}

def gp_generator(dist, mean=None, sigma=1, kernel='SE', c=0, l=1, alpha=1, p=1, psi=1, seed=1, y_known=None):
    np.random.seed(seed)
    n = dist.shape[0]
    mean = mean if mean is not None else np.zeros(n)
    y_known = y_known if y_known is not None else np.full(n, np.nan)
    
    if kernel == 'C':
        if c >= 1 or c < 0:
            raise ValueError("Invalid value of c!")
        K = np.full((n, n), c)
        np.fill_diagonal(K, 1)
    elif kernel == 'SE':
        K = np.exp(-(dist ** 2) / (2 * l ** 2))
    elif kernel == 'RQ':
        K = (1 + (dist ** 2) / (2 * alpha * l ** 2)) ** -alpha
    elif kernel == 'MA':
        nu = p + 0.5
        temp = sum(np.math.factorial(p + i) / (np.math.factorial(i) * np.math.factorial(p - i)) * (np.sqrt(8 * nu) * dist / l) ** (p - i) for i in range(p + 1))
        K = np.exp(-np.sqrt(2 * nu) * dist / l) * np.math.gamma(p + 1) / np.math.gamma(2 * p + 1) * temp
    elif kernel == 'PE':
        K = np.exp(-2 * np.sin(np.pi * dist / psi) ** 2 / l ** 2)
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals[eigvals < 0] = 0
        K = eigvecs @ np.diag(eigvals) @ inv(eigvecs)
    elif kernel == 'COS':
        K = np.cos(2 * np.pi * dist / psi)
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals[eigvals < 0] = 0
        K = eigvecs @ np.diag(eigvals) @ inv(eigvecs)
    else:
        raise ValueError("Invalid kernel name!")
    
    if np.isnan(y_known).all():
        y = np.random.multivariate_normal(mean, sigma ** 2 * K)
    else:
        y = np.full(n, np.nan)
        index_known = np.where(~np.isnan(y_known))[0]
        y[index_known] = y_known[index_known]
        mean_unknown = mean[~np.isnan(y_known)] + K[~np.isnan(y_known)][:, index_known] @ inv(K[index_known][:, index_known]) @ (y_known[index_known] - mean[index_known])
        K_unknown = K[~np.isnan(y_known)][:, ~np.isnan(y_known)] - K[~np.isnan(y_known)][:, index_known] @ inv(K[index_known][:, index_known]) @ K[index_known][:, ~np.isnan(y_known)]
        y[~np.isnan(y_known)] = np.random.multivariate_normal(mean_unknown, sigma ** 2 * K_unknown)
    
    return {'y': y, 'K': K, 'mean': mean, 'sigma': sigma}

def anscombe_transformer(count):
    var = np.var(count, axis=0)
    mean = np.mean(count, axis=0)
    phi = np.sum(mean ** 2 * (var - mean)) / np.sum(mean ** 4)
    return np.log(count + 1 / phi)

def normalize(count, mode='linear', alpha=5):
    if mode == 'linear':
        return (count - np.min(count)) / (np.max(count) - np.min(count))
    elif mode == 'sigmoid':
        return 1 / (1 + np.exp(-alpha * (count - np.median(count))))
    else:
        raise ValueError("Invalid normalization mode!")

def loglklh(y, dist, l, X=None, h=1000, alpha_sigma=3, beta_sigma=1):
    p = len(y)
    R = 1 if X is None else X.shape[1]
    H = np.exp(-(dist ** 2) / (2 * l ** 2)) if l != 0 else np.eye(p)
    
    try:
        Hi = inv(cholesky(H).T @ cholesky(H))
        if R == 1:
            G = 1 / h + np.sum(Hi)
            Gi = 1 / G
            F = Hi - (Gi * Hi @ np.ones((p, p)) @ Hi)
        else:
            G = np.eye(R) / h + X.T @ Hi @ X
            Gi = inv(cholesky(G).T @ cholesky(G))
            F = Hi - (Hi @ X @ Gi @ X.T @ Hi)
        res = -0.5 * np.linalg.slogdet(H)[1] - (alpha_sigma + p / 2) * np.log(beta_sigma + (y.T @ F @ y) / 2)
        if R == 1:
            res -= 0.5 * np.log(G)
        else:
            res -= 0.5 * np.linalg.slogdet(G)[1]
    except LinAlgError:
        res = -np.inf

    return float(res)

def combine_pval(pvals, weights=None):
    pvals[pvals == 0] = 5.55e-17
    pvals[pvals >= 1 - 1e-3] = 0.99

    n_pval, n_gene = pvals.shape
    if weights is None:
        weights = np.ones((n_pval, n_gene)) / n_pval

    if weights.shape != pvals.shape:
        raise ValueError("The dimensions of weights do not match that of combined p-values")

    Cstat = np.tan((0.5 - pvals) * np.pi)
    wCstat = weights * Cstat
    Cbar = np.sum(wCstat, axis=0)
    combined_pval = 1.0 - cauchy.cdf(Cbar)
    combined_pval[combined_pval <= 0] = 5.55e-17

    return combined_pval

def plot_expr(count, loc, main=""):
    data = pd.DataFrame({'expr': count.flatten(), 'x': loc[:, 0], 'y': loc[:, 1]})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='x', y='y', hue='expr', data=data, palette="Spectral", legend=False)
    plt.title(main)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

#########################boost.gp.R
import numpy as np
import pandas as pd
import time
from scipy.stats import chisquare
from functools import reduce


def boost_gp(Y, loc, iter=1000, burn=500, size_factor=None, init_b_sigma=None, init_h=1, update_prop=0.2, chain=1, return_lambda=False):
    res_list = []

    for i in range(chain):
        print(f"Chain {i+1}")
        
        # Parameter initialization
        if size_factor is None:
            size_factor = np.ones(Y.shape[0])
        
        if init_b_sigma is None:
            init_b_sigma = round(np.mean([np.var(np.log(x[x > 0])) for x in Y.T]) * 2, 3)
        
        dist_mat = distance(loc, 2) # Obtain the distance and neighbor matrix
        
        # Run model
        start_time = time.time()
        res = boost(Y=Y, dist=dist_mat['dist'], nei=dist_mat['nei'], 
                    iter=iter, burn=burn, s=size_factor, 
                    init_b_sigma=init_b_sigma, init_h=init_h, update_prop=update_prop)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"runtime is {run_time:.2f}s")
        
        # Significance
        loglr = res['logBF'][burn:]
        loglr[loglr < 0] = 0
        
        n = int((iter - burn) * update_prop)
        bf = np.array([np.mean(np.sort(x)[-n:]) for x in loglr.T])
        p_vals = chisquare(bf * 2, df=1)[1]
        
        # Parameters
        l = res['l'][burn:]
        l = np.array([np.sum(l[:, x] * loglr[:, x]) / np.sum(loglr[:, x]) for x in range(l.shape[1])])
        l[np.isnan(l)] = 0
        
        # LogLambda
        logLambda = np.round(res['logLambda'], 3)
        
        # Result
        result = np.vstack((l, bf, res['gamma_ppi'], p_vals, np.full(Y.shape[1], run_time / Y.shape[1])))
        colnames = list(Y.columns)
        result_df = pd.DataFrame(result.T, columns=['l', 'BF', 'PPI', 'pval', 'time'], index=colnames)
        
        if return_lambda:
            logLambda_df = pd.DataFrame(logLambda, columns=colnames)
            result_df = pd.concat([result_df, logLambda_df.T])
        
        res_list.append(result_df)
    
    res_ave = reduce(lambda x, y: x + y, res_list) / chain
    pvals = np.array([df['pval'].values for df in res_list])
    res_ave['pval'] = combine_pval(pvals)
    res_ave['time'] = res_ave['time'] * chain
    return res_ave
