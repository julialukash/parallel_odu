import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from numpy.linalg import norm as euclidean_norm
from scipy.stats import entropy

def kl_dist(p, q):
    return entropy(p, q)

def kl_sym_dist(p, q):
    return 0.5 * (entropy(p, q) + entropy(q, p)) 

def jaccard_dist(p, q, top_words_count=15):
    c1_top_words = p.sort_values()[::-1][0:top_words_count]
    c2_top_words = q.sort_values()[::-1][0:top_words_count]
    return 1 - 1.0 * len(c1_top_words.index.intersection(c2_top_words.index)) / len(c1_top_words.index.union(c2_top_words.index))

def euc_dist(p, q):
    return euclidean_norm(p - q)

def euc_dist_grad(b, A, x):
    x = x.reshape(1, -1)
    b = b.reshape(1, -1)
    norm = euc_dist(A.dot(x), b)
    res = A.T.dot(A.dot(x) - b)
    if norm != 0:
        res = res / norm
    return res

def cos_dist(p, q):
    p = p.values.reshape(1, -1)
    q = q.values.reshape(1, -1)
    return cosine_distances(p, q)[0][0]

def cos_dist_grad(b, A, x):
    x = x.reshape(1, -1)
    b = b.reshape(1, -1)
    y = A.dot(x)
    u = b.T.dot(y) # number
    deriv_u = A.T.dot(b) * x
    v = euclidean_norm(y) * euclidean_norm(b)
    nom = deriv_u * v - A.T.dot(A).dot(x) * u[0][0] * euclidean_norm(b) / euclidean_norm(y)
    denom = v * v
    if denom != 0:
        res = nom / denom
    else:
        res = nom
    return -res

def hellinger_dist(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

def hellinger_dist_grad(b, A, x):
    y = A.dot(x)
    nom = np.divide(np.sqrt(y) - np.sqrt(b), np.sqrt(y)).dot(A)
    denom = 2 * hellinger_dist(y, b) * np.sqrt(2)
    res = nom / denom
    return res

def hellinger_dist_grad_nan(b, A, x):
    y = A.dot(x)
    tmp = np.divide(np.sqrt(y) - np.sqrt(b), np.sqrt(y))
    tmp[np.isnan(tmp)] = 0
    nom = tmp.dot(A)
    denom = 2 * hellinger_dist(y, b) * np.sqrt(2)
    res = nom / denom
    return res

def hellinger_dist_grad_eps(b, A, x):
    y = A.dot(x)
    y[y == 0] = 1e-3
    tmp = np.divide(np.sqrt(y) - np.sqrt(b), np.sqrt(y))
    nom = tmp.dot(A)
    denom = 2 * hellinger_dist(y, b) * np.sqrt(2)
    res = nom / denom
    return res
