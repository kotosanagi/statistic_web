import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import math
import pandas as pd

def binomial(x, n, p):
    """
    二項分布
    x : 確率変数
    n : 試行回数
    p : 成功確率
    """
    ret = comb(n, x) * math.pow(p, x) * math.pow(1-p, n-x)
    return ret

def poisson(x, lmd):
    """
    ポアソン分布
    x : 確率変数
    lmd : 単位時間に起こる確率(=np)
    """
    ret = (math.pow(math.e, (-1)*lmd)) * (math.pow(lmd, k)) / (math.factorial(k))
    return ret

def geometric(x, p):
    """
    幾何分布
    x : 確率変数
    p : 成功確率
    """
    ret = (math.pow(1-p, x-1)) * p
    return ret 

def normdist(x, mu, sigma):
    """
    正規分布関数
    x : 確率変数
    mu : 平均
    sigma : 標準偏差
    """
    coef = 1/((math.sqrt(2*math.pi)) * sigma) # 係数
    exponent = (-1) * (pow((x-mu), 2)) / (2 * pow(sigma, 2)) # 指数
    ret = coef * pow(math.e, exponent)
    return ret

def std_normdist(x):
    """
    標準正規分布関数
    x : 確率変数
    mu : 平均 = 0
    sigma : 標準偏差 = 1
    """
    coef = 1/(math.sqrt(2*math.pi)) # 係数
    exponent = (-1) * (pow(x, 2)) / 2 # 指数
    ret = coef * pow(math.e, exponent)
    return ret

def standardize(x, mu, sigma):
    """
    標準化
    x : 確率変数
    mu : 平均
    sigma : 標準偏差
    """
    ret = (x - mu) / sigma
    return ret

def deviation_score(x, mu, sigma):
    """
    偏差値
    x : 確率変数
    mu : 平均
    sigma : 標準偏差
    """
    z = (x - mu) / sigma
    ret = (z * 10) + 50
    return ret
