import stumpy as sp
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def matrix_profile(query, whole_series, top_k: int = 100):
    distance_profile = sp.mass(query, whole_series)
    idxs = np.argsort(distance_profile)
    return idxs[:top_k]

def plot(y, lim = 3000):
    plt.figure(figsize=(10, 3), dpi = 100)
    plt.plot(y[:lim])
    plt.xlabel('t (second)')
    plt.ylabel('Amplitude')
    plt.title("Original Series")
    # plt.axis([0, 1, -1.2, 1.2])

    plt.show()
    
def plot_matrix_profile(y, query_length, idxs, lim = 3000):
    plt.figure(figsize=(10, 3), dpi = 100)
    plt.plot(y[:lim])
    plt.xlabel('t (second)')
    plt.ylabel('Amplitude')
    plt.title("Matrix Profile")
    x = np.arange(0, len(y))
    for idx in idxs:
        plt.plot(x[idx: idx + query_length], y[idx: idx + query_length], c = 'r')
    plt.show()
    
def generate_origin(length = 50000, period = 3e-1, lim = 1000):
    x = np.linspace(0, lim, length, endpoint=False)
    y = np.cos(2 * np.pi * period * x)
    residual = np.random.normal(loc=0.0, scale=1.0, size=length) * 6e-2
    y += residual
    return x, y

def generate_shapelet(length = 50000, period = 3e-1, lim = 1000):
    x, y = generate_origin(length, period, lim)
    t = int(np.ceil(1/period * length / lim))
    x -= (t / 4)
    z = signal.square(2 * np.pi * period * x)
    residual = np.random.normal(loc=0.0, scale=1.0, size=length) * 6e-2
    z += residual
    y[0: t] = z[0: t]
    y[5 * t: 6 * t] = z[5 * t: 6 * t]
    y[10 * t: 11 * t] = z[10 * t: 11 * t]
    return x, y

def generate_seasonal(length = 50000, period = 3e-1, lim = 1000):
    x, y = generate_origin(length, period, lim)
    t = int(np.ceil(1/period * length / lim))
    z = np.cos(2 * np.pi * period * 2 * x)
    residual = np.random.normal(loc=0.0, scale=1.0, size=length) * 6e-2
    z += residual
    y[0: t] = z[0: t]
    y[5 * t: 6 * t] = z[5 * t: 6 * t]
    y[10 * t: 11 * t] = z[10 * t: 11 * t]
    return x, y

def generate_trend(length = 50000, period = 3e-1, lim = 1000):
    x, y = generate_origin(length, period, lim)
    t = int(np.ceil(1/period * length / lim))
    z = 0.2 * x - 6
    z += y
    residual = np.random.normal(loc=0.0, scale=1.0, size=length) * 6e-2
    z += residual
    y[0: 3*t] = z[7 * t: 10 * t]
    y[10 * t: 13 * t] = z[7 * t: 10 * t]
    y[20 * t: 23 * t] = z[7 * t: 10 * t]
    return x, y