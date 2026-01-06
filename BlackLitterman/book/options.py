# /Users/dave/PythonProject1/PythonProject1/PythonProject/BlackLitterman/book/options.py

import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes price for a European option.

    Parameters:
        S (float): spot price
        K (float): strike
        T (float): time to expiry in years
        r (float): risk-free rate (annualised, decimal)
        sigma (float): volatility (annualised, decimal)
        option_type (str): 'call' or 'put'
    Returns:
        float: option price
    """
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes Greeks for a European option.

    Returns:
        dict with delta, gamma, vega, theta
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf = norm.pdf(d1)

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type.lower() == "put":
        delta = norm.cdf(d1) - 1
        theta = (-S * pdf * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}
