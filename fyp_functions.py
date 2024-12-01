import numpy as np
from itertools import product
import pandas as pd

def independent_market_mechanism(n, m, P):
    def compute_sum_medians(t):
        phantoms = np.clip(t * (n - np.arange(n + 1)), 0, 1)
        phantoms = phantoms.tolist()
        total_median = 0
        medians = []
        for i in range(m):
            votes_i = P[:, i].tolist()
            combined_list = votes_i + phantoms
            median_i = np.median(combined_list)
            medians.append(median_i)
            total_median += median_i
        return total_median, medians

    def find_t():
        t_low = 0.0
        t_high = 1.0
        epsilon = 1e-8
        max_iterations = 1000

        for _ in range(max_iterations):
            t_mid = (t_low + t_high) / 2.0
            sum_mid, _ = compute_sum_medians(t_mid)
            if abs(sum_mid - 1.0) < epsilon:
                return t_mid
            elif sum_mid < 1.0:
                t_low = t_mid
            else:
                t_high = t_mid
        raise ValueError("Failed to converge to a solution for t. IMM.")

    t = find_t()
    phantoms = np.clip(t * (n - np.arange(n + 1)), 0, 1)
    phantoms = phantoms.tolist()
    allocation = []
    detailed_info = []
    for i in range(m):
        votes_i = P[:, i].tolist()
        combined_list = votes_i + phantoms
        median_i = np.median(combined_list)
        allocation.append(median_i)
        detailed_info.append({
            'alternative': i + 1,
            'votes': [round(v, 3) for v in votes_i],
            'phantoms': phantoms,
            'combined_list': sorted([round(x, 3) for x in combined_list]),
            'median': median_i
        })
    allocation = np.array(allocation)
    return allocation, detailed_info

def welfare_maximizing_phantom_mechanism(n, m, P):
    def compute_phantoms(t):
        k_values = np.arange(n + 1)
        fk = np.zeros(n + 1)
        condition1 = t <= k_values / (n + 1)
        condition2 = (t > k_values / (n + 1)) & (t <= (k_values + 1) / (n + 1))
        condition3 = t > (k_values + 1) / (n + 1)

        fk[condition1] = 0
        fk[condition2] = (n + 1) * t - k_values[condition2]
        fk[condition3] = 1
        return fk.tolist()

    def compute_sum_medians(t):
        phantoms = compute_phantoms(t)
        total_median = 0
        medians = []
        for i in range(m):
            votes_i = P[:, i].tolist()
            combined_list = votes_i + phantoms
            median_i = np.median(combined_list)
            medians.append(median_i)
            total_median += median_i
        return total_median, medians

    def find_t():
        t_low = 0.0
        t_high = 1.0
        epsilon = 1e-8
        max_iterations = 1000

        for _ in range(max_iterations):
            t_mid = (t_low + t_high) / 2.0
            sum_mid, _ = compute_sum_medians(t_mid)
            if abs(sum_mid - 1.0) < epsilon:
                return t_mid
            elif sum_mid > 1.0:
                t_high = t_mid
            else:
                t_low = t_mid
        raise ValueError("Failed to converge to a solution for t. WMM")

    t = find_t()
    phantoms = compute_phantoms(t)
    allocation = []
    detailed_info = []
    for i in range(m):
        votes_i = P[:, i].tolist()
        combined_list = votes_i + phantoms
        median_i = np.median(combined_list)
        allocation.append(median_i)
        detailed_info.append({
            'alternative': i + 1,
            'votes': [round(v, 3) for v in votes_i],
            'phantoms': phantoms,
            'combined_list': sorted([round(x, 3) for x in combined_list]),
            'median': median_i
        })
    allocation = np.array(allocation)
    return allocation, detailed_info

def compute_disutility(P, allocation, type):
    disutility = np.sum(np.abs(P - allocation), axis=1)  #array of nx1, disutility per agent
    if type == 'welfare':
        return np.sum(disutility)
    elif type == 'fairness':
        return np.max(disutility)
    elif type == 'proportional':
        return np.sum(np.abs(np.mean(P, axis=0) - allocation))


def combined_allocation_and_loss(n, m, P):
    allocation_IMM, _ = independent_market_mechanism(n, m, P)
    allocation_WMPM, _ = welfare_maximizing_phantom_mechanism(n, m, P)

    allocation_IMM = np.round(allocation_IMM, 3)
    allocation_WMPM = np.round(allocation_WMPM, 3)

    allocation_C = np.round((allocation_IMM + allocation_WMPM) / 2, 3)

    total_disutility_IMM = np.round(compute_disutility(P, allocation_IMM, type="welfare"), 3)
    total_disutility_WMPM = np.round(compute_disutility(P, allocation_WMPM, type="welfare"), 3)
    total_disutility_C = np.round(compute_disutility(P, allocation_C, type="welfare"), 3)

    welfare_loss_IMM = total_disutility_C - total_disutility_IMM
    welfare_loss_WMPM = total_disutility_C - total_disutility_WMPM

    metrics = 'proportional'

    max_disutility_IMM = np.round(compute_disutility(P, allocation_IMM, type=metrics), 3)
    max_disutility_WMPM = np.round(compute_disutility(P, allocation_WMPM, type=metrics), 3)
    max_disutility_C = np.round(compute_disutility(P, allocation_C, type=metrics), 3)

    fairness_loss_IMM = max_disutility_C - max_disutility_IMM
    fairness_loss_WMPM = max_disutility_C - max_disutility_WMPM

    return {
        "allocation_IMM": allocation_IMM,
        "allocation_WMPM": allocation_WMPM,
        "combined_allocation": allocation_C,
        "total_disutility_IMM": total_disutility_IMM,
        "total_disutility_WMPM": total_disutility_WMPM,
        "total_disutility_C": total_disutility_C,
        "max_disutility_IMM": max_disutility_IMM,
        "max_disutility_WMPM": max_disutility_WMPM,
        "max_disutility_C": max_disutility_C,
        "welfare_loss_IMM": welfare_loss_IMM,
        "welfare_loss_WMPM": welfare_loss_WMPM,
        "fairness_loss_IMM": fairness_loss_IMM,
        "fairness_loss_WMPM": fairness_loss_WMPM,
    }

def generate_random_preferences(n, m):
    P = np.random.rand(n, m)
    P = P / P.sum(axis=1, keepdims=True)
    return np.round(P, 2)

def generate_all_voter_profiles(n,m):
    total = 10
    preferences = []
    if m == 3:
        for p1 in range(total + 1):
            for p2 in range(total - p1 + 1):
                p3 = total - p1 - p2
                preferences.append((p1 / 10, p2 / 10, p3 / 10))
        return preferences
    elif m == 2:
        for p1 in range(total + 1):
            p2 = total - p1
            preferences.append((p1 / 10, p2 / 10))
        return preferences
    return product(preferences, repeat=n)

def generate_data(n, m):
    all_profiles = generate_all_voter_profiles(n, m)
    results = []

    for profile in all_profiles:
        P = np.array(profile)

        results_dict = combined_allocation_and_loss(n, m, P)

        results.append({
            'P': P,
            'allocation_IMM': results_dict["allocation_IMM"],
            'allocation_WMPM': results_dict["allocation_WMPM"],
            'allocation_C': results_dict["combined_allocation"],
            'total_disutility_IMM': results_dict["total_disutility_IMM"],
            'total_disutility_WMPM': results_dict["total_disutility_WMPM"],
            'total_disutility_C': results_dict["total_disutility_C"],
            'max_disutility_IMM': results_dict["max_disutility_IMM"],
            'max_disutility_WMPM': results_dict["max_disutility_WMPM"],
            'max_disutility_C': results_dict["max_disutility_C"],
            'welfare_loss_IMM': round(results_dict["welfare_loss_IMM"], 3),
            'welfare_loss_WMPM': round(results_dict["welfare_loss_WMPM"], 3),
            'fairness_loss_IMM': round(results_dict["fairness_loss_IMM"], 3),
            'fairness_loss_WMPM': round(results_dict["fairness_loss_WMPM"], 3),
        })

    df = pd.DataFrame(results)
    return df

def compute_disutility_for_alpha(n, m, P, alpha, disutility_type='welfare'):

    allocation_IMM, _ = independent_market_mechanism(n, m, P)
    allocation_WMPM, _ = welfare_maximizing_phantom_mechanism(n, m, P)


    allocation = (1 - alpha) * allocation_IMM + alpha * allocation_WMPM


    if disutility_type == 'welfare':

        total_disutility = compute_disutility(P, allocation, type='welfare')
    elif disutility_type == 'fairness':

        total_disutility = compute_disutility(P, allocation, type='fairness')
    elif disutility_type == 'proportional':

        total_disutility = compute_disutility(P, allocation, type='proportional')
    else:
        raise ValueError("Invalid disutility_type. Choose 'welfare', 'fairness', or 'proportional'.")

    result = {
        'alpha': alpha,
        'allocation': allocation,
        'disutility': total_disutility
    }

    return result

def iterate_over_alphas(n, m, P, alpha_values, disutility_type='welfare'):
    results = []

    for alpha in alpha_values:

        result = compute_disutility_for_alpha(n, m, P, alpha, disutility_type)
        results.append(result)

    df = pd.DataFrame(results)

    return df