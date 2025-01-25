import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt



def independent_market_mechanism(n, m, P):
    def compute_sum_medians(t):
        phantoms = np.clip(t * (n - np.arange(n + 1)), 0, 1).tolist()
        medians = [np.median(P[:, i].tolist() + phantoms) for i in range(m)]
        return sum(medians), phantoms

    def find_t():
        t_low = 0.0
        t_high = 1.0
        epsilon = 1e-8
        max_iterations = 1000

        for _ in range(max_iterations):
            t_mid = (t_low + t_high) / 2.0
            sum_mid, phantoms = compute_sum_medians(t_mid)
            if abs(sum_mid - 1.0) < epsilon:
                return t_mid, phantoms
            elif sum_mid < 1.0:
                t_low = t_mid
            else:
                t_high = t_mid
        raise ValueError("Failed to converge to a solution for t. IMM.")

    t, phantoms = find_t()

    allocation = np.array([np.median(P[:, i].tolist() + phantoms) for i in range(m)])

    detailed_info = [{
        'alternative': i + 1,
        'votes': [round(v, 3) for v in P[:, i]],
        'combined_list': sorted([round(x, 3) for x in (P[:, i].tolist() + phantoms)]),
        'median': allocation[i]
    } for i in range(m)]

    mechanism_info = {'phantoms': phantoms, 't': t}
    return allocation, mechanism_info, detailed_info

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
        medians = [np.median(P[:, i].tolist() + phantoms) for i in range(m)]
        return sum(medians)

    def find_t():
        t_low = 0.0
        t_high = 1.0
        epsilon = 1e-8
        max_iterations = 1000

        for _ in range(max_iterations):
            t_mid = (t_low + t_high) / 2.0
            sum_mid = compute_sum_medians(t_mid)
            if abs(sum_mid - 1.0) < epsilon:
                return t_mid
            elif sum_mid > 1.0:
                t_high = t_mid
            else:
                t_low = t_mid
        raise ValueError("Failed to converge to a solution for t. WMM")

    t = find_t()
    phantoms = compute_phantoms(t)

    allocation = np.array([np.median(P[:, i].tolist() + phantoms) for i in range(m)])

    detailed_info = [{
        'alternative': i + 1,
        'votes': [round(v, 3) for v in P[:, i]],
        'combined_list': sorted([round(x, 3) for x in (P[:, i].tolist() + phantoms)]),
        'median': allocation[i]
    } for i in range(m)]

    mechanism_info = {'phantoms': phantoms, 't': t}
    return allocation, mechanism_info, detailed_info


def compute_disutility(P, allocation, type):
    disutility = np.sum(np.abs(P - allocation), axis=1)  #array of nx1, disutility per agent
    if type == 'util':
        return np.sum(disutility)
    elif type == 'egal':
        return np.max(disutility)
    elif type == 'mean':
        return np.sum(np.abs(np.mean(P, axis=0) - allocation))
    elif type == 'gini':
        n = len(disutility)
        
        total_disutility = np.sum(disutility)
        if total_disutility == 0:
            return 0.0  # Edge case: No disutility, perfect equality
        
        # Compute double summation of absolute differences
        double_sum = np.sum(np.abs(disutility[i] - disutility[j]) for i in range(n) for j in range(n))
        
        # Apply Gini coefficient formula
        gini = double_sum / (2 * n * total_disutility)
        
        return gini


def generate_random_preferences(n, m, decimal = None):
    P = np.random.rand(n, m)
    P = P / P.sum(axis=1, keepdims=True)
    if decimal is not None:
        P = np.round(P,decimal)
    return P



def generate_all_voter_profiles(n, m):
    """
    Generate all possible n-voter profiles for m alternatives,
    discretized in increments of 0.1, such that each single-voter
    preference sums to 1.

    Returns an iterator (or generator) of n-tuples, where each element
    of the tuple is an m-dimensional preference.
    """
    total = 10  # We'll split into 0.1 increments

    # Build the set of single-voter preferences
    single_voter_prefs = []
    if m == 3:
        # All (p1, p2, p3) with p1+p2+p3 = 1, in steps of 0.1
        for p1 in range(total + 1):
            for p2 in range(total - p1 + 1):
                p3 = total - p1 - p2
                single_voter_prefs.append((p1 / 10, p2 / 10, p3 / 10))
    elif m == 2:
        # All (p1, p2) with p1+p2 = 1, in steps of 0.1
        for p1 in range(total + 1):
            p2 = total - p1
            single_voter_prefs.append((p1 / 10, p2 / 10))
    else:
        # For m > 3, or other values, you might define your own method here.
        # For now, we leave an empty list or raise an error, depending on your needs.
        raise ValueError("generate_all_voter_profiles only handles m=2 or m=3 in this example.")

    # Now build all n-voter profiles as an n-tuple of single-voter preferences:
    return itertools.product(single_voter_prefs, repeat=n)


def compute_disutility_for_alpha(P, alpha, disutility_type):
    n = P.shape[0]
    m = P.shape[1]

    allocation_IMM, _,_ = independent_market_mechanism(n, m, P)
    allocation_WMPM, _, _ = welfare_maximizing_phantom_mechanism(n, m, P)


    allocation = (1 - alpha) * allocation_IMM + alpha * allocation_WMPM

    total_disutility = compute_disutility(P, allocation, type=disutility_type)

    result = {
        'alpha': alpha,
        'allocation': allocation,
        'disutility': total_disutility
    }

    return result

def iterate_over_alphas(P, alpha_values, disutility_type):

    results = []

    for alpha in alpha_values:

        result = compute_disutility_for_alpha(P, alpha, disutility_type)
        results.append(result)

    df = pd.DataFrame(results)

    return df

def tradeoff_with_alpha(P,alpha_count):

    alpha_values = np.linspace(0, 1, alpha_count)
    
    df1 = iterate_over_alphas(P, alpha_values, disutility_type='util')
    df2 = iterate_over_alphas(P, alpha_values, disutility_type='mean')

    both = pd.merge(df1,df2,on='alpha')
    both.rename(columns={'disutility_x':'Welfare','disutility_y':'Fairness'},inplace=True)
    both.drop(columns=['allocation_y'],inplace=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(both['Fairness'],both['Welfare'],marker ='o')

    for i, row in both.iterrows():
        label = f"{row['alpha']:.2f}"  # Format alpha to two decimal points
        ax.text(row['Fairness'],row['Welfare'],s = label, fontsize=9, ha='right', va='bottom')
        
    ax.set_title('Welfare & Fairness Tradeoff')
    ax.set_ylabel('Welfare')
    ax.set_xlabel('Fairness (Distance to Mean allocation)')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return both


def mass_calculate_fairness(n, m, mechanism, metrics):
    """
    Enumerate ALL possible n-voter profiles for m=2 or m=3 (in 0.1 increments),
    compute the chosen mechanism's allocation (IMM or WMPM),
    and then compute the specified fairness/disutility metrics for that allocation.

    Parameters:
    -----------
    n : int
        Number of voters.
    m : int
        Number of alternatives (should be 2 or 3 in this example).
    mechanism : str
        Either 'IMM' or 'WMPM'. Defaults to 'IMM'.
    metrics : list of str, optional
        The list of fairness/disutility metrics to compute. Options include
        'gini', 'fairness', 'proportional', 'welfare'. Defaults to ['gini'].

    Returns:
    --------
    pd.DataFrame
        A DataFrame where each row corresponds to a unique n-voter profile.
        Columns include:
            - 'profile': An n-tuple of single-voter preferences (each m-tuple).
            - 'allocation': The resulting allocation vector from the mechanism.
            - For each metric in metrics, a column with the respective value.
    """


    # Get an iterator of ALL n-voter profiles
    all_profiles_iterator = generate_all_voter_profiles(n, m)
    results = []

    for profile_tuple in all_profiles_iterator:
  
        P = np.array(profile_tuple) 

        if mechanism == 'IMM':
            allocation, mech_info, detail = independent_market_mechanism(n, m, P)
        elif mechanism == 'WMPM':
            allocation, mech_info, detail = welfare_maximizing_phantom_mechanism(n, m, P)

        metric_values = {}
        for metric in metrics:
            metric_values[metric] = compute_disutility(P, allocation, type=metric)

        row_data = {
            'profile': P,  # Use the NumPy array directly
            'allocation': np.round(allocation,3)
        }
        row_data.update(metric_values)

        results.append(row_data)

    return pd.DataFrame(results)


