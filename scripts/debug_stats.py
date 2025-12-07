import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader

def debug_male_stats():
    print("Loading data...")
    loader = MitochondriaDataLoader()
    data = loader.load_data()
    
    feature_cols = loader.get_feature_columns()
    
    # Aggregate by participant (mean)
    participant_agg = data.groupby(['Participant', 'Group', 'Sex', 'Age']).agg({
        **{col: 'mean' for col in feature_cols}
    }).reset_index()
    
    # Filter for Males
    male_data = participant_agg[participant_agg['Sex'] == 'Male']
    
    metric = feature_cols[0] # PROM IsoVol
    print(f"\nAnalyzing Metric: {metric}")
    
    ct_data = male_data[male_data['Group'] == 'CT'][metric].dropna().values
    ela_data = male_data[male_data['Group'] == 'ELA'][metric].dropna().values
    
    print(f"\nCT Data (n={len(ct_data)}): {ct_data}")
    print(f"ELA Data (n={len(ela_data)}): {ela_data}")
    
    print(f"\nCT Mean: {np.mean(ct_data)}")
    print(f"ELA Mean: {np.mean(ela_data)}")
    
    # Manual Mann-Whitney Check
    print("\n--- Mann-Whitney U Test ---")
    stat, p = stats.mannwhitneyu(ct_data, ela_data, alternative='two-sided')
    print(f"Statistic: {stat}")
    print(f"P-value: {p}")
    
    # Check ranks
    combined = np.concatenate([ct_data, ela_data])
    ranks = stats.rankdata(combined)
    print(f"\nAll Values: {combined}")
    print(f"Ranks: {ranks}")
    
    rank_sum_ct = np.sum(ranks[:len(ct_data)])
    rank_sum_ela = np.sum(ranks[len(ct_data):])
    
    print(f"Rank Sum CT: {rank_sum_ct}")
    print(f"Rank Sum ELA: {rank_sum_ela}")
    
    n1 = len(ct_data)
    n2 = len(ela_data)
    
    u1 = rank_sum_ct - n1*(n1+1)/2
    u2 = rank_sum_ela - n2*(n2+1)/2
    
    print(f"U1 (calculated): {u1}")
    print(f"U2 (calculated): {u2}")
    
    expected_u = n1 * n2 / 2
    print(f"Expected U under null: {expected_u}")

if __name__ == "__main__":
    debug_male_stats()
