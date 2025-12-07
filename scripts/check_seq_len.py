import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import MitochondriaDataLoader

def check_sequence_lengths():
    print("Loading data...")
    loader = MitochondriaDataLoader()
    data = loader.load_data()
    
    counts = data.groupby('Participant').size()
    print("\nMitochondria counts per participant:")
    print(counts.describe())
    print("\nTop 5 longest sequences:")
    print(counts.nlargest(5))
    print("\nTop 5 shortest sequences:")
    print(counts.nsmallest(5))

if __name__ == "__main__":
    check_sequence_lengths()
