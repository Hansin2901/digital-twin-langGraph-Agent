#!/usr/bin/env python3
"""
Generate UC Berkeley processed CSV data
"""

from uc_berkeley_data_loader import load_real_uc_berkeley_data

if __name__ == "__main__":
    print("🚀 Generating UC Berkeley processed CSV...")
    
    # Generate the CSV with at least 1 week of data
    df = load_real_uc_berkeley_data(
        output_csv_path='uc_berkeley_processed_data.csv'
    )
    
    print(f"\n✅ COMPLETED!")
    print(f"   Records: {len(df):,}")
    print(f"   Duration: {df['time'].max() - df['time'].min()}")
    print(f"   Schema: {df.columns.tolist()}")
    print(f"   File: uc_berkeley_processed_data.csv")
