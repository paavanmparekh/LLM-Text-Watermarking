import pandas as pd

# Lambda 4.0
print("=" * 70)
print("LAMBDA 4.0 - outputs/undetectable_results_lam4.0.csv")
print("=" * 70)
df1 = pd.read_csv('outputs/undetectable_results_lam4.0.csv')
print(f"Row Count: {len(df1)}")
print("\nColumns:", df1.columns.tolist())
print("\nDescriptive Statistics:")
print(df1.describe().to_string())

# Watermark detection rate for Lambda 4.0
if 'Watermark_Detected' in df1.columns:
    detection_rate_1 = (df1['Watermark_Detected'] == 'True').sum() / len(df1)
    print(f"\nWatermark Detection Rate: {detection_rate_1:.4f}")

print("\n\n" + "=" * 70)
print("LAMBDA 8.0 - outputs/undetectable_results_lam8.0.csv")
print("=" * 70)
df2 = pd.read_csv('outputs/undetectable_results_lam8.0.csv')
print(f"Row Count: {len(df2)}")
print("\nColumns:", df2.columns.tolist())
print("\nDescriptive Statistics:")
print(df2.describe().to_string())

# Watermark detection rate for Lambda 8.0
if 'Watermark_Detected' in df2.columns:
    detection_rate_2 = (df2['Watermark_Detected'] == 'True').sum() / len(df2)
    print(f"\nWatermark Detection Rate: {detection_rate_2:.4f}")
