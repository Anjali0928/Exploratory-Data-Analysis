import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler

# ─── A. Sample Data ─────────────────────────────────────────────────────────
csv = """ListingID,Price,Area,Location,YearBuilt,Bedrooms
L001,250000,1500,Downtown,2005,3
L002,180000,1200,Suburb,1998,
L003,350000,2000,,2015,4
L004,NaN,1800,Suburb,2010,3
L005,300000,,Downtown,2000,2
"""

df = pd.read_csv(StringIO(csv))
print("Original data:")
print(df)

# ─── B. Check Missing Values ────────────────────────────────────────────────
print("\nMissing per column:")
print(df.isna().sum())

# ─── C. Impute Missing Values ───────────────────────────────────────────────
# 1. Price & Area: fill with median
df['Price'] = df['Price'].fillna(df['Price'].median())
df['Area']  = df['Area'].fillna(df['Area'].median())

# 2. Bedrooms: fill with median
df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())

# 3. Location: fill with mode
df['Location'] = df['Location'].fillna(df['Location'].mode()[0])

print("\nAfter imputing missing values:")
print(df)

# ─── D. Feature Engineering ─────────────────────────────────────────────────
# Create HouseAge = current_year (2025) – YearBuilt
df['HouseAge'] = 2025 - df['YearBuilt']
print("\nWith new feature 'HouseAge':")
print(df[['ListingID','YearBuilt','HouseAge']])

# ─── E. Encoding Categorical ────────────────────────────────────────────────
df_enc = pd.get_dummies(df, columns=['Location'], prefix='Loc', drop_first=True)
print("\nAfter one‑hot encoding 'Location':")
print(df_enc)

# ─── F. Scaling Numeric Features ────────────────────────────────────────────
scaler = StandardScaler()
for col in ['Price','Area','Bedrooms','HouseAge']:
    df_enc[col] = scaler.fit_transform(df_enc[[col]])

print("\nScaled numeric features:")
print(df_enc[['ListingID','Price','Area','Bedrooms','HouseAge']])
