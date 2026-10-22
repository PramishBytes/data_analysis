#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("✅ Loading data...")

# Define the mapping for Likert scale responses
likert_map = {
    'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly Agree': 5,
    'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly Agree': 5
}

# Column names for each construct
taste_cols = [
    'The food served at this restaurant/cafeteria has a good taste.',
    'The food is flavourful and well seasoned.',
    'The taste of the food is consistent every time I visit.',
    'The food meets my expectations in terms of taste.',
    'I enjoy the taste of the food served at this restaurant/cafeteria.'
]

freshness_cols = [
    'The food served at this restaurant/cafeteria is fresh.',
    'The ingredients used in the food appear to be fresh.',
    'The food is served at an appropriate temperature.',
    'The food appears to be freshly prepared.',
    'I am satisfied with the freshness of the food served.'
]

hygiene_cols = [
    'The food is prepared in a hygienic manner.',
    'The dining area is clean and well-maintained.',
    'The plates, glasses and utensils are clean.',
    'The staff maintain good personal hygiene while serving food.',
    'I feel confident about the cleanliness and safety of the food served.'
]

presentation_cols = [
    'The food is served in an attractive manner.',
    'The appearance of the food is appealing.',
    'The food is presented neatly.',
    'The presentation of the food increases my appetite.',
    'I am satisfied with the overall presentation of the food.'
]

portion_cols = [
    'The portion size of the food is adequate.',
    'The quantity of food is worth the price paid.',
    'The portion size is consistent on every visit.',
    'The amount of food served is sufficient to satisfy my hunger.',
    'I am satisfied with the portion size provided by this restaurant/cafeteria.'
]

satisfaction_cols = [
    'Overall, I am satisfied with the food quality at this restaurant/cafeteria.',
    'This restaurant/cafeteria meets my expectations.',
    'I am satisfied with my overall dining experience.',
    'I intend to visit this restaurant/cafeteria again in the future.',
    'I would recommend this restaurant/cafeteria to my friends and family.'
]

# Load the data
try:
    df = pd.read_csv('rashmi.csv', encoding='utf-8')
    print(f"✅ Loaded {len(df)} rows")
except FileNotFoundError:
    print("❌ Error: rashmi.csv not found")
    exit()

# Clean column names
df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('  ', ' ')

# Map responses to numeric values
def map_likert(value):
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return value
    value = str(value).strip()
    if value in likert_map:
        return likert_map[value]
    if 'Strongly disagree' in value or 'Strongly Disagree' in value:
        return 1
    elif 'Disagree' in value:
        return 2
    elif 'Neutral' in value:
        return 3
    elif 'Agree' in value:
        return 4
    elif 'Strongly Agree' in value or 'Strongly agree' in value:
        return 5
    return np.nan

# Apply mapping to all columns
all_construct_cols = taste_cols + freshness_cols + hygiene_cols + presentation_cols + portion_cols + satisfaction_cols

for col in all_construct_cols:
    if col in df.columns:
        df[col + '_numeric'] = df[col].apply(map_likert)

# Create composite scores
composite_map = {
    'Taste_Composite': taste_cols,
    'Freshness_Composite': freshness_cols,
    'Hygiene_Composite': hygiene_cols,
    'Presentation_Composite': presentation_cols,
    'Portion_Composite': portion_cols,
    'Satisfaction_Composite': satisfaction_cols
}

for comp_name, cols in composite_map.items():
    existing_cols = [col + '_numeric' for col in cols if col + '_numeric' in df.columns]
    if existing_cols:
        df[comp_name] = df[existing_cols].mean(axis=1)

# ============================================
# BUILD THE SINGLE TABLE
# ============================================

results = []

# 1. Add Composite Scores
for comp_name in composite_map.keys():
    data = df[comp_name].dropna()
    if len(data) > 0:
        mode_val = data.mode().iloc[0] if not data.mode().empty else np.nan
        
        results.append({
            'Construct': comp_name.replace('_Composite', ''),
            'Type': 'Composite',
            'Statement': comp_name,
            'N': len(data),
            'Mean': data.mean(),
            'Median': data.median(),
            'Mode': mode_val,
            'Min': data.min(),
            'Max': data.max(),
            'Std_Dev': data.std(),
            'Skewness': data.skew()
        })

# 2. Add Individual Statements
all_statements = [
    (taste_cols, 'Taste'),
    (freshness_cols, 'Freshness'),
    (hygiene_cols, 'Hygiene'),
    (presentation_cols, 'Presentation'),
    (portion_cols, 'Portion'),
    (satisfaction_cols, 'Satisfaction')
]

for cols, construct in all_statements:
    for col in cols:
        numeric_col = col + '_numeric'
        if numeric_col in df.columns:
            data = df[numeric_col].dropna()
            if len(data) > 0:
                mode_val = data.mode().iloc[0] if not data.mode().empty else np.nan
                
                results.append({
                    'Construct': construct,
                    'Type': 'Statement',
                    'Statement': col,
                    'N': len(data),
                    'Mean': data.mean(),
                    'Median': data.median(),
                    'Mode': mode_val,
                    'Min': data.min(),
                    'Max': data.max(),
                    'Std_Dev': data.std(),
                    'Skewness': data.skew()
                })

# Create DataFrame
final_df = pd.DataFrame(results)

# Round numeric columns
numeric_cols = ['Mean', 'Median', 'Mode', 'Min', 'Max', 'Std_Dev', 'Skewness']
for col in numeric_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].round(3)

# Reorder columns
column_order = ['Construct', 'Type', 'Statement', 'N', 'Mean', 'Median', 'Mode', 'Min', 'Max', 'Std_Dev', 'Skewness']
final_df = final_df[column_order]

# ============================================
# SAVE ONLY ONE CSV FILE
# ============================================

final_df.to_csv('Statistics_Single_Table.csv', index=False)

print("\n" + "=" * 130)
print("✅ SUCCESS! Only ONE file created:")
print("📁 Statistics_Single_Table.csv")
print("=" * 130)

# Display preview
print("\n" + "=" * 130)
print("PREVIEW OF THE TABLE:")
print("=" * 130)
print(final_df.to_string(index=False))

print("\n" + "=" * 130)
print(f"Total rows: {len(final_df)}")
print(f"  - Composite scores: {len(final_df[final_df['Type'] == 'Composite'])}")
print(f"  - Individual statements: {len(final_df[final_df['Type'] == 'Statement'])}")
print("=" * 130)