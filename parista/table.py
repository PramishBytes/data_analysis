import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Read the CSV file
df = pd.read_csv('parista.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Define Likert scale mapping
likert_mapping = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

# Define clean variable labels for Likert items
variable_labels = {
    'Part I: Frequency of Use [I use YouTube Shorts or TikTok Reels every day.]': 'Use every day',
    'Part I: Frequency of Use [I open YouTube Shorts or TikTok Reels many times a day.]': 'Open many times a day',
    'Part I: Frequency of Use [I use YouTube Shorts or TikTok Reels whenever I have free time.]': 'Use whenever free',
    'Part I: Frequency of Use [Watching YouTube Shorts or TikTok Reels is part of my daily routine.]': 'Part of daily routine',
    'Part I: Frequency of Use [I use YouTube Shorts or TikTok Reels more frequently than other social media platforms.]': 'Use more than other platforms',
    'Part II: Duration of Use [I spend more than one hour per day watching YouTube Shorts or TikTok Reels.]': 'Spend >1 hour daily',
    'Part II: Duration of Use [I usually continue watching short videos longer than I originally planned.]': 'Continue longer than planned',
    'Part II: Duration of Use [I spend a large amount of my free time watching YouTube Shorts or TikTok Reels.]': 'Spend large amount of free time',
    'Part II: Duration of Use [I often lose track of time while watching short videos.]': 'Lose track of time',
    'Part II: Duration of Use [I spend more time watching YouTube Shorts or TikTok Reels than I intend.]': 'Spend more time than intended',
    'Part III: Viewing Habits\n [I watch YouTube Shorts or TikTok Reels immediately after waking up.]': 'Watch immediately after waking',
    'Part III: Viewing Habits\n [I watch short videos before going to sleep.]': 'Watch before sleep',
    'Part III: Viewing Habits\n [I check YouTube Shorts or TikTok Reels whenever I have free time.]': 'Check whenever free',
    'Part III: Viewing Habits\n [I watch short videos during study breaks.]': 'Watch during study breaks',
    'Part III: Viewing Habits\n [I continue watching one short video after another without stopping.]': 'Continue watching one after another',
    'Section C: Attention Span (Dependent Variable)\nInstructions: Please indicate your level of agreement with each of the following statements by selecting one option.\nAttention Span  [I can concentrate on my studies without getting distracted.]': 'Can concentrate on studies',
    'Section C: Attention Span (Dependent Variable)\nInstructions: Please indicate your level of agreement with each of the following statements by selecting one option.\nAttention Span  [I find it difficult to stay focused during lectures.]': 'Difficult to stay focused in lectures',
    'Section C: Attention Span (Dependent Variable)\nInstructions: Please indicate your level of agreement with each of the following statements by selecting one option.\nAttention Span  [I can complete academic tasks without losing concentration.]': 'Can complete tasks without losing concentration',
    'Section C: Attention Span (Dependent Variable)\nInstructions: Please indicate your level of agreement with each of the following statements by selecting one option.\nAttention Span  [I am easily distracted while studying.]': 'Easily distracted while studying',
    'Section C: Attention Span (Dependent Variable)\nInstructions: Please indicate your level of agreement with each of the following statements by selecting one option.\nAttention Span  [I can pay attention to one task for a long period.]': 'Can pay attention for long period'
}

# Define factor groups
factor_groups = {
    'Frequency of Use (Part I)': [
        'Part I: Frequency of Use [I use YouTube Shorts or TikTok Reels every day.]',
        'Part I: Frequency of Use [I open YouTube Shorts or TikTok Reels many times a day.]',
        'Part I: Frequency of Use [I use YouTube Shorts or TikTok Reels whenever I have free time.]',
        'Part I: Frequency of Use [Watching YouTube Shorts or TikTok Reels is part of my daily routine.]',
        'Part I: Frequency of Use [I use YouTube Shorts or TikTok Reels more frequently than other social media platforms.]'
    ],
    'Duration of Use (Part II)': [
        'Part II: Duration of Use [I spend more than one hour per day watching YouTube Shorts or TikTok Reels.]',
        'Part II: Duration of Use [I usually continue watching short videos longer than I originally planned.]',
        'Part II: Duration of Use [I spend a large amount of my free time watching YouTube Shorts or TikTok Reels.]',
        'Part II: Duration of Use [I often lose track of time while watching short videos.]',
        'Part II: Duration of Use [I spend more time watching YouTube Shorts or TikTok Reels than I intend.]'
    ],
    'Viewing Habits (Part III)': [
        'Part III: Viewing Habits\n [I watch YouTube Shorts or TikTok Reels immediately after waking up.]',
        'Part III: Viewing Habits\n [I watch short videos before going to sleep.]',
        'Part III: Viewing Habits\n [I check YouTube Shorts or TikTok Reels whenever I have free time.]',
        'Part III: Viewing Habits\n [I watch short videos during study breaks.]',
        'Part III: Viewing Habits\n [I continue watching one short video after another without stopping.]'
    ],
    'Attention Span (Dependent Variable)': [
        'Section C: Attention Span (Dependent Variable)\nInstructions: Please indicate your level of agreement with each of the following statements by selecting one option.\nAttention Span  [I can concentrate on my studies without getting distracted.]',
        'Section C: Attention Span (Dependent Variable)\nInstructions: Please indicate your level of agreement with each of the following statements by selecting one option.\nAttention Span  [I find it difficult to stay focused during lectures.]',
        'Section C: Attention Span (Dependent Variable)\nInstructions: Please indicate your level of agreement with each of the following statements by selecting one option.\nAttention Span  [I can complete academic tasks without losing concentration.]',
        'Section C: Attention Span (Dependent Variable)\nInstructions: Please indicate your level of agreement with each of the following statements by selecting one option.\nAttention Span  [I am easily distracted while studying.]',
        'Section C: Attention Span (Dependent Variable)\nInstructions: Please indicate your level of agreement with each of the following statements by selecting one option.\nAttention Span  [I can pay attention to one task for a long period.]'
    ]
}

# Get all Likert variables
all_likert_vars = []
for vars_list in factor_groups.values():
    all_likert_vars.extend(vars_list)

# Clean function
def clean_likert(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ['strongly agree', 'strongly agreee']:
            return 'Strongly Agree'
        if value.lower() in ['agree', 'agreee']:
            return 'Agree'
        if value.lower() in ['neutral', 'nertral', 'neutal']:
            return 'Neutral'
        if value.lower() in ['disagree', 'disagreee']:
            return 'Disagree'
        if value.lower() in ['strongly disagree', 'strongly disagreee']:
            return 'Strongly Disagree'
    return value

# Clean the data
df_clean = df.copy()
df_clean = df_clean.dropna(how='all')

for var in all_likert_vars:
    if var in df_clean.columns:
        df_clean[var] = df_clean[var].apply(clean_likert)

# Convert to numeric
likert_data = {}
for var in all_likert_vars:
    if var in df_clean.columns:
        likert_data[var] = df_clean[var].map(likert_mapping)

likert_df = pd.DataFrame(likert_data)

# ============================================================================
# 1. DEMOGRAPHIC FREQUENCY TABLES
# ============================================================================

print("\n" + "="*120)
print("DEMOGRAPHIC FREQUENCY DISTRIBUTIONS")
print("="*120)

demographic_vars = ['1. Gender', '2. Age', '3. Semester', '4. College']
for var in demographic_vars:
    if var in df_clean.columns:
        print(f"\n{var}:")
        freq = df_clean[var].value_counts(dropna=False)
        percent = df_clean[var].value_counts(normalize=True, dropna=False) * 100
        table = pd.DataFrame({'Frequency': freq, 'Percentage (%)': percent.round(2)})
        print(table)
        print("-"*80)

# ============================================================================
# 2. PLATFORM USAGE FREQUENCY TABLES
# ============================================================================

print("\n" + "="*120)
print("PLATFORM USAGE FREQUENCY DISTRIBUTIONS")
print("="*120)

platform_vars = ['1. Do you use YouTube Shorts?', 'Do you use TikTok Reels?', 
                 'Which platform do you use more frequently?', 
                 'Approximately how much time do you spend watching YouTube Shorts or TikTok Reels each day?']
for var in platform_vars:
    if var in df_clean.columns:
        print(f"\n{var}:")
        freq = df_clean[var].value_counts(dropna=False)
        percent = df_clean[var].value_counts(normalize=True, dropna=False) * 100
        table = pd.DataFrame({'Frequency': freq, 'Percentage (%)': percent.round(2)})
        print(table)
        print("-"*80)

# ============================================================================
# 3. COMPLETE DESCRIPTIVE STATISTICS - INDIVIDUAL ITEMS
# ============================================================================

print("\n" + "="*120)
print("COMPLETE LIKERT SCALE DESCRIPTIVE STATISTICS (N={})".format(len(df_clean)))
print("="*120)
print("\nInterpretation: 1=Strongly Disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Strongly Agree")
print("-"*120)

# Create descriptive statistics table
descriptive_rows = []
for var in all_likert_vars:
    if var in likert_df.columns:
        series = likert_df[var].dropna()
        if len(series) > 0:
            # Get clean variable name
            clean_name = variable_labels.get(var, var[:50])
            
            # Get factor group
            factor_name = next((g for g, v in factor_groups.items() if var in v), 'Unknown')
            
            # Get frequency distribution
            freq_dist = series.value_counts().sort_index()
            
            row = {
                'No': len(descriptive_rows) + 1,
                'Variable': clean_name,
                'Factor': factor_name,
                'N': len(series),
                'Mean': round(series.mean(), 3),
                'Median': round(series.median(), 3),
                'Mode': series.mode().iloc[0] if len(series.mode()) > 0 else np.nan,
                'Std Dev': round(series.std(), 3),
                'Min': series.min(),
                'Max': series.max(),
                'Skewness': round(series.skew(), 3),
                'Kurtosis': round(series.kurtosis(), 3),
                'SA %': round((freq_dist.get(5, 0) / len(series)) * 100, 1),
                'A %': round((freq_dist.get(4, 0) / len(series)) * 100, 1),
                'N %': round((freq_dist.get(3, 0) / len(series)) * 100, 1),
                'D %': round((freq_dist.get(2, 0) / len(series)) * 100, 1),
                'SD %': round((freq_dist.get(1, 0) / len(series)) * 100, 1)
            }
            descriptive_rows.append(row)

descriptive_df = pd.DataFrame(descriptive_rows)

# Print the table
print(descriptive_df.to_string(index=False))
print("-"*120)

# ============================================================================
# 4. SAVE TO CSV FOR BETTER VIEWING
# ============================================================================

# Save descriptive stats to CSV
descriptive_df.to_csv('descriptive_stats.csv', index=False)
print("\n✓ Descriptive statistics saved to 'descriptive_stats.csv'")

# ============================================================================
# 5. FACTOR-LEVEL SUMMARY
# ============================================================================

print("\n" + "="*120)
print("FACTOR-LEVEL SUMMARY (Composite Scores)")
print("="*120)

def cronbach_alpha(data):
    """Calculate Cronbach's alpha"""
    if data.shape[1] < 2:
        return np.nan
    if len(data) < 2:
        return np.nan
    try:
        items = data.shape[1]
        variances = data.var(axis=0, ddof=1)
        total_variance = data.sum(axis=1).var(ddof=1)
        alpha = (items / (items - 1)) * (1 - (variances.sum() / total_variance))
        return alpha
    except:
        return np.nan

summary_rows = []
for factor_name, var_list in factor_groups.items():
    valid_vars = [v for v in var_list if v in likert_df.columns]
    if valid_vars:
        factor_data = likert_df[valid_vars].dropna()
        if len(factor_data) > 0:
            # Calculate composite score
            composite = factor_data.mean(axis=1)
            
            # Calculate Cronbach's Alpha
            alpha = cronbach_alpha(factor_data)
            
            row = {
                'Factor': factor_name,
                'Items': len(valid_vars),
                'N': len(factor_data),
                'Mean': round(composite.mean(), 3),
                'Median': round(composite.median(), 3),
                'Std Dev': round(composite.std(), 3),
                'Min': round(composite.min(), 3),
                'Max': round(composite.max(), 3),
                'Skewness': round(composite.skew(), 3),
                'Kurtosis': round(composite.kurtosis(), 3),
                'Cronbach α': round(alpha, 3) if not np.isnan(alpha) else 'N/A'
            }
            summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# Save factor summary to CSV
summary_df.to_csv('factor_summary.csv', index=False)
print("\n✓ Factor summary saved to 'factor_summary.csv'")

# ============================================================================
# 6. ATTENTION SPAN - REVERSE CODED COMPOSITE
# ============================================================================

print("\n" + "="*120)
print("ATTENTION SPAN ANALYSIS (Dependent Variable)")
print("="*120)

# Get attention span variables
attention_vars = factor_groups['Attention Span (Dependent Variable)']
attention_vars_valid = [v for v in attention_vars if v in likert_df.columns]

if attention_vars_valid:
    # Reverse code negative items (items 2 and 4)
    negative_items_indices = [1, 3]  # Second and fourth items are negative
    reverse_coded = likert_df[attention_vars_valid].copy()
    
    for idx in negative_items_indices:
        if idx < len(attention_vars_valid):
            var = attention_vars_valid[idx]
            reverse_coded[var] = 6 - reverse_coded[var]  # 1->5, 2->4, 3->3, 4->2, 5->1
    
    # Calculate composite attention score
    attention_composite = reverse_coded.mean(axis=1).dropna()
    
    print("\nAttention Span Composite Score (Higher = Better Attention):")
    print(f"  N = {len(attention_composite)}")
    print(f"  Mean = {attention_composite.mean():.3f}")
    print(f"  Median = {attention_composite.median():.3f}")
    print(f"  Std Dev = {attention_composite.std():.3f}")
    print(f"  Min = {attention_composite.min():.3f}")
    print(f"  Max = {attention_composite.max():.3f}")
    print(f"  Skewness = {attention_composite.skew():.3f}")
    print(f"  Kurtosis = {attention_composite.kurtosis():.3f}")
    
    # Interpretation
    mean_attention = attention_composite.mean()
    if mean_attention >= 3.5:
        interpretation = "Good attention span"
    elif mean_attention >= 2.5:
        interpretation = "Moderate attention span"
    else:
        interpretation = "Poor attention span"
    print(f"  Interpretation: {interpretation}")

# ============================================================================
# 7. ATTENTION SPAN BY USAGE TIME
# ============================================================================

print("\n" + "="*120)
print("ATTENTION SPAN BY DAILY USAGE TIME")
print("="*120)

if attention_vars_valid and 'Approximately how much time do you spend watching YouTube Shorts or TikTok Reels each day?' in df_clean.columns:
    # Add attention composite to main dataframe
    df_clean['Attention_Composite'] = np.nan
    df_clean.loc[attention_composite.index, 'Attention_Composite'] = attention_composite
    
    # Group by usage time
    time_group = df_clean.groupby('Approximately how much time do you spend watching YouTube Shorts or TikTok Reels each day?')['Attention_Composite'].agg(['count', 'mean', 'median', 'std'])
    print("\nBy Daily Usage Time:")
    print(time_group.round(3))

# ============================================================================
# 8. CORRELATION MATRIX
# ============================================================================

print("\n" + "="*120)
print("CORRELATION MATRIX BETWEEN FACTORS")
print("="*120)

# Calculate factor scores
factor_scores = {}
for factor_name, var_list in factor_groups.items():
    valid_vars = [v for v in var_list if v in likert_df.columns]
    if valid_vars:
        factor_scores[factor_name] = likert_df[valid_vars].mean(axis=1)

if len(factor_scores) > 1:
    factor_df = pd.DataFrame(factor_scores)
    corr_matrix = factor_df.corr()
    print("\nPearson Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # Save correlation matrix
    corr_matrix.to_csv('correlation_matrix.csv')
    print("\n✓ Correlation matrix saved to 'correlation_matrix.csv'")

# ============================================================================
# 9. KEY FINDINGS
# ============================================================================

print("\n" + "="*120)
print("KEY FINDINGS SUMMARY")
print("="*120)

# Sample size
print(f"\nTotal Sample Size: {len(df_clean)} respondents")

# Gender distribution
if '1. Gender' in df_clean.columns:
    gender_counts = df_clean['1. Gender'].value_counts()
    print("\nGender Distribution:")
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count} ({count/len(df_clean)*100:.1f}%)")

# Platform preference
if 'Which platform do you use more frequently?' in df_clean.columns:
    platform_counts = df_clean['Which platform do you use more frequently?'].value_counts()
    print("\nPlatform Preference:")
    for platform, count in platform_counts.items():
        print(f"  {platform}: {count} ({count/len(df_clean)*100:.1f}%)")

# Usage time
if 'Approximately how much time do you spend watching YouTube Shorts or TikTok Reels each day?' in df_clean.columns:
    time_counts = df_clean['Approximately how much time do you spend watching YouTube Shorts or TikTok Reels each day?'].value_counts()
    print("\nDaily Usage Time:")
    for time, count in time_counts.items():
        print(f"  {time}: {count} ({count/len(df_clean)*100:.1f}%)")

# Highest rated items
print("\nTop 5 Highest Rated Items (Strongest Agreement):")
top_items = descriptive_df.nlargest(5, 'Mean')
for _, row in top_items.iterrows():
    print(f"  {row['Variable']}: Mean = {row['Mean']:.3f} (SA={row['SA %']:.1f}%)")

# Lowest rated items
print("\nTop 5 Lowest Rated Items (Strongest Disagreement):")
bottom_items = descriptive_df.nsmallest(5, 'Mean')
for _, row in bottom_items.iterrows():
    print(f"  {row['Variable']}: Mean = {row['Mean']:.3f} (SD={row['SD %']:.1f}%)")

# Factor reliability
print("\nFactor Reliability (Cronbach's Alpha):")
for _, row in summary_df.iterrows():
    alpha = row['Cronbach α']
    if alpha != 'N/A':
        status = "Good" if alpha >= 0.7 else "Acceptable" if alpha >= 0.6 else "Questionable"
        print(f"  {row['Factor']}: α = {alpha} ({status})")

# Attention span
if attention_vars_valid:
    mean_att = attention_composite.mean()
    print(f"\nOverall Attention Span: Mean = {mean_att:.3f}")
    if mean_att >= 3.5:
        print("  Interpretation: Students demonstrate GOOD attention span")
    elif mean_att >= 2.5:
        print("  Interpretation: Students demonstrate MODERATE attention span")
    else:
        print("  Interpretation: Students demonstrate POOR attention span")

print("\n" + "="*120)
print("ANALYSIS COMPLETE! All tables saved as CSV files.")
print("="*120)