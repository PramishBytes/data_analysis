import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Read the CSV file
df = pd.read_csv('prasansa.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Define the Likert scale mapping
likert_mapping = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

# Define variable labels (clear, readable descriptions)
variable_labels = {
    'Independent Variables\n1. Parental Education [My parents encourage me to continue my education]': 'Parental Encouragement for Education',
    'Independent Variables\n1. Parental Education [My parents discuss career opportunities with me.]': 'Parents Discuss Career Opportunities',
    'Independent Variables\n1. Parental Education [My parents educational background motivates me to achieve higher goals.]': 'Parental Education as Motivation',
    'Independent Variables\n1. Parental Education [ I seek career advice from my parents.]': 'Seek Career Advice from Parents',
    '2. Family Income Level [Family financial condition affect my educational opportunities.]': 'Family Finances Affect Education',
    '2. Family Income Level [I consider the cost of further education when planning my career.]': 'Consider Cost in Career Planning',
    '2. Family Income Level [I prefer a career that offers long-term financial security.]': 'Prefer Long-term Financial Security',
    '2. Family Income Level [Financial responsibilities influene my career plans.]': 'Financial Responsibilities Affect Career',
    '2. Family Income Level [I may change my career choices due to financial limitations.]': 'May Change Career Due to Finances',
    '3. Peer Influence [I discuss career plans with my friends.]': 'Discuss Career with Friends',
    '3. Peer Influence [My friends introduce me to different careeer success.]': 'Friends Introduce Career Success',
    '3. Peer Influence [I feel motivated when my friends achieve career success]': 'Motivated by Friends\' Success',
    '3. Peer Influence [I value suggestion from my friends when making career decisions.]': 'Value Friends\' Suggestions',
    '4. Job Market Awareness [I regularly follow information about employment opportunities]': 'Follow Employment Information',
    '4. Job Market Awareness [I am aware of the skill employers expect from graduates ]': 'Aware of Employer Expectations',
    '4. Job Market Awareness [I keep myself updated on current labor market trends.]': 'Updated on Labor Market Trends',
    '4. Job Market Awareness [I believe understanding the job market helps me make better career decision]': 'Job Market Understanding Helps',
    '5. Personality Type [I prefer a career that matches my personal strengths]': 'Career Matches Personal Strengths',
    '5. Personality Type [I enjoy work that reflect my interest.]': 'Enjoy Interest-Matching Work',
    '5. Personality Type [I believe my personality fits my preferred profession]': 'Personality Fits Preferred Profession',
    '5. Personality Type [My values are important when deciding on a career]': 'Values Important in Career Choice',
    'Dependent Variable: Career preferencee pattern [I have a clear defined career preference.]': 'Clear Defined Career Preference',
    'Dependent Variable: Career preferencee pattern [I am confident in pursuing my preferred careeer path.]': 'Confident in Career Path',
    'Dependent Variable: Career preferencee pattern [I intend to pursue the career that best matches my interests, skill, and values.]': 'Pursue Career Matching Interests',
    'Dependent Variable: Career preferencee pattern [I am committed to achieving my career despite potential challenges.]': 'Committed to Career Goals',
    'Dependent Variable: Career preferencee pattern [My preferred career aligns with my long term professional goals.]': 'Career Aligns with Long-term Goals'
}

# Clean the data
def clean_likert(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ['neutral', 'neeutral', 'nneutral', 'nertral']:
            return 'Neutral'
        if value.lower() in ['strongly agree', 'strongly agreee']:
            return 'Strongly Agree'
        if value.lower() in ['strongly disagree', 'strongly disagreee']:
            return 'Strongly Disagree'
        if value.lower() in ['agree', 'agreee']:
            return 'Agree'
        if value.lower() in ['disagree', 'disagreee']:
            return 'Disagree'
    return value

# Prepare Likert variables
likert_vars = list(variable_labels.keys())
df_clean = df.copy()
df_clean = df_clean.dropna(how='all')

for var in likert_vars:
    if var in df_clean.columns:
        df_clean[var] = df_clean[var].apply(clean_likert)

# Convert to numeric
likert_data = {}
for var in likert_vars:
    if var in df_clean.columns:
        numeric_vals = df_clean[var].map(likert_mapping)
        likert_data[var] = numeric_vals

likert_df = pd.DataFrame(likert_data)

# ============================================================================
# MAIN DESCRIPTIVE STATISTICS TABLE
# ============================================================================

print("\n" + "="*120)
print("DESCRIPTIVE STATISTICS FOR LIKERT SCALE ITEMS (N=162)")
print("="*120)
print("\nInterpretation Guide:")
print("  • Mean scores range from 1 (Strongly Disagree) to 5 (Strongly Agree)")
print("  • Mean ≥ 4.0 indicates Strong Agreement tendency")
print("  • Mean between 3.5-3.9 indicates Agreement tendency")
print("  • Mean between 2.5-3.4 indicates Neutral tendency")
print("  • Mean ≤ 2.4 indicates Disagreement tendency")
print("  • Skewness: Negative = left-skewed (responses concentrated toward higher values)")
print("  • Skewness: Positive = right-skewed (responses concentrated toward lower values)")
print("-"*120)

# Create clean descriptive statistics table
descriptive_rows = []
for var in likert_vars:
    if var in likert_df.columns:
        series = likert_df[var].dropna()
        if len(series) > 0:
            # Get frequency distribution
            freq_dist = series.value_counts().sort_index()
            
            row = {
                'No': len(descriptive_rows) + 1,
                'Variable': variable_labels.get(var, var[:60] + '...'),
                'N': len(series),
                'Mean': series.mean(),
                'Median': series.median(),
                'Mode': series.mode().iloc[0] if len(series.mode()) > 0 else np.nan,
                'Std. Dev': series.std(),
                'Min': series.min(),
                'Max': series.max(),
                'Skewness': series.skew(),
                'Kurtosis': series.kurtosis(),
                '% Strongly Agree': (freq_dist.get(5, 0) / len(series)) * 100,
                '% Agree': (freq_dist.get(4, 0) / len(series)) * 100,
                '% Neutral': (freq_dist.get(3, 0) / len(series)) * 100,
                '% Disagree': (freq_dist.get(2, 0) / len(series)) * 100,
                '% Strongly Disagree': (freq_dist.get(1, 0) / len(series)) * 100,
            }
            descriptive_rows.append(row)

descriptive_df = pd.DataFrame(descriptive_rows)

# Format the table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.3f}'.format)

# Display the main table
print(descriptive_df.to_string(index=False))

# ============================================================================
# SUMMARY TABLE BY FACTOR
# ============================================================================

print("\n" + "="*120)
print("SUMMARY STATISTICS BY FACTOR (Composite Scores)")
print("="*120)

# Define factor groupings
factor_groups = {
    'Parental Education': [
        'Independent Variables\n1. Parental Education [My parents encourage me to continue my education]',
        'Independent Variables\n1. Parental Education [My parents discuss career opportunities with me.]',
        'Independent Variables\n1. Parental Education [My parents educational background motivates me to achieve higher goals.]',
        'Independent Variables\n1. Parental Education [ I seek career advice from my parents.]'
    ],
    'Family Income Level': [
        '2. Family Income Level [Family financial condition affect my educational opportunities.]',
        '2. Family Income Level [I consider the cost of further education when planning my career.]',
        '2. Family Income Level [I prefer a career that offers long-term financial security.]',
        '2. Family Income Level [Financial responsibilities influene my career plans.]',
        '2. Family Income Level [I may change my career choices due to financial limitations.]'
    ],
    'Peer Influence': [
        '3. Peer Influence [I discuss career plans with my friends.]',
        '3. Peer Influence [My friends introduce me to different careeer success.]',
        '3. Peer Influence [I feel motivated when my friends achieve career success]',
        '3. Peer Influence [I value suggestion from my friends when making career decisions.]'
    ],
    'Job Market Awareness': [
        '4. Job Market Awareness [I regularly follow information about employment opportunities]',
        '4. Job Market Awareness [I am aware of the skill employers expect from graduates ]',
        '4. Job Market Awareness [I keep myself updated on current labor market trends.]',
        '4. Job Market Awareness [I believe understanding the job market helps me make better career decision]'
    ],
    'Personality Type': [
        '5. Personality Type [I prefer a career that matches my personal strengths]',
        '5. Personality Type [I enjoy work that reflect my interest.]',
        '5. Personality Type [I believe my personality fits my preferred profession]',
        '5. Personality Type [My values are important when deciding on a career]'
    ],
    'Career Preference': [
        'Dependent Variable: Career preferencee pattern [I have a clear defined career preference.]',
        'Dependent Variable: Career preferencee pattern [I am confident in pursuing my preferred careeer path.]',
        'Dependent Variable: Career preferencee pattern [I intend to pursue the career that best matches my interests, skill, and values.]',
        'Dependent Variable: Career preferencee pattern [I am committed to achieving my career despite potential challenges.]',
        'Dependent Variable: Career preferencee pattern [My preferred career aligns with my long term professional goals.]'
    ]
}

summary_rows = []
for factor_name, var_list in factor_groups.items():
    valid_vars = [v for v in var_list if v in likert_df.columns]
    if valid_vars:
        factor_data = likert_df[valid_vars].dropna()
        if len(factor_data) > 0:
            # Calculate mean score for each respondent
            composite = factor_data.mean(axis=1)
            # Calculate Cronbach's Alpha
            items = factor_data.shape[1]
            if items >= 2:
                variances = factor_data.var(axis=0, ddof=1)
                total_variance = factor_data.sum(axis=1).var(ddof=1)
                alpha = (items / (items - 1)) * (1 - (variances.sum() / total_variance))
            else:
                alpha = np.nan
            
            row = {
                'Factor': factor_name,
                'Number of Items': items,
                'N': len(factor_data),
                'Mean': composite.mean(),
                'Median': composite.median(),
                'Std. Dev': composite.std(),
                'Min': composite.min(),
                'Max': composite.max(),
                'Cronbach\'s α': alpha,
                'Interpretation': 'Strong Agreement' if composite.mean() >= 4.0 else 
                                'Agreement' if composite.mean() >= 3.5 else
                                'Neutral' if composite.mean() >= 2.5 else
                                'Disagreement' if composite.mean() >= 1.5 else
                                'Strong Disagreement'
            }
            summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# ============================================================================
# CORRELATION MATRIX
# ============================================================================

print("\n" + "="*120)
print("CORRELATION MATRIX BETWEEN FACTORS (Pearson's r)")
print("="*120)

# Calculate factor scores
factor_scores = {}
for factor_name, var_list in factor_groups.items():
    valid_vars = [v for v in var_list if v in likert_df.columns]
    if valid_vars:
        factor_scores[factor_name] = likert_df[valid_vars].mean(axis=1)

factor_df = pd.DataFrame(factor_scores)
corr_matrix = factor_df.corr()

# Display correlation matrix with styling
print("\nCorrelation Matrix:")
print(corr_matrix.round(3))
print("\nInterpretation: Values close to 1 = Strong positive correlation")
print("              Values close to -1 = Strong negative correlation")
print("              Values close to 0 = No correlation")

# ============================================================================
# KEY FINDINGS STATEMENT
# ============================================================================

print("\n" + "="*120)
print("KEY FINDINGS AND INTERPRETATION STATEMENT")
print("="*120)

print("\n1. OVERALL RESPONSE PATTERN:")
print("-" * 80)
print(f"Sample Size: {len(df_clean)} respondents")
print(f"Average response across all items: {descriptive_df['Mean'].mean():.3f} (Agreement tendency)")

# Find highest and lowest rated items
highest_row = descriptive_df.loc[descriptive_df['Mean'].idxmax()]
lowest_row = descriptive_df.loc[descriptive_df['Mean'].idxmin()]

print(f"\n2. HIGHEST RATED ITEMS:")
print("-" * 80)
print(f"Top Item: '{highest_row['Variable']}'")
print(f"  Mean = {highest_row['Mean']:.3f}, {highest_row['% Strongly Agree']:.1f}% Strongly Agree")
print(f"\n3. LOWEST RATED ITEMS:")
print("-" * 80)
print(f"Bottom Item: '{lowest_row['Variable']}'")
print(f"  Mean = {lowest_row['Mean']:.3f}, {lowest_row['% Strongly Disagree']:.1f}% Strongly Disagree")

print("\n4. FACTOR-LEVEL INTERPRETATION:")
print("-" * 80)
for _, row in summary_df.iterrows():
    print(f"\n{row['Factor']}:")
    print(f"  • Mean Score: {row['Mean']:.3f} ({row['Interpretation']})")
    print(f"  • Reliability (Cronbach's α): {row['Cronbach\'s α']:.3f}")
    if row['Cronbach\'s α'] >= 0.7:
        print(f"  • Good internal consistency (α ≥ 0.70)")
    elif row['Cronbach\'s α'] >= 0.6:
        print(f"  • Acceptable internal consistency (α ≥ 0.60)")
    else:
        print(f"  • Questionable internal consistency (α < 0.60)")

print("\n5. CORRELATION INSIGHTS:")
print("-" * 80)
strongest_corr = corr_matrix.unstack().sort_values(ascending=False)
strongest_corr = strongest_corr[strongest_corr < 1].drop_duplicates()
if len(strongest_corr) > 0:
    top_corr = strongest_corr.head(3)
    for (var1, var2), corr_val in top_corr.items():
        print(f"  • {var1} ↔ {var2}: r = {corr_val:.3f}")

# ============================================================================
# EXPORT TO EXCEL
# ============================================================================

print("\n" + "="*120)
print("EXPORTING RESULTS")
print("="*120)

# Create Excel writer
with pd.ExcelWriter('career_preference_analysis.xlsx', engine='openpyxl') as writer:
    descriptive_df.to_excel(writer, sheet_name='Likert_Descriptive', index=False)
    summary_df.to_excel(writer, sheet_name='Factor_Summary', index=False)
    corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
    
    # Also export frequency tables
    freq_data = []
    for var in likert_vars:
        if var in likert_df.columns:
            series = likert_df[var].dropna()
            freq = series.value_counts().sort_index()
            for val, count in freq.items():
                label = {1: 'Strongly Disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly Agree'}.get(val, 'Unknown')
                freq_data.append({
                    'Variable': variable_labels.get(var, var),
                    'Response': label,
                    'Frequency': count,
                    'Percentage': (count / len(series)) * 100
                })
    
    freq_df = pd.DataFrame(freq_data)
    pivot_freq = freq_df.pivot(index='Variable', columns='Response', values=['Frequency', 'Percentage'])
    pivot_freq.to_excel(writer, sheet_name='Frequency_Distributions')

print("\n✓ Results exported to 'career_preference_analysis.xlsx'")
print("\n" + "="*120)
print("ANALYSIS COMPLETE!")
print("="*120)