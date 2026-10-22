import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, f_oneway, ttest_ind, mannwhitneyu, levene, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for SPSS-like visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("=" * 100)
print("SPSS-STYLE DATA ANALYSIS FOR SHORT-FORM VIDEO USAGE STUDY")
print("=" * 100)

# Load the data
df = pd.read_csv('parista.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Rename columns for easier analysis
df.columns = ['Timestamp', 'Respondent', 'Gender', 'Age', 'Semester', 'College', 
              'Uses_YouTube_Shorts', 'Uses_TikTok_Reels', 'Preferred_Platform', 'Daily_Usage',
              'Use_Everyday', 'Open_Multiple_Times', 'Use_Free_Time', 'Part_of_Routine', 
              'More_Frequent_Than_Others', 'Spend_More_Than_Hour', 'Continue_Longer_Than_Planned',
              'Spend_Large_Free_Time', 'Lose_Track_of_Time', 'Spend_More_Than_Intended',
              'Watch_After_Waking', 'Watch_Before_Sleep', 'Watch_When_Free', 'Watch_During_Breaks',
              'Watch_Consecutively', 'Concentrate_Without_Distraction', 'Difficulty_Focused_Lectures',
              'Complete_Tasks_Without_Losing_Concentration', 'Easily_Distracted_Studying',
              'Pay_Attention_Long_Period']

print(f"Data loaded successfully!")
print(f"Total respondents: {len(df)}")
print(f"Total variables: {len(df.columns)}")

# ============================================================================
# 2. DATA PREPROCESSING - LIKERT SCALE MAPPING
# ============================================================================

print("\n" + "=" * 100)
print("2. DATA PREPROCESSING")
print("=" * 100)

# Define Likert scale mapping
likert_map = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

# Reverse mapping for display
likert_reverse = {
    1: 'Strongly Disagree',
    2: 'Disagree',
    3: 'Neutral',
    4: 'Agree',
    5: 'Strongly Agree'
}

# Identify Likert columns (from column 10 onward)
likert_columns = df.columns[10:]

# Convert Likert responses to numeric values
for col in likert_columns:
    df[col + '_numeric'] = df[col].map(likert_map)

# Define column groups for scales
use_columns = ['Use_Everyday', 'Open_Multiple_Times', 'Use_Free_Time', 'Part_of_Routine', 'More_Frequent_Than_Others']
duration_columns = ['Spend_More_Than_Hour', 'Continue_Longer_Than_Planned', 'Spend_Large_Free_Time', 
                    'Lose_Track_of_Time', 'Spend_More_Than_Intended']
habit_columns = ['Watch_After_Waking', 'Watch_Before_Sleep', 'Watch_When_Free', 'Watch_During_Breaks', 'Watch_Consecutively']
attention_columns = ['Concentrate_Without_Distraction', 'Difficulty_Focused_Lectures', 
                     'Complete_Tasks_Without_Losing_Concentration', 'Easily_Distracted_Studying',
                     'Pay_Attention_Long_Period']

# Create composite scores (average of each scale)
df['Avg_Use'] = df[[col + '_numeric' for col in use_columns if col + '_numeric' in df.columns]].mean(axis=1)
df['Avg_Duration'] = df[[col + '_numeric' for col in duration_columns if col + '_numeric' in df.columns]].mean(axis=1)
df['Avg_Habits'] = df[[col + '_numeric' for col in habit_columns if col + '_numeric' in df.columns]].mean(axis=1)

# For attention, we need to reverse code attention items where higher score = better attention
# Reverse code: Concentrate_Without_Distraction, Complete_Tasks_Without_Losing_Concentration, Pay_Attention_Long_Period
reverse_cols = ['Concentrate_Without_Distraction', 'Complete_Tasks_Without_Losing_Concentration', 'Pay_Attention_Long_Period']
for col in reverse_cols:
    if col + '_numeric' in df.columns:
        df[col + '_numeric_reversed'] = 6 - df[col + '_numeric']

# Create attention composite with reversed items
attention_cols_rev = ['Concentrate_Without_Distraction_numeric_reversed', 
                      'Difficulty_Focused_Lectures_numeric',
                      'Complete_Tasks_Without_Losing_Concentration_numeric_reversed',
                      'Easily_Distracted_Studying_numeric',
                      'Pay_Attention_Long_Period_numeric_reversed']

# Check which columns exist
attention_cols_exist = [col for col in attention_cols_rev if col in df.columns]
df['Avg_Attention'] = df[attention_cols_exist].mean(axis=1)

# Create Usage Category based on Daily_Usage
def categorize_usage(usage):
    if pd.isna(usage):
        return 'Unknown'
    elif usage == 'Less than 30 minutes':
        return 'Low'
    elif usage == '30 minutes–1 hour':
        return 'Medium'
    elif usage == '1–2 hours':
        return 'High'
    elif usage == 'More than 2 hours':
        return 'Very High'
    else:
        return 'Other'

df['Usage_Category'] = df['Daily_Usage'].apply(categorize_usage)

print(f"Composite scores created:")
print(f"  - Avg_Use: {len(df['Avg_Use'].dropna())} valid responses")
print(f"  - Avg_Duration: {len(df['Avg_Duration'].dropna())} valid responses")
print(f"  - Avg_Habits: {len(df['Avg_Habits'].dropna())} valid responses")
print(f"  - Avg_Attention: {len(df['Avg_Attention'].dropna())} valid responses")

# ============================================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "=" * 100)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 100)

def descriptive_stats_complete(data, variable_name):
    """Calculate comprehensive descriptive statistics"""
    data_clean = data.dropna()
    n = len(data_clean)
    
    if n == 0:
        return pd.Series(dtype=float)
    
    try:
        mode_val = data_clean.mode().iloc[0] if len(data_clean.mode()) > 0 else np.nan
    except:
        mode_val = np.nan
    
    stats_dict = {
        'N': n,
        'Mean': data_clean.mean(),
        'Median': data_clean.median(),
        'Mode': mode_val,
        'Std_Deviation': data_clean.std(),
        'Variance': data_clean.var(),
        'Minimum': data_clean.min(),
        'Maximum': data_clean.max(),
        'Range': data_clean.max() - data_clean.min(),
        'Skewness': data_clean.skew(),
        'Kurtosis': data_clean.kurtosis(),
        'Std_Error_Mean': data_clean.std() / np.sqrt(n),
        'Percentile_25': data_clean.quantile(0.25),
        'Percentile_75': data_clean.quantile(0.75),
        'IQR': data_clean.quantile(0.75) - data_clean.quantile(0.25)
    }
    return pd.Series(stats_dict, name=variable_name)

print("\n--- Descriptive Statistics for Composite Scores ---")
composite_vars = ['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']
desc_stats_df = pd.DataFrame()
for var in composite_vars:
    desc_stats_df = pd.concat([desc_stats_df, descriptive_stats_complete(df[var], var)], axis=1)

print(desc_stats_df.round(3))

# Save descriptive statistics
desc_stats_df.to_csv('parista_descriptive_statistics.csv')

# ============================================================================
# 4. FREQUENCY DISTRIBUTIONS
# ============================================================================

print("\n" + "=" * 100)
print("4. FREQUENCY DISTRIBUTIONS")
print("=" * 100)

def frequency_table_complete(data, variable_name):
    """Create comprehensive frequency table"""
    freq = data.value_counts().sort_index()
    percent = (freq / len(data) * 100).round(2)
    valid_percent = percent
    cumulative = percent.cumsum()
    
    table = pd.DataFrame({
        'Frequency': freq,
        'Percent': percent,
        'Valid Percent': valid_percent,
        'Cumulative Percent': cumulative
    })
    
    total_row = pd.DataFrame({
        'Frequency': [freq.sum()],
        'Percent': [100.0],
        'Valid Percent': [100.0],
        'Cumulative Percent': [100.0]
    }, index=['Total'])
    table = pd.concat([table, total_row])
    return table

print("\n--- Demographic Frequencies ---")
demo_cols = ['Gender', 'Age', 'Semester', 'College', 'Preferred_Platform', 'Daily_Usage']

for col in demo_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print("-" * 40)
        freq_table = frequency_table_complete(df[col], col)
        print(freq_table)

print("\n--- Likert Scale Frequencies (First 5 items) ---")
for col in use_columns[:3] + duration_columns[:2]:
    if col + '_numeric' in df.columns:
        print(f"\n{col}:")
        print("-" * 40)
        # Map numeric back to labels
        data = df[col + '_numeric'].map(likert_reverse)
        freq_table = frequency_table_complete(data, col)
        print(freq_table)

# ============================================================================
# 5. RELIABILITY ANALYSIS (Cronbach's Alpha)
# ============================================================================

print("\n" + "=" * 100)
print("5. RELIABILITY ANALYSIS - CRONBACH'S ALPHA")
print("=" * 100)

def cronbach_alpha_complete(items, verbose=True):
    """Calculate Cronbach's Alpha with comprehensive statistics"""
    items = np.array(items)
    n_items = items.shape[1]
    n_respondents = items.shape[0]
    
    item_variances = items.var(axis=0, ddof=1)
    item_means = items.mean(axis=0)
    item_sds = items.std(axis=0, ddof=1)
    
    total_scores = items.sum(axis=1)
    total_variance = total_scores.var(ddof=1)
    total_mean = total_scores.mean()
    total_sd = total_scores.std(ddof=1)
    
    alpha = (n_items / (n_items - 1)) * (1 - (item_variances.sum() / total_variance))
    
    item_total_corr = []
    alpha_if_deleted = []
    
    for i in range(n_items):
        total_without_item = np.delete(items, i, axis=1).sum(axis=1)
        corr, _ = pearsonr(items[:, i], total_without_item)
        item_total_corr.append(corr)
        
        items_without = np.delete(items, i, axis=1)
        var_without = items_without.var(axis=0, ddof=1).sum()
        total_var_without = items_without.sum(axis=1).var(ddof=1)
        alpha_if = ((n_items - 1) / (n_items - 2)) * (1 - (var_without / total_var_without)) if (n_items - 2) > 0 else np.nan
        alpha_if_deleted.append(alpha_if)
    
    if verbose:
        print(f"\nNumber of items: {n_items}")
        print(f"Number of respondents: {n_respondents}")
        print(f"Cronbach's Alpha: {alpha:.4f}")
        print(f"Total Scale Mean: {total_mean:.4f}")
        print(f"Total Scale Variance: {total_variance:.4f}")
        
        if alpha >= 0.9:
            print("Interpretation: Excellent reliability")
        elif alpha >= 0.8:
            print("Interpretation: Good reliability")
        elif alpha >= 0.7:
            print("Interpretation: Acceptable reliability")
        elif alpha >= 0.6:
            print("Interpretation: Questionable reliability")
        else:
            print("Interpretation: Poor reliability")
        
        print("\n--- Item-Total Statistics ---")
        print(f"{'Item':<15} {'Item-Total Corr':<18} {'Alpha if Deleted':<18}")
        print("-" * 55)
        for i in range(n_items):
            print(f"Item {i+1:<10} {item_total_corr[i]:<18.4f} {alpha_if_deleted[i]:<18.4f}")
    
    return alpha, item_total_corr, alpha_if_deleted

# Calculate reliability for each scale
scales = {
    'Frequency of Use': use_columns,
    'Duration of Use': duration_columns,
    'Viewing Habits': habit_columns,
    'Attention Span': attention_columns
}

reliability_results = {}
for scale_name, cols in scales.items():
    numeric_cols = [col + '_numeric' for col in cols if col + '_numeric' in df.columns]
    # For attention, use reversed items
    if scale_name == 'Attention Span':
        numeric_cols = attention_cols_exist
    
    if len(numeric_cols) > 1:
        data = df[numeric_cols].dropna()
        if len(data) > 1 and len(data.columns) > 1:
            print(f"\n{'='*50}")
            print(f"--- {scale_name} Scale ---")
            print(f"{'='*50}")
            alpha, item_corr, alpha_if = cronbach_alpha_complete(data.values, verbose=True)
            reliability_results[scale_name] = {
                'Alpha': alpha,
                'Item_Total_Corr': item_corr,
                'Alpha_If_Deleted': alpha_if
            }

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("6. CORRELATION ANALYSIS")
print("=" * 100)

# Create correlation data
correlation_vars = ['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']
correlation_data = df[correlation_vars].dropna()

# Pearson Correlation
print("\n--- Pearson Correlation Matrix ---")
pearson_corr = correlation_data.corr(method='pearson')
print(pearson_corr.round(3))

# Calculate p-values
def calculate_pvalues(data):
    p_values = pd.DataFrame(np.zeros((data.shape[1], data.shape[1])), 
                           columns=data.columns, index=data.columns)
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if i != j:
                corr, p_val = pearsonr(data.iloc[:, i], data.iloc[:, j])
                p_values.iloc[i, j] = p_val
    return p_values

p_values = calculate_pvalues(correlation_data)
print("\nP-values:")
print(p_values.round(4))

# Spearman Correlation
print("\n--- Spearman Correlation Matrix ---")
spearman_corr = correlation_data.corr(method='spearman')
print(spearman_corr.round(3))

# ============================================================================
# 7. COMPARATIVE TESTS
# ============================================================================

print("\n" + "=" * 100)
print("7. COMPARATIVE TESTS")
print("=" * 100)

# 7.1 Independent T-test (Gender differences)
print("\n--- Independent T-test: Gender Differences ---")

def t_test_complete(group1, group2, variable_name):
    """Perform comprehensive independent t-test"""
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    n1, n2 = len(g1), len(g2)
    mean1, mean2 = g1.mean(), g2.mean()
    std1, std2 = g1.std(), g2.std()
    var1, var2 = g1.var(), g2.var()
    
    if n1 > 1 and n2 > 1:
        levene_stat, levene_p = levene(g1, g2)
        
        if levene_p < 0.05:
            t_stat, p_val = ttest_ind(g1, g2, equal_var=False)
        else:
            t_stat, p_val = ttest_ind(g1, g2, equal_var=True)
        
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        d = (mean1 - mean2) / np.sqrt(pooled_var)
        
        print(f"\n{'='*60}")
        print(f"T-test Results for {variable_name}")
        print(f"{'='*60}")
        print(f"Levene's Test: F = {levene_stat:.4f}, p = {levene_p:.4f}")
        print(f"Group Statistics:")
        print(f"  Male:  n = {n1}, Mean = {mean1:.4f}, SD = {std1:.4f}")
        print(f"  Female: n = {n2}, Mean = {mean2:.4f}, SD = {std2:.4f}")
        print(f"T-test Results:")
        print(f"  t-statistic = {t_stat:.4f}")
        print(f"  p-value = {p_val:.4f}")
        print(f"  Cohen's d = {abs(d):.4f}")
        
        return t_stat, p_val, d

for var in ['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']:
    male_data = df[df['Gender'] == 'Male'][var]
    female_data = df[df['Gender'] == 'Female'][var]
    if len(male_data) > 1 and len(female_data) > 1:
        t_test_complete(male_data, female_data, var)

# 7.2 One-Way ANOVA (Usage Category differences)
print("\n--- One-Way ANOVA: Usage Category Differences ---")

def anova_complete(df, dependent_var, independent_var):
    """Perform comprehensive one-way ANOVA"""
    groups = df.groupby(independent_var)[dependent_var]
    group_data = [group.dropna().values for name, group in groups]
    group_names = [name for name, group in groups]
    
    valid_groups = [(name, data) for name, data in zip(group_names, group_data) if len(data) > 1]
    if len(valid_groups) < 2:
        return None, None, None
    
    valid_names = [g[0] for g in valid_groups]
    valid_data = [g[1] for g in valid_groups]
    
    print(f"\n{'='*60}")
    print(f"ANOVA Results for {dependent_var} by {independent_var}")
    print(f"{'='*60}")
    print("\nGroup Descriptive Statistics:")
    print(f"{'Group':<20} {'N':<10} {'Mean':<12} {'SD':<12}")
    print("-" * 55)
    for name, data in zip(valid_names, valid_data):
        n = len(data)
        mean = data.mean()
        sd = data.std()
        print(f"{name:<20} {n:<10} {mean:<12.4f} {sd:<12.4f}")
    
    f_stat, p_val = f_oneway(*valid_data)
    
    print(f"\nANOVA Results:")
    print(f"  F-statistic = {f_stat:.4f}")
    print(f"  p-value = {p_val:.4f}")
    
    return f_stat, p_val

for var in ['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']:
    anova_complete(df, var, 'Usage_Category')

# ============================================================================
# 8. REGRESSION ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("8. MULTIPLE LINEAR REGRESSION")
print("=" * 100)

print("Dependent Variable: Avg_Attention")
print("Independent Variables: Avg_Use, Avg_Duration, Avg_Habits")

# Prepare data for regression
regression_data = df[['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']].dropna()
X = regression_data[['Avg_Use', 'Avg_Duration', 'Avg_Habits']]
y = regression_data['Avg_Attention']

if len(X) > 0:
    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Add constant
    X_with_const = sm.add_constant(X_scaled)
    
    # Fit model
    model = sm.OLS(y, X_with_const).fit()
    
    print("\n--- Model Summary ---")
    print("=" * 80)
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"p-value (F-test): {model.f_pvalue:.4f}")
    print(f"Number of Observations: {model.nobs}")
    
    print("\n--- Coefficients ---")
    print("=" * 80)
    print(f"{'Variable':<30} {'Coefficient':<15} {'Std Error':<15} {'t-value':<15} {'p-value':<15}")
    print("-" * 90)
    
    for var in model.params.index:
        print(f"{var:<30} {model.params[var]:<15.4f} {model.bse[var]:<15.4f} {model.tvalues[var]:<15.4f} {model.pvalues[var]:<15.4f}")
    
    # VIF for multicollinearity
    print("\n--- Variance Inflation Factor (VIF) ---")
    print("=" * 80)
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    try:
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(vif_data)
    except:
        print("VIF calculation not available")

# ============================================================================
# 9. CROSS-TABULATION
# ============================================================================

print("\n" + "=" * 100)
print("9. CROSS-TABULATION WITH CHI-SQUARE TESTS")
print("=" * 100)

def cross_tab_complete(df, row_var, col_var, title=""):
    """Perform comprehensive cross-tabulation with chi-square test"""
    if row_var not in df.columns or col_var not in df.columns:
        print(f"\nWarning: Columns not found for {row_var} or {col_var}")
        return None, None, None
    
    crosstab = pd.crosstab(df[row_var], df[col_var])
    
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        return None, None, None
    
    try:
        chi2, p_val, dof, expected_freq = chi2_contingency(crosstab)
        
        print(f"\n{'='*60}")
        print(f"Cross-Tabulation: {title}")
        print(f"{'='*60}")
        print(f"\n--- Cross-Tabulation Table ---")
        crosstab_with_totals = crosstab.copy()
        crosstab_with_totals['Total'] = crosstab.sum(axis=1)
        total_row = pd.Series(crosstab.sum(axis=0), name='Total')
        crosstab_with_totals = pd.concat([crosstab_with_totals, total_row.to_frame().T])
        print(crosstab_with_totals)
        
        print(f"\n--- Chi-Square Test Results ---")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"p-value: {p_val:.4f}")
        
        n_total = crosstab.sum().sum()
        min_dim = min(crosstab.shape) - 1
        cramer_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0
        print(f"Cramer's V: {cramer_v:.4f}")
        
        if p_val < 0.05:
            print("\n  ✓ Significant association found at p < 0.05")
        else:
            print("\n  ✗ No significant association found")
        
        return chi2, p_val, cramer_v
    except Exception as e:
        print(f"Error in chi-square test: {e}")
        return None, None, None

# Cross-tabulations
cross_tab_complete(df, 'Gender', 'Preferred_Platform', "Gender vs Preferred Platform")
cross_tab_complete(df, 'Usage_Category', 'Preferred_Platform', "Usage Category vs Preferred Platform")
cross_tab_complete(df, 'Gender', 'Usage_Category', "Gender vs Usage Category")

# ============================================================================
# 10. GENERATING VISUALIZATIONS (FIXED)
# ============================================================================

print("\n" + "=" * 100)
print("10. GENERATING VISUALIZATIONS")
print("=" * 100)

# Create directory for results if it doesn't exist
import os
if not os.path.exists('parista_analysis_results'):
    os.makedirs('parista_analysis_results')

# Create comprehensive figure
fig = plt.figure(figsize=(24, 20))

# 1. Boxplots for composite variables
ax1 = plt.subplot(3, 4, 1)
data_to_plot = [df[var].dropna() for var in ['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']]
bp = ax1.boxplot(data_to_plot, labels=['Use', 'Duration', 'Habits', 'Attention'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax1.axhline(y=3, color='red', linestyle='--', alpha=0.5)
ax1.set_title('Boxplots of All Variables')
ax1.set_ylabel('Score (1-5)')
ax1.tick_params(axis='x', rotation=45)

# 2. Histogram of Attention
ax2 = plt.subplot(3, 4, 2)
attention_data = df['Avg_Attention'].dropna()
ax2.hist(attention_data, bins=10, edgecolor='black', color='lightgreen', alpha=0.7)
ax2.axvline(attention_data.mean(), color='red', linestyle='--', 
           label=f'Mean: {attention_data.mean():.2f}')
ax2.axvline(attention_data.median(), color='blue', linestyle='--',
           label=f'Median: {attention_data.median():.2f}')
ax2.set_xlabel('Attention Score')
ax2.set_ylabel('Frequency')
ax2.set_title('Attention Span Distribution')
ax2.legend()

# 3. Correlation Heatmap
ax3 = plt.subplot(3, 4, 3)
sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax3, cbar_kws={'shrink': 0.8})
ax3.set_title('Pearson Correlation Heatmap')

# 4. Scatter plot: Use vs Attention
ax4 = plt.subplot(3, 4, 4)
use_data = df['Avg_Use']
att_data = df['Avg_Attention']
common_idx = use_data.notna() & att_data.notna()
use_clean = use_data[common_idx]
att_clean = att_data[common_idx]

ax4.scatter(use_clean, att_clean, alpha=0.6)
if len(use_clean) > 1:
    z = np.polyfit(use_clean, att_clean, 1)
    p = np.poly1d(z)
    ax4.plot(np.sort(use_clean), p(np.sort(use_clean)), "r-", alpha=0.8)
    
    # Add correlation
    corr, p_val = pearsonr(use_clean, att_clean)
    ax4.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.4f}', 
            transform=ax4.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax4.set_xlabel('Usage Score')
ax4.set_ylabel('Attention Score')
ax4.set_title('Usage vs Attention')

# 5. Bar chart with error bars
ax5 = plt.subplot(3, 4, 5)
category_means = df[['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']].mean()
category_stds = df[['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']].std()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax5.bar(category_means.index, category_means, yerr=category_stds, capsize=5, color=colors)
ax5.axhline(y=3, color='red', linestyle='--', alpha=0.5)
ax5.set_ylabel('Mean Score')
ax5.set_title('Mean Scores with Error Bars')
ax5.tick_params(axis='x', rotation=45)

# 6. Gender comparison
ax6 = plt.subplot(3, 4, 6)
valid_gender = df['Gender'].notna()
gender_means = df['Avg_Attention'][valid_gender].groupby(df['Gender'][valid_gender]).mean()
gender_std = df['Avg_Attention'][valid_gender].groupby(df['Gender'][valid_gender]).std()
bars = ax6.bar(gender_means.index, gender_means, yerr=gender_std, capsize=5, 
        color=['pink', 'lightblue'])
ax6.set_ylabel('Mean Attention')
ax6.set_title('Attention by Gender')

# 7. Usage Category comparison
ax7 = plt.subplot(3, 4, 7)
valid_cat = df['Usage_Category'].notna()
cat_means = df['Avg_Attention'][valid_cat].groupby(df['Usage_Category'][valid_cat]).mean()
cat_std = df['Avg_Attention'][valid_cat].groupby(df['Usage_Category'][valid_cat]).std()
bars = ax7.bar(cat_means.index, cat_means, yerr=cat_std, capsize=5, color='lightgreen')
ax7.set_ylabel('Mean Attention')
ax7.set_title('Attention by Usage Category')
ax7.tick_params(axis='x', rotation=45)

# 8. Age comparison
ax8 = plt.subplot(3, 4, 8)
valid_age = df['Age'].notna()
age_means = df['Avg_Attention'][valid_age].groupby(df['Age'][valid_age]).mean()
age_std = df['Avg_Attention'][valid_age].groupby(df['Age'][valid_age]).std()
bars = ax8.bar(age_means.index, age_means, yerr=age_std, capsize=5, color='coral')
ax8.set_ylabel('Mean Attention')
ax8.set_title('Attention by Age')
ax8.tick_params(axis='x', rotation=45)

# 9. Pie chart - Gender
ax9 = plt.subplot(3, 4, 9)
gender_counts = df['Gender'].value_counts()
ax9.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
         startangle=90, colors=['pink', 'lightblue'])
ax9.set_title('Gender Distribution')

# 10. Pie chart - Preferred Platform
ax10 = plt.subplot(3, 4, 10)
platform_counts = df['Preferred_Platform'].value_counts()
ax10.pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%', 
         startangle=90)
ax10.set_title('Preferred Platform Distribution')

# 11. Pie chart - Usage Category
ax11 = plt.subplot(3, 4, 11)
usage_counts = df['Usage_Category'].value_counts()
ax11.pie(usage_counts.values, labels=usage_counts.index, autopct='%1.1f%%', 
         startangle=90, colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax11.set_title('Usage Category Distribution')

# 12. Regression plot
ax12 = plt.subplot(3, 4, 12)
if 'model' in locals():
    y_pred = model.predict(X_with_const)
    ax12.scatter(y, y_pred, alpha=0.6)
    ax12.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax12.set_xlabel('Actual Attention')
    ax12.set_ylabel('Predicted Attention')
    ax12.set_title(f'Actual vs Predicted (R²={model.rsquared:.3f})')
    
    # Add correlation
    corr, p_val = pearsonr(y, y_pred)
    ax12.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.4f}', 
            transform=ax12.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
else:
    ax12.text(0.5, 0.5, 'Regression model not available', 
              horizontalalignment='center', verticalalignment='center',
              transform=ax12.transAxes)
    ax12.set_title('Regression Plot Not Available')

plt.tight_layout()

# Save the figure - directory already exists
try:
    plt.savefig('parista_analysis_results/comprehensive_visualizations.png', dpi=300, bbox_inches='tight')
    print("\nVisualizations saved to: parista_analysis_results/comprehensive_visualizations.png")
except Exception as e:
    print(f"\nError saving visualization: {e}")
    # Try saving in current directory as fallback
    plt.savefig('comprehensive_visualizations.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as: comprehensive_visualizations.png (in current directory)")

plt.show()

# ============================================================================
# 11. HYPOTHESIS TESTING SUMMARY (DEFINED BEFORE EXPORT)
# ============================================================================

print("\n" + "=" * 100)
print("11. HYPOTHESIS TESTING SUMMARY")
print("=" * 100)

# Collect p-values
p_values = {}

# Get correlation p-values
for var in ['Avg_Use', 'Avg_Duration', 'Avg_Habits']:
    if var in df.columns:
        x = df[var]
        y = df['Avg_Attention']
        common_idx = x.notna() & y.notna()
        x_aligned = x[common_idx]
        y_aligned = y[common_idx]
        if len(x_aligned) > 2:
            corr, p_val = pearsonr(x_aligned, y_aligned)
            p_values[f'Corr_{var}'] = p_val

# Get ANOVA p-values
for var in ['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']:
    groups = []
    for cat in df['Usage_Category'].unique():
        data = df[var][df['Usage_Category'] == cat].dropna()
        if len(data) > 1:
            groups.append(data)
    if len(groups) >= 2:
        try:
            f_stat, p_val = f_oneway(*groups)
            p_values[f'ANOVA_{var}'] = p_val
        except:
            pass

# Create hypotheses
hypotheses = [
    {
        'Hypothesis': 'H1: Frequency of use significantly affects attention span',
        'Test': 'Pearson Correlation',
        'Result': 'Rejected' if p_values.get('Corr_Avg_Use', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('Corr_Avg_Use', 'N/A'),
        'Conclusion': 'Use correlates with attention' if p_values.get('Corr_Avg_Use', 1) < 0.05 else 'No significant correlation'
    },
    {
        'Hypothesis': 'H2: Duration of use significantly affects attention span',
        'Test': 'Pearson Correlation',
        'Result': 'Rejected' if p_values.get('Corr_Avg_Duration', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('Corr_Avg_Duration', 'N/A'),
        'Conclusion': 'Duration correlates with attention' if p_values.get('Corr_Avg_Duration', 1) < 0.05 else 'No significant correlation'
    },
    {
        'Hypothesis': 'H3: Viewing habits significantly affect attention span',
        'Test': 'Pearson Correlation',
        'Result': 'Rejected' if p_values.get('Corr_Avg_Habits', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('Corr_Avg_Habits', 'N/A'),
        'Conclusion': 'Habits correlate with attention' if p_values.get('Corr_Avg_Habits', 1) < 0.05 else 'No significant correlation'
    },
    {
        'Hypothesis': 'H4: Usage category significantly affects attention span',
        'Test': 'One-Way ANOVA',
        'Result': 'Rejected' if p_values.get('ANOVA_Avg_Attention', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('ANOVA_Avg_Attention', 'N/A'),
        'Conclusion': 'Category differences exist' if p_values.get('ANOVA_Avg_Attention', 1) < 0.05 else 'No differences'
    },
    {
        'Hypothesis': 'H5: Usage significantly affects duration of use',
        'Test': 'One-Way ANOVA',
        'Result': 'Rejected' if p_values.get('ANOVA_Avg_Use', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('ANOVA_Avg_Use', 'N/A'),
        'Conclusion': 'Usage affects duration' if p_values.get('ANOVA_Avg_Use', 1) < 0.05 else 'No effect'
    },
    {
        'Hypothesis': 'H6: All factors together predict attention span',
        'Test': 'Multiple Regression',
        'Result': 'Rejected' if 'model' in locals() and model.f_pvalue < 0.05 else 'Not Rejected',
        'p-value': model.f_pvalue if 'model' in locals() else 'N/A',
        'Conclusion': 'Model is significant' if 'model' in locals() and model.f_pvalue < 0.05 else 'Model is not significant'
    }
]

print("\nHypothesis Testing Summary Table:")
print("=" * 120)
print(f"{'Hypothesis':<70} {'Test':<25} {'Result':<15} {'p-value':<12} {'Conclusion':<30}")
print("-" * 120)
for h in hypotheses:
    p_val_str = f"{h['p-value']:.4f}" if isinstance(h['p-value'], float) else h['p-value']
    print(f"{h['Hypothesis']:<70} {h['Test']:<25} {h['Result']:<15} {p_val_str:<12} {h['Conclusion']:<30}")

# ============================================================================
# 12. DATA EXPORT (COMPLETE WITH ALL CSV FILES)
# ============================================================================

print("\n" + "=" * 100)
print("12. DATA EXPORT - ALL CSV FILES")
print("=" * 100)

# Create directory for results
import os
if not os.path.exists('parista_analysis_results'):
    os.makedirs('parista_analysis_results')

# 1. Export Descriptive Statistics
desc_stats_df.to_csv('parista_analysis_results/01_descriptive_statistics.csv')
print("✓ 01_descriptive_statistics.csv - Mean, Median, Mode, Std Dev, etc.")

# 2. Export Correlation Matrices
pearson_corr.to_csv('parista_analysis_results/02_pearson_correlation.csv')
print("✓ 02_pearson_correlation.csv - Pearson correlation matrix with p-values")

spearman_corr.to_csv('parista_analysis_results/03_spearman_correlation.csv')
print("✓ 03_spearman_correlation.csv - Spearman correlation matrix")

# 3. Export Reliability Results
if 'reliability_results' in locals():
    reliability_df = pd.DataFrame({
        'Scale': list(reliability_results.keys()),
        'Cronbach_Alpha': [v['Alpha'] for v in reliability_results.values()]
    })
    reliability_df.to_csv('parista_analysis_results/04_reliability_analysis.csv')
    print("✓ 04_reliability_analysis.csv - Cronbach's Alpha for each scale")

# 4. Export Regression Results
if 'model' in locals():
    regression_summary = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std_Error': model.bse.values,
        't_value': model.tvalues.values,
        'p_value': model.pvalues.values,
        'Lower_95_CI': model.conf_int()[0],
        'Upper_95_CI': model.conf_int()[1]
    })
    regression_summary.to_csv('parista_analysis_results/05_regression_coefficients.csv')
    print("✓ 05_regression_coefficients.csv - Regression coefficients with stats")

# 5. Export Composite Scores
composite_data = df[['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']].copy()
composite_data.to_csv('parista_analysis_results/06_composite_scores.csv')
print("✓ 06_composite_scores.csv - Composite scores for all respondents")

# 6. Export Frequency Tables - Demographics
for col in ['Gender', 'Age', 'Semester', 'College', 'Preferred_Platform', 'Daily_Usage']:
    if col in df.columns:
        freq_table = frequency_table_complete(df[col], col)
        freq_table.to_csv(f'parista_analysis_results/07_frequency_{col}.csv')
        print(f"✓ 07_frequency_{col}.csv - Frequency distribution for {col}")

# 7. Export Frequency Tables - Likert Items
for col in use_columns[:3] + duration_columns[:2] + habit_columns[:2]:
    if col + '_numeric' in df.columns:
        data = df[col + '_numeric'].map(likert_reverse)
        freq_table = frequency_table_complete(data, col)
        freq_table.to_csv(f'parista_analysis_results/08_frequency_{col}.csv')
        print(f"✓ 08_frequency_{col}.csv - Likert frequency for {col}")

# 8. Export Cleaned Data
df.to_csv('parista_analysis_results/09_cleaned_data.csv', index=False)
print("✓ 09_cleaned_data.csv - Complete cleaned dataset with all variables")

# 9. Export Hypothesis Testing Summary
if 'hypotheses' in locals():
    hypotheses_df = pd.DataFrame(hypotheses)
    hypotheses_df.to_csv('parista_analysis_results/10_hypothesis_testing_summary.csv', index=False)
    print("✓ 10_hypothesis_testing_summary.csv - Hypothesis testing results")

# 10. Export T-test Results (if available)
t_test_results_list = []
for var in ['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']:
    male_data = df[df['Gender'] == 'Male'][var].dropna()
    female_data = df[df['Gender'] == 'Female'][var].dropna()
    if len(male_data) > 1 and len(female_data) > 1:
        try:
            # Get t-test results
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(male_data, female_data)
            t_test_results_list.append({
                'Variable': var,
                'Male_Mean': male_data.mean(),
                'Male_SD': male_data.std(),
                'Female_Mean': female_data.mean(),
                'Female_SD': female_data.std(),
                't_statistic': t_stat,
                'p_value': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })
        except:
            pass

if t_test_results_list:
    t_test_df = pd.DataFrame(t_test_results_list)
    t_test_df.to_csv('parista_analysis_results/11_t_test_results.csv', index=False)
    print("✓ 11_t_test_results.csv - T-test results for gender differences")

# 11. Export ANOVA Results
anova_results_list = []
for var in ['Avg_Use', 'Avg_Duration', 'Avg_Habits', 'Avg_Attention']:
    groups = []
    group_names = []
    for cat in df['Usage_Category'].unique():
        data = df[var][df['Usage_Category'] == cat].dropna()
        if len(data) > 1:
            groups.append(data)
            group_names.append(cat)
    if len(groups) >= 2:
        try:
            f_stat, p_val = f_oneway(*groups)
            anova_results_list.append({
                'Variable': var,
                'F_statistic': f_stat,
                'p_value': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No',
                'Groups': ', '.join(group_names)
            })
        except:
            pass

if anova_results_list:
    anova_df = pd.DataFrame(anova_results_list)
    anova_df.to_csv('parista_analysis_results/12_anova_results.csv', index=False)
    print("✓ 12_anova_results.csv - ANOVA results for usage category differences")

# 12. Export Cross-tabulation Results
cross_tab_list = []
cross_tabs = [
    ('Gender', 'Preferred_Platform', "Gender vs Preferred Platform"),
    ('Usage_Category', 'Preferred_Platform', "Usage Category vs Preferred Platform"),
    ('Gender', 'Usage_Category', "Gender vs Usage Category")
]

for row_var, col_var, title in cross_tabs:
    if row_var in df.columns and col_var in df.columns:
        crosstab = pd.crosstab(df[row_var], df[col_var])
        if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
            try:
                chi2, p_val, dof, expected = chi2_contingency(crosstab)
                n_total = crosstab.sum().sum()
                min_dim = min(crosstab.shape) - 1
                cramer_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0
                
                cross_tab_list.append({
                    'Cross_tab': title,
                    'Chi_square': chi2,
                    'df': dof,
                    'p_value': p_val,
                    'Cramer_V': cramer_v,
                    'Significant': 'Yes' if p_val < 0.05 else 'No'
                })
                
                # Save individual crosstab
                crosstab.to_csv(f'parista_analysis_results/13_crosstab_{row_var}_vs_{col_var}.csv')
                print(f"✓ 13_crosstab_{row_var}_vs_{col_var}.csv - Cross-tabulation table")
            except:
                pass

if cross_tab_list:
    cross_tab_df = pd.DataFrame(cross_tab_list)
    cross_tab_df.to_csv('parista_analysis_results/14_cross_tab_summary.csv', index=False)
    print("✓ 14_cross_tab_summary.csv - Summary of cross-tabulation tests")

# ============================================================================
# 13. SUMMARY OF ALL GENERATED CSV FILES
# ============================================================================

print("\n" + "=" * 100)
print("13. SUMMARY OF ALL GENERATED CSV FILES")
print("=" * 100)

print("""
All CSV files are saved in the 'parista_analysis_results' folder:

1. 01_descriptive_statistics.csv       - Mean, Median, Mode, Std Dev, Skewness, Kurtosis
2. 02_pearson_correlation.csv          - Pearson correlation matrix with p-values
3. 03_spearman_correlation.csv         - Spearman correlation matrix
4. 04_reliability_analysis.csv         - Cronbach's Alpha for each scale
5. 05_regression_coefficients.csv      - Regression coefficients with standard errors
6. 06_composite_scores.csv             - Individual composite scores for all respondents
7. 07_frequency_*.csv                  - Frequency distributions for demographic variables
8. 08_frequency_*.csv                  - Likert scale frequency distributions
9. 09_cleaned_data.csv                 - Complete cleaned dataset
10. 10_hypothesis_testing_summary.csv   - Summary of all hypothesis tests
11. 11_t_test_results.csv              - T-test results for gender differences
12. 12_anova_results.csv               - ANOVA results for usage category differences
13. 13_crosstab_*.csv                  - Cross-tabulation tables
14. 14_cross_tab_summary.csv           - Summary of chi-square tests

Additional files:
- comprehensive_visualizations.png     - Main visualization dashboard
- additional_scatter_plots.png         - Additional scatter plots
""")

print("\n" + "=" * 100)
print("EXPORT COMPLETE! All files saved to: parista_analysis_results/")
print("=" * 100)

# ============================================================================
# 14. OPTIONAL: CREATE EXCEL FILE WITH ALL DATA
# ============================================================================

print("\n" + "=" * 100)
print("14. CREATING EXCEL FILE WITH ALL CSV DATA")
print("=" * 100)

try:
    with pd.ExcelWriter('parista_analysis_results/complete_analysis.xlsx', engine='openpyxl') as writer:
        for file in os.listdir('parista_analysis_results'):
            if file.endswith('.csv') and not file.startswith('._'):
                try:
                    df_temp = pd.read_csv(f'parista_analysis_results/{file}')
                    sheet_name = file.replace('.csv', '')[:31]  # Excel sheet name max 31 chars
                    df_temp.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"✓ Added {file} to Excel workbook")
                except Exception as e:
                    print(f"  Could not add {file}: {e}")
    
    print("\n✓ Excel file created: parista_analysis_results/complete_analysis.xlsx")
    print("  This file contains all CSV data in separate sheets for easy viewing")
except Exception as e:
    print(f"\nCould not create Excel file: {e}")
    print("  You can still view individual CSV files in the folder")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE! 🎉")
print("=" * 100)