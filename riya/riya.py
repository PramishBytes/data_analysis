import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, f_oneway, ttest_ind, mannwhitneyu, levene, kruskal, friedmanchisquare
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for SPSS-like visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory
output_dir = '/home/pramiz/Desktop/riya/Analysis_Results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=" * 100)
print("SPSS-STYLE STATISTICAL ANALYSIS - MOBILE NETWORK PROVIDER SATISFACTION STUDY (RIYA)")
print("=" * 100)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n" + "=" * 100)
print("1. DATA LOADING AND PREPROCESSING")
print("=" * 100)

# Load the data
file_path = '/home/pramiz/Desktop/riya/riya.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Clean column names
df.columns = df.columns.str.strip()

print(f"\nDataset Shape: {df.shape} rows x {df.shape[1]} columns")
print(f"Number of Respondents: {len(df)}")

# Rename columns for easier analysis
df.columns = ['Timestamp', 'Name', 'Age', 'Gender', 'Marital_Status', 'Education_Level', 
              'Occupation', 'Family_Income', 'Mobile_Operator', 'Tenure', 'Usage_Frequency', 
              'Usage_Reason', 'Monthly_Spend',
              'Service_Quality_1', 'Service_Quality_2', 'Service_Quality_3', 'Service_Quality_4',
              'Pricing_1', 'Pricing_2', 'Pricing_3', 'Pricing_4',
              'Assurance_1', 'Assurance_2', 'Assurance_3', 'Assurance_4',
              'Satisfaction_1', 'Satisfaction_2', 'Satisfaction_3', 'Satisfaction_4']

# Define Likert scale mapping
likert_map = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5,
    'Neeutral': 3,
    np.nan: np.nan
}

# Identify Likert columns
likert_columns = ['Service_Quality_1', 'Service_Quality_2', 'Service_Quality_3', 'Service_Quality_4',
                  'Pricing_1', 'Pricing_2', 'Pricing_3', 'Pricing_4',
                  'Assurance_1', 'Assurance_2', 'Assurance_3', 'Assurance_4',
                  'Satisfaction_1', 'Satisfaction_2', 'Satisfaction_3', 'Satisfaction_4']

# Convert Likert responses to numeric values
for col in likert_columns:
    df[col + '_numeric'] = df[col].map(likert_map)

# Create composite scores
service_quality_columns = ['Service_Quality_1', 'Service_Quality_2', 'Service_Quality_3', 'Service_Quality_4']
pricing_columns = ['Pricing_1', 'Pricing_2', 'Pricing_3', 'Pricing_4']
assurance_columns = ['Assurance_1', 'Assurance_2', 'Assurance_3', 'Assurance_4']
satisfaction_columns = ['Satisfaction_1', 'Satisfaction_2', 'Satisfaction_3', 'Satisfaction_4']

df['Avg_Service_Quality'] = df[[col + '_numeric' for col in service_quality_columns if col + '_numeric' in df.columns]].mean(axis=1)
df['Avg_Pricing'] = df[[col + '_numeric' for col in pricing_columns if col + '_numeric' in df.columns]].mean(axis=1)
df['Avg_Assurance'] = df[[col + '_numeric' for col in assurance_columns if col + '_numeric' in df.columns]].mean(axis=1)
df['Avg_Satisfaction'] = df[[col + '_numeric' for col in satisfaction_columns if col + '_numeric' in df.columns]].mean(axis=1)

# Create categorical variable for income
def categorize_income(income):
    if pd.isna(income):
        return 'Unknown'
    income_str = str(income).strip()
    if 'Below' in income_str or 'below' in income_str:
        return 'Low'
    elif '50000' in income_str and '75000' in income_str:
        return 'Medium-Low'
    elif 'Above' in income_str or 'above' in income_str:
        return 'High'
    else:
        return 'Unknown'

df['Income_Category'] = df['Family_Income'].apply(categorize_income)

# Create categorical variable for age
def categorize_age(age):
    if pd.isna(age):
        return 'Unknown'
    age_str = str(age).strip()
    if '20 years and below' in age_str:
        return 'Below 20'
    elif '21-30 years' in age_str:
        return '21-30'
    elif 'Above 30' in age_str:
        return 'Above 30'
    else:
        return 'Unknown'

df['Age_Category'] = df['Age'].apply(categorize_age)

print("\nComposite scores created:")
print(f"  - Avg_Service_Quality: {len(service_quality_columns)} items")
print(f"  - Avg_Pricing: {len(pricing_columns)} items")
print(f"  - Avg_Assurance: {len(assurance_columns)} items")
print(f"  - Avg_Satisfaction: {len(satisfaction_columns)} items")

# ============================================================================
# 1. DESCRIPTIVE STATISTICS (COMPLETE)
# ============================================================================

print("\n" + "=" * 100)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 100)

def descriptive_stats_complete(data, variable_name):
    """Calculate comprehensive descriptive statistics"""
    data_clean = data.dropna()
    n = len(data_clean)
    
    if n == 0:
        return pd.Series([np.nan]*17, 
                        index=['N', 'Mean', 'Median', 'Mode', 'Std_Deviation', 'Variance', 
                               'Minimum', 'Maximum', 'Range', 'Skewness', 'Kurtosis', 
                               'Std_Error_Mean', 'Sum', 'Percentile_25', 'Percentile_75', 'IQR', 'CV'],
                        name=variable_name)
    
    stats_dict = {
        'N': n,
        'Mean': data_clean.mean(),
        'Median': data_clean.median(),
        'Mode': data_clean.mode().iloc[0] if len(data_clean.mode()) > 0 else np.nan,
        'Std_Deviation': data_clean.std(),
        'Variance': data_clean.var(),
        'Minimum': data_clean.min(),
        'Maximum': data_clean.max(),
        'Range': data_clean.max() - data_clean.min(),
        'Skewness': data_clean.skew(),
        'Kurtosis': data_clean.kurtosis(),
        'Std_Error_Mean': data_clean.std() / np.sqrt(n),
        'Sum': data_clean.sum(),
        'Percentile_25': data_clean.quantile(0.25),
        'Percentile_75': data_clean.quantile(0.75),
        'IQR': data_clean.quantile(0.75) - data_clean.quantile(0.25),
        'CV': (data_clean.std() / data_clean.mean()) * 100 if data_clean.mean() != 0 else np.nan
    }
    return pd.Series(stats_dict, name=variable_name)

print("\n--- Descriptive Statistics for Scale Variables ---")
scale_variables = ['Avg_Service_Quality', 'Avg_Pricing', 'Avg_Assurance', 'Avg_Satisfaction']
desc_stats_df = pd.DataFrame()
for var in scale_variables:
    if var in df.columns:
        desc_stats_df = pd.concat([desc_stats_df, descriptive_stats_complete(df[var], var)], axis=1)

if not desc_stats_df.empty:
    print(desc_stats_df.round(3))
    desc_stats_df.to_csv(f'{output_dir}/descriptive_statistics_complete.csv')

print("\n--- Descriptive Statistics for Individual Likert Items ---")
likert_stats = pd.DataFrame()
for col in likert_columns:
    numeric_col = col + '_numeric'
    if numeric_col in df.columns and col in df.columns:
        likert_stats = pd.concat([likert_stats, descriptive_stats_complete(df[numeric_col], col)], axis=1)

if not likert_stats.empty:
    print(likert_stats.iloc[:, :10].round(3))
    likert_stats.to_csv(f'{output_dir}/likert_descriptive_statistics.csv')

# ============================================================================
# 2. FREQUENCY DISTRIBUTIONS (COMPLETE)
# ============================================================================

print("\n" + "=" * 100)
print("2. FREQUENCY DISTRIBUTIONS")
print("=" * 100)

def frequency_table_complete(data, variable_name):
    """Create comprehensive frequency table with percentages"""
    if data is None or len(data.dropna()) == 0:
        return pd.DataFrame({'Frequency': [], 'Percent': [], 'Valid Percent': [], 'Cumulative Percent': []})
    
    data_clean = data.dropna()
    freq = data_clean.value_counts().sort_index()
    percent = (freq / len(data_clean) * 100).round(2)
    valid_percent = percent
    cumulative = percent.cumsum()
    
    table = pd.DataFrame({
        'Frequency': freq,
        'Percent': percent,
        'Valid Percent': valid_percent,
        'Cumulative Percent': cumulative
    })
    # Add total row
    total_row = pd.DataFrame({
        'Frequency': [freq.sum()],
        'Percent': [100.0],
        'Valid Percent': [100.0],
        'Cumulative Percent': [100.0]
    }, index=['Total'])
    table = pd.concat([table, total_row])
    return table

print("\n--- Likert Scale Frequencies ---")
likert_freq_dict = {}
for col in service_quality_columns + pricing_columns + assurance_columns + satisfaction_columns:
    if col in df.columns and col + '_numeric' in df.columns:
        # Get original Likert labels
        numeric_values = df[col + '_numeric'].dropna()
        if len(numeric_values) > 0:
            # Map back to labels
            rev_map = {1: 'Strongly Disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly Agree'}
            labeled_data = numeric_values.map(rev_map)
            freq_table = frequency_table_complete(labeled_data, col)
            likert_freq_dict[col] = freq_table
            print(f"\n{col}:")
            print(freq_table)
            if not freq_table.empty:
                freq_table.to_csv(f'{output_dir}/frequency_likert_{col}.csv')

print("\n--- Demographic Frequencies ---")
demo_columns = ['Age', 'Gender', 'Marital_Status', 'Education_Level', 'Occupation', 
                'Family_Income', 'Mobile_Operator', 'Tenure', 'Usage_Frequency', 
                'Usage_Reason', 'Monthly_Spend']

demo_freq_dict = {}
for col in demo_columns:
    if col in df.columns:
        freq_table = frequency_table_complete(df[col], col)
        demo_freq_dict[col] = freq_table
        print(f"\n{col}:")
        print(freq_table)
        if not freq_table.empty:
            freq_table.to_csv(f'{output_dir}/frequency_demo_{col}.csv')

# ============================================================================
# 3. RELIABILITY ANALYSIS (COMPLETE)
# ============================================================================

print("\n" + "=" * 100)
print("3. RELIABILITY ANALYSIS")
print("=" * 100)

def cronbach_alpha_complete(items, verbose=True):
    """Calculate Cronbach's Alpha with comprehensive item-total statistics"""
    items = np.array(items)
    n_items = items.shape[1]
    n_respondents = items.shape[0]
    
    # Calculate item variances
    item_variances = items.var(axis=0, ddof=1)
    item_means = items.mean(axis=0)
    item_sds = items.std(axis=0, ddof=1)
    
    # Calculate total variance
    total_scores = items.sum(axis=1)
    total_variance = total_scores.var(ddof=1)
    total_mean = total_scores.mean()
    total_sd = total_scores.std(ddof=1)
    
    # Calculate Cronbach's Alpha
    alpha = (n_items / (n_items - 1)) * (1 - (item_variances.sum() / total_variance))
    
    # Item-total statistics
    item_total_corr = []
    alpha_if_deleted = []
    item_means_if_deleted = []
    item_variances_if_deleted = []
    
    for i in range(n_items):
        # Correlation of item i with total score (excluding item i)
        total_without_item = np.delete(items, i, axis=1).sum(axis=1)
        corr, _ = pearsonr(items[:, i], total_without_item)
        item_total_corr.append(corr)
        
        # Alpha if item deleted
        items_without = np.delete(items, i, axis=1)
        var_without = items_without.var(axis=0, ddof=1).sum()
        total_var_without = items_without.sum(axis=1).var(ddof=1)
        alpha_if = ((n_items - 1) / (n_items - 2)) * (1 - (var_without / total_var_without))
        alpha_if_deleted.append(alpha_if)
        
        # Mean if item deleted
        item_means_if_deleted.append(total_without_item.mean())
        item_variances_if_deleted.append(total_without_item.var(ddof=1))
    
    if verbose:
        print(f"\n--- Scale Statistics ---")
        print(f"Number of items: {n_items}")
        print(f"Number of respondents: {n_respondents}")
        print(f"Cronbach's Alpha: {alpha:.4f}")
        print(f"Standardized Alpha: {(n_items / (n_items - 1)) * (1 - (item_variances.sum() / total_variance)):.4f}")
        print(f"Total Scale Mean: {total_mean:.4f}")
        print(f"Total Scale Variance: {total_variance:.4f}")
        print(f"Total Scale Std Dev: {total_sd:.4f}")
        
        # Interpretation
        if alpha >= 0.9:
            print("Interpretation: Excellent reliability")
        elif alpha >= 0.8:
            print("Interpretation: Good reliability")
        elif alpha >= 0.7:
            print("Interpretation: Acceptable reliability")
        elif alpha >= 0.6:
            print("Interpretation: Questionable reliability")
        elif alpha >= 0.5:
            print("Interpretation: Poor reliability")
        else:
            print("Interpretation: Unacceptable reliability")
        
        print("\n--- Item Statistics ---")
        print(f"{'Item':<15} {'Mean':<10} {'Std Dev':<10} {'Variance':<10}")
        print("-" * 45)
        for i in range(n_items):
            print(f"Item {i+1:<10} {item_means[i]:<10.4f} {item_sds[i]:<10.4f} {item_variances[i]:<10.4f}")
        
        print("\n--- Item-Total Statistics ---")
        print(f"{'Item':<15} {'Item-Total Corr':<18} {'Alpha if Deleted':<18} {'Mean if Deleted':<15} {'Var if Deleted':<15}")
        print("-" * 80)
        for i in range(n_items):
            print(f"Item {i+1:<10} {item_total_corr[i]:<18.4f} {alpha_if_deleted[i]:<18.4f} {item_means_if_deleted[i]:<15.4f} {item_variances_if_deleted[i]:<15.4f}")
    
    return alpha, item_total_corr, alpha_if_deleted, item_means, item_sds

# Calculate reliability for each scale
scales = {
    'Service Quality': service_quality_columns,
    'Pricing': pricing_columns,
    'Assurance': assurance_columns,
    'Customer Satisfaction': satisfaction_columns
}

reliability_results = {}
reliability_summary = []

for scale_name, cols in scales.items():
    numeric_cols = [col + '_numeric' for col in cols if col + '_numeric' in df.columns]
    if len(numeric_cols) > 1:
        data = df[numeric_cols].dropna()
        if len(data) > 1:
            print(f"\n{'='*50}")
            print(f"--- {scale_name} Scale ---")
            print(f"{'='*50}")
            alpha, item_corr, alpha_if, item_means, item_sds = cronbach_alpha_complete(data.values, verbose=True)
            reliability_results[scale_name] = {
                'Alpha': alpha,
                'Item_Total_Corr': item_corr,
                'Alpha_If_Deleted': alpha_if,
                'Item_Means': item_means,
                'Item_SDs': item_sds
            }
            reliability_summary.append({
                'Scale': scale_name,
                'Cronbach_Alpha': alpha,
                'Number_of_Items': len(numeric_cols),
                'Interpretation': 'Excellent' if alpha >= 0.9 else 'Good' if alpha >= 0.8 else 'Acceptable' if alpha >= 0.7 else 'Questionable' if alpha >= 0.6 else 'Unacceptable'
            })

# Save reliability results
if reliability_summary:
    reliability_df = pd.DataFrame(reliability_summary)
    reliability_df.to_csv(f'{output_dir}/reliability_analysis.csv', index=False)

# ============================================================================
# 4. CORRELATION ANALYSIS (COMPLETE)
# ============================================================================

print("\n" + "=" * 100)
print("4. CORRELATION ANALYSIS")
print("=" * 100)

def correlation_analysis_complete(df, variables, method='pearson'):
    """Calculate correlation matrix with comprehensive statistics"""
    corr_matrix = pd.DataFrame(index=variables, columns=variables)
    p_matrix = pd.DataFrame(index=variables, columns=variables)
    n_matrix = pd.DataFrame(index=variables, columns=variables)
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if var1 not in df.columns or var2 not in df.columns:
                corr_matrix.iloc[i, j] = np.nan
                p_matrix.iloc[i, j] = np.nan
                n_matrix.iloc[i, j] = 0
                continue
                
            data1 = df[var1].dropna()
            data2 = df[var2].dropna()
            # Align data
            valid_idx = data1.index.intersection(data2.index)
            data1_aligned = data1.loc[valid_idx]
            data2_aligned = data2.loc[valid_idx]
            
            if len(data1_aligned) < 2:
                corr_matrix.iloc[i, j] = np.nan
                p_matrix.iloc[i, j] = np.nan
                n_matrix.iloc[i, j] = 0
                continue
            
            if method == 'pearson':
                corr, p_val = pearsonr(data1_aligned, data2_aligned)
            else:  # spearman
                corr, p_val = spearmanr(data1_aligned, data2_aligned)
            
            corr_matrix.iloc[i, j] = corr
            p_matrix.iloc[i, j] = p_val
            n_matrix.iloc[i, j] = len(valid_idx)
    
    corr_matrix = corr_matrix.astype(float)
    p_matrix = p_matrix.astype(float)
    n_matrix = n_matrix.astype(int)
    
    return corr_matrix, p_matrix, n_matrix

# Pearson Correlation
print("\n--- Pearson Correlation Matrix ---")
variables = ['Avg_Service_Quality', 'Avg_Pricing', 'Avg_Assurance', 'Avg_Satisfaction']

# Filter variables that exist in the dataframe
existing_vars = [var for var in variables if var in df.columns]
if len(existing_vars) >= 2:
    pearson_corr, pearson_p, pearson_n = correlation_analysis_complete(df, existing_vars, method='pearson')
    
    print("\nCorrelation Coefficients:")
    print(pearson_corr.round(3))
    print("\nP-values:")
    print(pearson_p.round(4))
    print("\nSample Sizes:")
    print(pearson_n)
    
    # Spearman Correlation
    print("\n--- Spearman Correlation Matrix ---")
    spearman_corr, spearman_p, spearman_n = correlation_analysis_complete(df, existing_vars, method='spearman')
    
    print("\nCorrelation Coefficients:")
    print(spearman_corr.round(3))
    print("\nP-values:")
    print(spearman_p.round(4))
    print("\nSample Sizes:")
    print(spearman_n)
    
    # Save correlation matrices
    pearson_corr.to_csv(f'{output_dir}/pearson_correlation.csv')
    spearman_corr.to_csv(f'{output_dir}/spearman_correlation.csv')
    pearson_p.to_csv(f'{output_dir}/pearson_pvalues.csv')
    spearman_p.to_csv(f'{output_dir}/spearman_pvalues.csv')

# ============================================================================
# 5. COMPARATIVE TESTS (COMPLETE)
# ============================================================================

print("\n" + "=" * 100)
print("5. COMPARATIVE TESTS")
print("=" * 100)

# 5.1 Independent T-test (Gender differences)
print("\n--- Independent T-test: Gender Differences ---")

def t_test_complete(group1, group2, variable_name):
    """Perform comprehensive independent t-test"""
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    if len(g1) < 2 or len(g2) < 2:
        print(f"\nInsufficient data for {variable_name}")
        return None, None, None, None, None, None
    
    n1, n2 = len(g1), len(g2)
    mean1, mean2 = g1.mean(), g2.mean()
    std1, std2 = g1.std(), g2.std()
    var1, var2 = g1.var(), g2.var()
    
    # Levene's test for homogeneity of variances
    levene_stat, levene_p = levene(g1, g2)
    
    # T-test
    if levene_p < 0.05:
        t_stat, p_val = ttest_ind(g1, g2, equal_var=False)
        equal_var_assumed = False
    else:
        t_stat, p_val = ttest_ind(g1, g2, equal_var=True)
        equal_var_assumed = True
    
    # Calculate effect size (Cohen's d)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    d = (mean1 - mean2) / np.sqrt(pooled_var)
    
    # Hedge's g (corrected for small samples)
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    
    # 95% Confidence Interval for mean difference
    se_diff = np.sqrt(var1/n1 + var2/n2) if not equal_var_assumed else np.sqrt(pooled_var * (1/n1 + 1/n2))
    t_critical = stats.t.ppf(0.975, df=min(n1, n2) - 1)
    ci_lower = (mean1 - mean2) - t_critical * se_diff
    ci_upper = (mean1 - mean2) + t_critical * se_diff
    
    print(f"\n{'='*60}")
    print(f"T-test Results for {variable_name}")
    print(f"{'='*60}")
    print(f"Levene's Test: F = {levene_stat:.4f}, p = {levene_p:.4f}")
    print(f"Equal variances assumed: {equal_var_assumed}")
    print(f"\nGroup Statistics:")
    print(f"  Group 1 (Male):  n = {n1}, Mean = {mean1:.4f}, SD = {std1:.4f}, SEM = {std1/np.sqrt(n1):.4f}")
    print(f"  Group 2 (Female): n = {n2}, Mean = {mean2:.4f}, SD = {std2:.4f}, SEM = {std2/np.sqrt(n2):.4f}")
    print(f"\nT-test Results:")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  df = {n1 + n2 - 2}")
    print(f"  p-value = {p_val:.4f}")
    print(f"  Mean Difference = {mean1 - mean2:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"\nEffect Sizes:")
    print(f"  Cohen's d = {abs(d):.4f}")
    print(f"  Hedge's g = {abs(g):.4f}")
    
    if abs(d) < 0.2:
        print("  → Very small effect")
    elif abs(d) < 0.5:
        print("  → Small effect")
    elif abs(d) < 0.8:
        print("  → Medium effect")
    else:
        print("  → Large effect")
    
    return t_stat, p_val, d, g, ci_lower, ci_upper

t_test_results = {}
t_test_list = []
if 'Gender' in df.columns:
    for var in scale_variables:
        if var in df.columns:
            male_data = df[df['Gender'] == 'Male'][var]
            female_data = df[df['Gender'] == 'Female'][var]
            if len(male_data) > 1 and len(female_data) > 1:
                result = t_test_complete(male_data, female_data, var)
                if result[0] is not None:
                    t_test_results[var] = result
                    t_test_list.append({
                        'Variable': var,
                        't_stat': result[0],
                        'p_value': result[1],
                        'Cohen_d': abs(result[2]),
                        'Hedges_g': abs(result[3])
                    })

if t_test_list:
    t_test_df = pd.DataFrame(t_test_list)
    t_test_df.to_csv(f'{output_dir}/t_test_results.csv', index=False)

# 5.2 Mann-Whitney U Test
print("\n--- Mann-Whitney U Test (Non-parametric) ---")

def mann_whitney_complete(group1, group2, variable_name):
    """Perform comprehensive Mann-Whitney U test"""
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    if len(g1) < 2 or len(g2) < 2:
        print(f"\nInsufficient data for {variable_name}")
        return None, None, None
    
    n1, n2 = len(g1), len(g2)
    median1, median2 = g1.median(), g2.median()
    mean_rank1 = g1.rank().mean() if len(g1) > 0 else 0
    mean_rank2 = g2.rank().mean() if len(g2) > 0 else 0
    
    u_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
    
    # Calculate effect size (r = Z / sqrt(N))
    z_score = stats.norm.ppf(p_val / 2) if p_val < 0.5 else 0
    r_effect = abs(z_score) / np.sqrt(n1 + n2)
    
    print(f"\n{'='*60}")
    print(f"Mann-Whitney U Test for {variable_name}")
    print(f"{'='*60}")
    print(f"Group Statistics:")
    print(f"  Group 1 (Male):  n = {n1}, Median = {median1:.4f}, Mean Rank = {mean_rank1:.4f}")
    print(f"  Group 2 (Female): n = {n2}, Median = {median2:.4f}, Mean Rank = {mean_rank2:.4f}")
    print(f"\nTest Results:")
    print(f"  U-statistic = {u_stat:.4f}")
    print(f"  p-value = {p_val:.4f}")
    print(f"  Effect Size (r) = {r_effect:.4f}")
    
    if p_val < 0.05:
        print("  → Significant difference found at p < 0.05")
    else:
        print("  → No significant difference found")
    
    return u_stat, p_val, r_effect

mw_list = []
if 'Gender' in df.columns:
    for var in scale_variables:
        if var in df.columns:
            male_data = df[df['Gender'] == 'Male'][var]
            female_data = df[df['Gender'] == 'Female'][var]
            if len(male_data) > 1 and len(female_data) > 1:
                result = mann_whitney_complete(male_data, female_data, var)
                if result[0] is not None:
                    mw_list.append({
                        'Variable': var,
                        'U_stat': result[0],
                        'p_value': result[1],
                        'r_effect': result[2]
                    })

if mw_list:
    mw_df = pd.DataFrame(mw_list)
    mw_df.to_csv(f'{output_dir}/mann_whitney_results.csv', index=False)

# 5.3 One-Way ANOVA with Post-hoc
print("\n--- One-Way ANOVA: Age Category Differences ---")

def anova_complete(df, dependent_var, independent_var):
    """Perform comprehensive one-way ANOVA with post-hoc tests"""
    groups = df.groupby(independent_var)[dependent_var]
    group_data = [group.dropna().values for name, group in groups]
    group_names = [name for name, group in groups]
    
    # Filter groups with sufficient data
    valid_groups = [(name, data) for name, data in zip(group_names, group_data) if len(data) > 1]
    if len(valid_groups) < 2:
        print("Insufficient data for ANOVA")
        return None, None, None
    
    valid_names = [g[0] for g in valid_groups]
    valid_data = [g[1] for g in valid_groups]
    
    # Descriptive statistics for each group
    print(f"\n{'='*60}")
    print(f"ANOVA Results for {dependent_var} by {independent_var}")
    print(f"{'='*60}")
    print("\nGroup Descriptive Statistics:")
    print(f"{'Group':<15} {'N':<10} {'Mean':<12} {'SD':<12} {'SEM':<12}")
    print("-" * 60)
    for name, data in zip(valid_names, valid_data):
        n = len(data)
        mean = data.mean()
        sd = data.std()
        sem = sd / np.sqrt(n)
        print(f"{name:<15} {n:<10} {mean:<12.4f} {sd:<12.4f} {sem:<12.4f}")
    
    # Levene's test for homogeneity
    levene_stat, levene_p = levene(*valid_data)
    print(f"\nLevene's Test for Homogeneity of Variances:")
    print(f"  F-statistic = {levene_stat:.4f}, p-value = {levene_p:.4f}")
    if levene_p < 0.05:
        print("  → Variances are significantly different (use Welch's ANOVA)")
    else:
        print("  → Variances are homogeneous")
    
    # Perform ANOVA
    f_stat, p_val = f_oneway(*valid_data)
    
    print(f"\nANOVA Results:")
    print(f"  F-statistic = {f_stat:.4f}")
    print(f"  df = {len(valid_data)-1}, {sum([len(d) for d in valid_data]) - len(valid_data)}")
    print(f"  p-value = {p_val:.4f}")
    
    # Calculate Eta-squared
    grand_mean = np.concatenate(valid_data).mean()
    ss_between = sum([len(d) * (d.mean() - grand_mean)**2 for d in valid_data])
    ss_within = sum([((d - d.mean())**2).sum() for d in valid_data])
    ss_total = ss_between + ss_within
    eta_sq = ss_between / ss_total
    omega_sq = (ss_between - (len(valid_data)-1) * (ss_within / (sum([len(d) for d in valid_data]) - len(valid_data)))) / (ss_total + (ss_within / (sum([len(d) for d in valid_data]) - len(valid_data))))
    
    print(f"\nEffect Sizes:")
    print(f"  Eta-squared = {eta_sq:.4f}")
    print(f"  Omega-squared = {omega_sq:.4f}")
    
    if eta_sq < 0.01:
        print("  → Very small effect")
    elif eta_sq < 0.06:
        print("  → Small effect")
    elif eta_sq < 0.14:
        print("  → Medium effect")
    else:
        print("  → Large effect")
    
    # Post-hoc tests (Tukey HSD) if ANOVA is significant
    if p_val < 0.05 and len(valid_data) > 2:
        print("\n--- Post-hoc Tests (Tukey HSD) ---")
        # Prepare data for Tukey
        all_data = []
        all_labels = []
        for name, data in zip(valid_names, valid_data):
            all_data.extend(data)
            all_labels.extend([name] * len(data))
        
        tukey = pairwise_tukeyhsd(all_data, all_labels, alpha=0.05)
        print(tukey)
        
        # Also perform Games-Howell (for unequal variances)
        print("\n--- Post-hoc Tests (Games-Howell) ---")
        def games_howell(data1, data2):
            n1, n2 = len(data1), len(data2)
            mean1, mean2 = data1.mean(), data2.mean()
            var1, var2 = data1.var(), data2.var()
            
            se = np.sqrt(var1/n1 + var2/n2)
            t_stat = (mean1 - mean2) / se
            df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            return t_stat, df, p_val
        
        gh_results = []
        for i in range(len(valid_data)):
            for j in range(i+1, len(valid_data)):
                t_stat, df, p_val = games_howell(valid_data[i], valid_data[j])
                gh_results.append({
                    'Group1': valid_names[i],
                    'Group2': valid_names[j],
                    't-statistic': t_stat,
                    'df': df,
                    'p-value': p_val,
                    'Significant': p_val < 0.05
                })
        
        gh_df = pd.DataFrame(gh_results)
        print(gh_df.round(4))
    
    return f_stat, p_val, eta_sq

# ANOVA for each dependent variable
anova_results = {}
anova_list = []
for var in scale_variables:
    if var in df.columns:
        result = anova_complete(df, var, 'Age_Category')
        if result[0] is not None:
            anova_results[var] = result
            anova_list.append({
                'Variable': var,
                'F_stat': result[0],
                'p_value': result[1],
                'Eta_sq': result[2]
            })

if anova_list:
    anova_df = pd.DataFrame(anova_list)
    anova_df.to_csv(f'{output_dir}/anova_results.csv', index=False)

# 5.4 Kruskal-Wallis Test (Non-parametric ANOVA)
print("\n--- Kruskal-Wallis Test (Non-parametric ANOVA) ---")

def kruskal_wallis_complete(df, dependent_var, independent_var):
    """Perform comprehensive Kruskal-Wallis test"""
    groups = df.groupby(independent_var)[dependent_var]
    group_data = [group.dropna().values for name, group in groups]
    group_names = [name for name, group in groups]
    
    # Filter groups with sufficient data
    valid_groups = [(name, data) for name, data in zip(group_names, group_data) if len(data) > 1]
    if len(valid_groups) < 2:
        print("Insufficient data for Kruskal-Wallis test")
        return None, None
    
    valid_names = [g[0] for g in valid_groups]
    valid_data = [g[1] for g in valid_groups]
    
    # Descriptive statistics
    print(f"\n{'='*60}")
    print(f"Kruskal-Wallis Test for {dependent_var} by {independent_var}")
    print(f"{'='*60}")
    print("\nGroup Descriptive Statistics:")
    print(f"{'Group':<15} {'N':<10} {'Median':<12} {'IQR':<12}")
    print("-" * 50)
    for name, data in zip(valid_names, valid_data):
        n = len(data)
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        print(f"{name:<15} {n:<10} {median:<12.4f} {iqr:<12.4f}")
    
    # Perform Kruskal-Wallis test
    h_stat, p_val = kruskal(*valid_data)
    
    print(f"\nTest Results:")
    print(f"  H-statistic = {h_stat:.4f}")
    print(f"  df = {len(valid_data)-1}")
    print(f"  p-value = {p_val:.4f}")
    
    # Calculate effect size (epsilon-squared)
    n_total = sum([len(d) for d in valid_data])
    epsilon_sq = h_stat / (n_total - 1)
    print(f"  Effect Size (epsilon-squared) = {epsilon_sq:.4f}")
    
    if p_val < 0.05:
        print("  → Significant difference found at p < 0.05")
    else:
        print("  → No significant difference found")
    
    return h_stat, p_val, epsilon_sq

kw_list = []
for var in scale_variables:
    if var in df.columns:
        result = kruskal_wallis_complete(df, var, 'Age_Category')
        if result[0] is not None:
            kw_list.append({
                'Variable': var,
                'H_stat': result[0],
                'p_value': result[1],
                'Epsilon_sq': result[2]
            })

if kw_list:
    kw_df = pd.DataFrame(kw_list)
    kw_df.to_csv(f'{output_dir}/kruskal_wallis_results.csv', index=False)

# ============================================================================
# 6. REGRESSION ANALYSIS (COMPLETE)
# ============================================================================

print("\n" + "=" * 100)
print("6. REGRESSION ANALYSIS")
print("=" * 100)

# 6.1 Multiple Linear Regression with Diagnostics
print("\n--- Multiple Linear Regression ---")
print("Dependent Variable: Avg_Satisfaction")
print("Independent Variables: Avg_Service_Quality, Avg_Pricing, Avg_Assurance")

# Prepare data
X_columns = ['Avg_Service_Quality', 'Avg_Pricing', 'Avg_Assurance']

# Filter columns that exist
X_columns_existing = [col for col in X_columns if col in df.columns]

if len(X_columns_existing) > 0:
    X = df[X_columns_existing].dropna()
    y = df.loc[X.index, 'Avg_Satisfaction']
    
    if len(X) > 0:
        # Standardize predictors for better interpretation
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
        print(f"Log-Likelihood: {model.llf:.4f}")
        print(f"AIC: {model.aic:.4f}")
        print(f"BIC: {model.bic:.4f}")
        print(f"Number of Observations: {model.nobs}")
        print(f"Degrees of Freedom: {model.df_model}")
        print(f"Residual Degrees of Freedom: {model.df_resid}")
        
        print("\n--- Coefficients ---")
        print("=" * 80)
        print(f"{'Variable':<30} {'Coefficient':<15} {'Std Error':<15} {'t-value':<15} {'p-value':<15} {'95% CI Lower':<15} {'95% CI Upper':<15}")
        print("-" * 120)
        
        conf_int = model.conf_int()
        for var in model.params.index:
            ci_lower = conf_int.loc[var, 0]
            ci_upper = conf_int.loc[var, 1]
            print(f"{var:<30} {model.params[var]:<15.4f} {model.bse[var]:<15.4f} {model.tvalues[var]:<15.4f} {model.pvalues[var]:<15.4f} {ci_lower:<15.4f} {ci_upper:<15.4f}")
        
        # Standardized coefficients (Beta)
        print("\n--- Standardized Coefficients (Beta) ---")
        print("=" * 80)
        y_scaled = (y - y.mean()) / y.std()
        X_scaled_full = (X - X.mean()) / X.std()
        X_scaled_const = sm.add_constant(X_scaled_full)
        model_std = sm.OLS(y_scaled, X_scaled_const).fit()
        
        print(f"{'Variable':<30} {'Beta Coefficient':<20}")
        print("-" * 50)
        for var in model_std.params.index:
            if var != 'const':
                print(f"{var:<30} {model_std.params[var]:<20.4f}")
        
        # VIF for multicollinearity
        print("\n--- Variance Inflation Factor (VIF) ---")
        print("=" * 80)
        vif_data = pd.DataFrame()
        vif_data['Variable'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data['Tolerance'] = 1 / vif_data['VIF']
        print(vif_data)
        vif_data.to_csv(f'{output_dir}/vif_analysis.csv', index=False)
        
        # Interpretation of VIF
        print("\nMulticollinearity Assessment:")
        if (vif_data['VIF'] > 10).any():
            print("  ⚠️ Warning: Some variables have VIF > 10, indicating severe multicollinearity")
        elif (vif_data['VIF'] > 5).any():
            print("  ⚠️ Note: Some variables have VIF > 5, moderate multicollinearity")
        else:
            print("  ✓ No significant multicollinearity detected")
        
        # Durbin-Watson Test for Autocorrelation
        print("\n--- Durbin-Watson Test (Autocorrelation) ---")
        dw_stat = sm.stats.durbin_watson(model.resid)
        print(f"Durbin-Watson statistic: {dw_stat:.4f}")
        if dw_stat < 1.5:
            print("  → Positive autocorrelation detected")
        elif dw_stat > 2.5:
            print("  → Negative autocorrelation detected")
        else:
            print("  → No significant autocorrelation")
        
        # Residual diagnostics
        print("\n--- Residual Diagnostics ---")
        resid = model.resid
        print(f"Mean of Residuals: {resid.mean():.4f}")
        print(f"Std of Residuals: {resid.std():.4f}")
        print(f"Skewness of Residuals: {resid.skew():.4f}")
        print(f"Kurtosis of Residuals: {resid.kurtosis():.4f}")
        
        # Shapiro-Wilk test for normality of residuals
        from scipy.stats import shapiro
        shapiro_stat, shapiro_p = shapiro(resid)
        print(f"\nShapiro-Wilk Test for Normality of Residuals:")
        print(f"  W-statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
        if shapiro_p < 0.05:
            print("  → Residuals significantly deviate from normality")
        else:
            print("  → Residuals follow normal distribution")
        
        # Breusch-Pagan test for heteroscedasticity
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_test = het_breuschpagan(resid, X_with_const)
        print(f"\nBreusch-Pagan Test (Heteroscedasticity):")
        print(f"  LM-statistic = {bp_test[0]:.4f}, p-value = {bp_test[1]:.4f}")
        if bp_test[1] < 0.05:
            print("  → Heteroscedasticity detected")
        else:
            print("  → Homoscedasticity present")
        
        # Save regression results
        regression_summary = pd.DataFrame({
            'Variable': model.params.index,
            'Coefficient': model.params.values,
            'Std_Error': model.bse.values,
            't_value': model.tvalues.values,
            'p_value': model.pvalues.values
        })
        regression_summary.to_csv(f'{output_dir}/regression_coefficients.csv', index=False)

# ============================================================================
# 7. CROSS-TABULATION (COMPLETE)
# ============================================================================

print("\n" + "=" * 100)
print("7. CROSS-TABULATION WITH CHI-SQUARE TESTS")
print("=" * 100)

def cross_tab_complete(df, row_var, col_var, title=""):
    """Perform comprehensive cross-tabulation with chi-square test and measures of association"""
    # Check if both variables exist
    if row_var not in df.columns or col_var not in df.columns:
        print(f"\nWarning: {row_var} or {col_var} not found in dataframe")
        return None, None, None, None
    
    # Create cross-tabulation
    crosstab = pd.crosstab(df[row_var], df[col_var])
    
    # Row and column percentages
    row_percent = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    col_percent = crosstab.div(crosstab.sum(axis=0), axis=1) * 100
    
    # Chi-square test
    chi2, p_val, dof, expected_freq = chi2_contingency(crosstab)
    
    print(f"\n{'='*60}")
    print(f"Cross-Tabulation: {title}")
    print(f"{'='*60}")
    
    print(f"\n--- Cross-Tabulation Table ---")
    print(f"{row_var} vs {col_var}")
    print("-" * 60)
    # Add total row and column
    crosstab_with_totals = crosstab.copy()
    crosstab_with_totals['Total'] = crosstab.sum(axis=1)
    total_row = pd.Series(crosstab.sum(axis=0), name='Total')
    crosstab_with_totals = pd.concat([crosstab_with_totals, total_row.to_frame().T])
    print(crosstab_with_totals)
    crosstab_with_totals.to_csv(f'{output_dir}/crosstab_{row_var}_{col_var}.csv')
    
    print(f"\n--- Row Percentages (%) ---")
    row_percent_with_totals = row_percent.copy()
    row_percent_with_totals['Total'] = row_percent.sum(axis=1)
    total_row_pct = pd.Series([100.0] * len(row_percent.columns), 
                              index=row_percent.columns, name='Total')
    row_percent_with_totals = pd.concat([row_percent_with_totals, total_row_pct.to_frame().T])
    print(row_percent_with_totals.round(2))
    row_percent_with_totals.to_csv(f'{output_dir}/crosstab_{row_var}_{col_var}_row_pct.csv')
    
    print(f"\n--- Column Percentages (%) ---")
    col_percent_with_totals = col_percent.copy()
    col_percent_with_totals['Total'] = col_percent.sum(axis=1)
    total_col_pct = pd.Series([100.0] * len(col_percent.index), 
                              index=col_percent.index, name='Total')
    col_percent_with_totals = pd.concat([col_percent_with_totals, total_col_pct.to_frame().T])
    print(col_percent_with_totals.round(2))
    col_percent_with_totals.to_csv(f'{output_dir}/crosstab_{row_var}_{col_var}_col_pct.csv')
    
    print(f"\n--- Expected Frequencies ---")
    expected_df = pd.DataFrame(expected_freq, index=crosstab.index, columns=crosstab.columns)
    print(expected_df.round(2))
    expected_df.to_csv(f'{output_dir}/crosstab_{row_var}_{col_var}_expected.csv')
    
    print(f"\n--- Chi-Square Test Results ---")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_val:.4f}")
    
    # Measures of association
    print(f"\n--- Measures of Association ---")
    n_total = crosstab.sum().sum()
    phi = np.sqrt(chi2 / n_total)
    print(f"Phi coefficient: {phi:.4f}")
    
    min_dim = min(crosstab.shape) - 1
    cramer_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0
    print(f"Cramer's V: {cramer_v:.4f}")
    
    contingency = np.sqrt(chi2 / (chi2 + n_total))
    print(f"Contingency Coefficient: {contingency:.4f}")
    
    # Interpretation
    if p_val < 0.05:
        print("\n  ✓ Significant association found at p < 0.05")
        if cramer_v < 0.1:
            print("  → Very weak association")
        elif cramer_v < 0.3:
            print("  → Weak association")
        elif cramer_v < 0.5:
            print("  → Moderate association")
        else:
            print("  → Strong association")
    else:
        print("\n  ✗ No significant association found")
    
    # Standardized residuals
    residuals = (crosstab - expected_df) / np.sqrt(expected_df)
    print(f"\n--- Standardized Residuals ---")
    print(residuals.round(2))
    residuals.to_csv(f'{output_dir}/crosstab_{row_var}_{col_var}_residuals.csv')
    
    return chi2, p_val, cramer_v, phi

# Perform cross-tabulations
if 'Gender' in df.columns and 'Mobile_Operator' in df.columns:
    chi2_gender_operator, p_gender_operator, cramer_gender, phi_gender = cross_tab_complete(
        df, 'Gender', 'Mobile_Operator', "Gender vs Mobile Operator")

if 'Income_Category' in df.columns and 'Mobile_Operator' in df.columns:
    chi2_income_operator, p_income_operator, cramer_income, phi_income = cross_tab_complete(
        df, 'Income_Category', 'Mobile_Operator', "Income Category vs Mobile Operator")

if 'Age_Category' in df.columns and 'Mobile_Operator' in df.columns:
    chi2_age_operator, p_age_operator, cramer_age, phi_age = cross_tab_complete(
        df, 'Age_Category', 'Mobile_Operator', "Age Category vs Mobile Operator")

# ============================================================================
# 8. DATA EXPORT (COMPLETE)
# ============================================================================

print("\n" + "=" * 100)
print("8. DATA EXPORT")
print("=" * 100)

# Export descriptive statistics
if not desc_stats_df.empty:
    desc_stats_df.to_csv(f'{output_dir}/descriptive_statistics_complete.csv')
if not likert_stats.empty:
    likert_stats.to_csv(f'{output_dir}/likert_descriptive_statistics.csv')

# Export frequency tables
for name, table in likert_freq_dict.items():
    if not table.empty:
        table.to_csv(f'{output_dir}/frequency_likert_{name}.csv')
for name, table in demo_freq_dict.items():
    if not table.empty:
        table.to_csv(f'{output_dir}/frequency_demo_{name}.csv')

# Export correlation matrices
if len(existing_vars) >= 2:
    pearson_corr.to_csv(f'{output_dir}/pearson_correlation.csv')
    spearman_corr.to_csv(f'{output_dir}/spearman_correlation.csv')
    pearson_p.to_csv(f'{output_dir}/pearson_pvalues.csv')
    spearman_p.to_csv(f'{output_dir}/spearman_pvalues.csv')

# Export reliability results
if reliability_summary:
    reliability_df.to_csv(f'{output_dir}/reliability_analysis.csv', index=False)

# Export regression results
if 'model' in locals():
    regression_summary.to_csv(f'{output_dir}/regression_coefficients.csv', index=False)
    vif_data.to_csv(f'{output_dir}/vif_analysis.csv', index=False)

# Export comparative test results
if 't_test_list' in locals() and t_test_list:
    t_test_df.to_csv(f'{output_dir}/t_test_results.csv', index=False)
if 'mw_list' in locals() and mw_list:
    mw_df.to_csv(f'{output_dir}/mann_whitney_results.csv', index=False)
if 'anova_list' in locals() and anova_list:
    anova_df.to_csv(f'{output_dir}/anova_results.csv', index=False)
if 'kw_list' in locals() and kw_list:
    kw_df.to_csv(f'{output_dir}/kruskal_wallis_results.csv', index=False)

# Export cleaned data
df.to_csv(f'{output_dir}/cleaned_data_with_scores.csv', index=False)

print("\nAll results exported to:", output_dir)
print("Export complete!")

# ============================================================================
# 9. ADDITIONAL TESTS
# ============================================================================

print("\n" + "=" * 100)
print("9. ADDITIONAL STATISTICAL TESTS")
print("=" * 100)

# 9.1 Shapiro-Wilk Test for Normality
print("\n--- Shapiro-Wilk Test for Normality ---")
shapiro_results = []
for var in scale_variables:
    if var in df.columns:
        data = df[var].dropna()
        if len(data) > 3 and len(data) < 5000:
            stat, p_val = stats.shapiro(data)
            shapiro_results.append({
                'Variable': var,
                'W_Statistic': stat,
                'p_value': p_val,
                'Normal': 'Yes' if p_val > 0.05 else 'No'
            })
            print(f"{var}: W = {stat:.4f}, p = {p_val:.4f}")
            if p_val < 0.05:
                print(f"  → Data significantly deviates from normal distribution")
            else:
                print(f"  → Data follows normal distribution")

if shapiro_results:
    shapiro_df = pd.DataFrame(shapiro_results)
    shapiro_df.to_csv(f'{output_dir}/shapiro_wilk_test.csv', index=False)

# 9.2 Kolmogorov-Smirnov Test
print("\n--- Kolmogorov-Smirnov Test for Normality ---")
ks_results = []
for var in scale_variables:
    if var in df.columns:
        data = df[var].dropna()
        if len(data) > 0:
            stat, p_val = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            ks_results.append({
                'Variable': var,
                'KS_Statistic': stat,
                'p_value': p_val,
                'Normal': 'Yes' if p_val > 0.05 else 'No'
            })
            print(f"{var}: KS = {stat:.4f}, p = {p_val:.4f}")

if ks_results:
    ks_df = pd.DataFrame(ks_results)
    ks_df.to_csv(f'{output_dir}/kolmogorov_smirnov_test.csv', index=False)

# 9.3 Bartlett's Test for Homogeneity of Variances
print("\n--- Bartlett's Test for Homogeneity of Variances ---")
groups = []
group_labels = []
if 'Income_Category' in df.columns:
    for category in df['Income_Category'].unique():
        data = df[df['Income_Category'] == category]['Avg_Satisfaction'].dropna()
        if len(data) > 1:
            groups.append(data)
            group_labels.append(category)

if len(groups) > 1:
    bartlett_stat, bartlett_p = stats.bartlett(*groups)
    print(f"Bartlett's Test: Chi-square = {bartlett_stat:.4f}, p = {bartlett_p:.4f}")
    if bartlett_p < 0.05:
        print("  → Variances are significantly different")
    else:
        print("  → Variances are homogeneous")
    
    bartlett_df = pd.DataFrame({
        'Test': ['Bartlett\'s Test'],
        'Chi_Square': [bartlett_stat],
        'p_value': [bartlett_p]
    })
    bartlett_df.to_csv(f'{output_dir}/bartlett_test.csv', index=False)

# ============================================================================
# 10. HYPOTHESIS TESTING SUMMARY (COMPLETE)
# ============================================================================

print("\n" + "=" * 100)
print("10. HYPOTHESIS TESTING SUMMARY")
print("=" * 100)

# Collect p-values from previous tests
p_values = {}

# Get ANOVA p-values
if 'anova_list' in locals() and anova_list:
    for item in anova_list:
        p_values[f'ANOVA_{item["Variable"]}'] = item['p_value']

# Get correlation p-values
for var in ['Avg_Service_Quality', 'Avg_Pricing', 'Avg_Assurance']:
    if var in df.columns and 'Avg_Satisfaction' in df.columns:
        corr_val, p_val = pearsonr(df[var].dropna(), df['Avg_Satisfaction'].dropna())
        p_values[f'Correlation_{var}'] = p_val

# Create comprehensive hypothesis testing summary
hypotheses = [
    {
        'Hypothesis': 'H1: Service quality significantly affects customer satisfaction',
        'Test': 'Pearson Correlation',
        'p-value': p_values.get('Correlation_Avg_Service_Quality', 'N/A'),
        'Result': 'Rejected' if p_values.get('Correlation_Avg_Service_Quality', 1) < 0.05 else 'Not Rejected',
        'Conclusion': 'Significant relationship' if p_values.get('Correlation_Avg_Service_Quality', 1) < 0.05 else 'No significant relationship'
    },
    {
        'Hypothesis': 'H2: Pricing significantly affects customer satisfaction',
        'Test': 'Pearson Correlation',
        'p-value': p_values.get('Correlation_Avg_Pricing', 'N/A'),
        'Result': 'Rejected' if p_values.get('Correlation_Avg_Pricing', 1) < 0.05 else 'Not Rejected',
        'Conclusion': 'Significant relationship' if p_values.get('Correlation_Avg_Pricing', 1) < 0.05 else 'No significant relationship'
    },
    {
        'Hypothesis': 'H3: Assurance significantly affects customer satisfaction',
        'Test': 'Pearson Correlation',
        'p-value': p_values.get('Correlation_Avg_Assurance', 'N/A'),
        'Result': 'Rejected' if p_values.get('Correlation_Avg_Assurance', 1) < 0.05 else 'Not Rejected',
        'Conclusion': 'Significant relationship' if p_values.get('Correlation_Avg_Assurance', 1) < 0.05 else 'No significant relationship'
    },
    {
        'Hypothesis': 'H4: Multiple factors predict customer satisfaction (Regression Model)',
        'Test': 'Multiple Linear Regression',
        'p-value': model.f_pvalue if 'model' in locals() else 'N/A',
        'Result': 'Rejected' if 'model' in locals() and model.f_pvalue < 0.05 else 'Not Rejected',
        'Conclusion': 'Model is significant' if 'model' in locals() and model.f_pvalue < 0.05 else 'Model is not significant'
    },
    {
        'Hypothesis': 'H5: There is a significant association between gender and mobile operator choice',
        'Test': 'Chi-Square Test',
        'p-value': p_gender_operator if 'p_gender_operator' in locals() else 'N/A',
        'Result': 'Rejected' if 'p_gender_operator' in locals() and p_gender_operator < 0.05 else 'Not Rejected',
        'Conclusion': 'Association exists' if 'p_gender_operator' in locals() and p_gender_operator < 0.05 else 'No association'
    },
    {
        'Hypothesis': 'H6: There is a significant association between income category and mobile operator choice',
        'Test': 'Chi-Square Test',
        'p-value': p_income_operator if 'p_income_operator' in locals() else 'N/A',
        'Result': 'Rejected' if 'p_income_operator' in locals() and p_income_operator < 0.05 else 'Not Rejected',
        'Conclusion': 'Association exists' if 'p_income_operator' in locals() and p_income_operator < 0.05 else 'No association'
    },
    {
        'Hypothesis': 'H7: There is a significant association between age category and mobile operator choice',
        'Test': 'Chi-Square Test',
        'p-value': p_age_operator if 'p_age_operator' in locals() else 'N/A',
        'Result': 'Rejected' if 'p_age_operator' in locals() and p_age_operator < 0.05 else 'Not Rejected',
        'Conclusion': 'Association exists' if 'p_age_operator' in locals() and p_age_operator < 0.05 else 'No association'
    }
]

print("\nHypothesis Testing Summary Table:")
print("=" * 120)
print(f"{'Hypothesis':<70} {'Test':<25} {'Result':<15} {'p-value':<12} {'Conclusion':<30}")
print("-" * 120)
for h in hypotheses:
    p_val_str = f"{h['p-value']:.4f}" if isinstance(h['p-value'], float) else h['p-value']
    print(f"{h['Hypothesis']:<70} {h['Test']:<25} {h['Result']:<15} {p_val_str:<12} {h['Conclusion']:<30}")

# Save hypothesis testing summary
hypotheses_df = pd.DataFrame(hypotheses)
hypotheses_df.to_csv(f'{output_dir}/hypothesis_testing_summary.csv', index=False)

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("11. FINAL ANALYSIS SUMMARY")
print("=" * 100)

print("\nSPSS-Style Analysis Complete!")
print("-" * 80)
print(f"Total Respondents: {len(df)}")
print(f"Total Variables Analyzed: {len(df.columns)}")
print(f"Number of Likert Items: {len(likert_columns)}")
print(f"Composite Scales Created: 4 (Service Quality, Pricing, Assurance, Customer Satisfaction)")
print(f"Reliability Tests Performed: {len(scales)} scales")
print(f"Correlation Tests: Pearson and Spearman")
print(f"Comparative Tests: T-test, ANOVA, Mann-Whitney U, Kruskal-Wallis")
print(f"Regression Models: Multiple Linear Regression")
print(f"Cross-tabulations: 3 (Gender, Income, Age vs Mobile Operator)")
print(f"Hypotheses Tested: {len(hypotheses)}")
print("-" * 80)

print("\nAll results have been exported to:")
print(f"  → '{output_dir}/' directory")
print("  → CSV files for all tables")
print("\nAnalysis Complete! 🎉")

print("\n" + "=" * 100)
print("END OF ANALYSIS")
print("=" * 100)