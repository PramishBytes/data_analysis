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
from sklearn.metrics import confusion_matrix
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
print("SPSS-STYLE DATA ANALYSIS FOR RESTAURANT SATISFACTION STUDY")
print("=" * 100)

# Load the data
df = pd.read_csv('rashmi.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Define Likert scale mapping
likert_map = {
    'Strongly disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5,
    'Strongly Disagree': 1,
    '': np.nan,
    ' ': np.nan
}

# Reverse mapping for display
likert_reverse = {
    1: 'Strongly Disagree',
    2: 'Disagree', 
    3: 'Neutral',
    4: 'Agree',
    5: 'Strongly Agree'
}

# Define column groups
taste_cols = [
    'Independent Variable\n1.Taste [The food served at this restaurant/cafeteria has a good taste.]',
    'Independent Variable\n1.Taste [The food is flavourful and well seasoned.]',
    'Independent Variable\n1.Taste [The taste of the food is consistent every time I visit.]',
    'Independent Variable\n1.Taste [The food meets my expectations in terms of taste.]',
    'Independent Variable\n1.Taste [I enjoy the taste of the food served at this restaurant/cafeteria.]'
]

freshness_cols = [
    '2. Freshness  [The food served at this restaurant/cafeteria is fresh.]',
    '2. Freshness  [The ingredients used in the food appear to be fresh.]',
    '2. Freshness  [The food is served at an appropriate temperature.]',
    '2. Freshness  [The food appears to be freshly prepared.]',
    '2. Freshness  [I am satisfied with the freshness of the food served.]'
]

hygiene_cols = [
    '3.Hygiene [The food is prepared in a hygienic manner.]',
    '3.Hygiene [The dining area is clean and well-maintained.]',
    '3.Hygiene [The plates, glasses and utensils are clean.]',
    '3.Hygiene [The staff maintain good personal hygiene while serving food.]',
    '3.Hygiene [I feel confident about the cleanliness and safety of the food served.]'
]

presentation_cols = [
    '4.Food Presentation [The food is served in an attractive manner.]',
    '4.Food Presentation [The appearance of the food is appealing.]',
    '4.Food Presentation [The food is presented neatly.]',
    '4.Food Presentation [The presentation of the food increases my appetite.]',
    '4.Food Presentation [I am satisfied with the overall presentation of the food.]'
]

portion_cols = [
    '5. Portion size [The portion size of the food is adequate.]',
    '5. Portion size [The quantity of food is worth the price paid.]',
    '5. Portion size [The portion size is consistent on every visit.]',
    '5. Portion size [The amount of food served is sufficient to satisfy my hunger.]',
    '5. Portion size [I am satisfied with the portion size provided by this restaurant/cafeteria.]'
]

dv_cols = [
    'Dependent Variable :\nCustomer Satisfaction  [Overall, I am satisfied with the food quality at this restaurant/cafeteria.]',
    'Dependent Variable :\nCustomer Satisfaction  [This restaurant/cafeteria meets my expectations.]',
    'Dependent Variable :\nCustomer Satisfaction  [I am satisfied with my overall dining experience.]',
    'Dependent Variable :\nCustomer Satisfaction  [I intend to visit this restaurant/cafeteria again in the future.]',
    'Dependent Variable :\nCustomer Satisfaction  [I would recommend this restaurant/cafeteria to my friends and family.]'
]

all_iv_cols = taste_cols + freshness_cols + hygiene_cols + presentation_cols + portion_cols

# Function to convert Likert to numeric
def convert_likert_to_numeric(df, columns, likert_map):
    df_numeric = pd.DataFrame()
    for col in columns:
        if col in df.columns:
            df_numeric[col] = df[col].map(likert_map)
    return df_numeric

# Convert all variables to numeric
iv_numeric = convert_likert_to_numeric(df, all_iv_cols, likert_map)
dv_numeric = convert_likert_to_numeric(df, dv_cols, likert_map)

# Create composite scores
iv_composite = pd.DataFrame({
    'Taste': iv_numeric[taste_cols].mean(axis=1) if all(col in iv_numeric.columns for col in taste_cols) else pd.Series(dtype=float),
    'Freshness': iv_numeric[freshness_cols].mean(axis=1) if all(col in iv_numeric.columns for col in freshness_cols) else pd.Series(dtype=float),
    'Hygiene': iv_numeric[hygiene_cols].mean(axis=1) if all(col in iv_numeric.columns for col in hygiene_cols) else pd.Series(dtype=float),
    'Presentation': iv_numeric[presentation_cols].mean(axis=1) if all(col in iv_numeric.columns for col in presentation_cols) else pd.Series(dtype=float),
    'Portion': iv_numeric[portion_cols].mean(axis=1) if all(col in iv_numeric.columns for col in portion_cols) else pd.Series(dtype=float)
})

dv_composite = dv_numeric.mean(axis=1)

# Create final composite dataframe
composite_df = pd.DataFrame({
    'Taste_Composite': iv_composite['Taste'],
    'Freshness_Composite': iv_composite['Freshness'],
    'Hygiene_Composite': iv_composite['Hygiene'],
    'Presentation_Composite': iv_composite['Presentation'],
    'Portion_Composite': iv_composite['Portion'],
    'Satisfaction_Composite': dv_composite
})

print(f"Data loaded successfully!")
print(f"Total respondents: {len(df)}")
print(f"Number of independent variables: {len(all_iv_cols)}")
print(f"Number of dependent variables: {len(dv_cols)}")

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
        return pd.Series(dtype=float)
    
    # Calculate mode safely
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
        'Sum': data_clean.sum(),
        'Percentile_25': data_clean.quantile(0.25),
        'Percentile_75': data_clean.quantile(0.75),
        'IQR': data_clean.quantile(0.75) - data_clean.quantile(0.25)
    }
    return pd.Series(stats_dict, name=variable_name)

print("\n--- Descriptive Statistics for Composite Scores ---")
desc_stats_df = pd.DataFrame()
for var in composite_df.columns:
    desc_stats_df = pd.concat([desc_stats_df, descriptive_stats_complete(composite_df[var], var)], axis=1)

print(desc_stats_df.round(3))

# Save descriptive statistics
desc_stats_df.to_csv('descriptive_statistics_complete.csv')

# ============================================================================
# 2. FREQUENCY DISTRIBUTIONS - LIKERT SCALE RESPONSES (COMPLETE)
# ============================================================================

print("\n" + "=" * 100)
print("2. FREQUENCY DISTRIBUTIONS - LIKERT SCALE RESPONSES")
print("=" * 100)

def frequency_table_likert(data, variable_name):
    """Create comprehensive frequency table for Likert scale responses"""
    # Get original Likert labels
    rev_map = {1: 'Strongly Disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly Agree'}
    labeled_data = data.map(rev_map)
    
    freq = labeled_data.value_counts().sort_index()
    percent = (freq / len(labeled_data.dropna()) * 100).round(2)
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

print("\n--- TASTE VARIABLES ---")
print("-" * 80)
taste_freq_dict = {}
for col in taste_cols[:3]:  # Show first 3 items
    if col in iv_numeric.columns:
        data = iv_numeric[col]
        if len(data.dropna()) > 0:
            freq_table = frequency_table_likert(data, col)
            short_name = col.split('[')[-1].split(']')[0] if '[' in col else col[:40]
            print(f"\n{short_name}:")
            print(freq_table)
            taste_freq_dict[short_name] = freq_table

print("\n--- FRESHNESS VARIABLES ---")
print("-" * 80)
freshness_freq_dict = {}
for col in freshness_cols[:3]:
    if col in iv_numeric.columns:
        data = iv_numeric[col]
        if len(data.dropna()) > 0:
            freq_table = frequency_table_likert(data, col)
            short_name = col.split('[')[-1].split(']')[0] if '[' in col else col[:40]
            print(f"\n{short_name}:")
            print(freq_table)
            freshness_freq_dict[short_name] = freq_table

print("\n--- HYGIENE VARIABLES ---")
print("-" * 80)
hygiene_freq_dict = {}
for col in hygiene_cols[:3]:
    if col in iv_numeric.columns:
        data = iv_numeric[col]
        if len(data.dropna()) > 0:
            freq_table = frequency_table_likert(data, col)
            short_name = col.split('[')[-1].split(']')[0] if '[' in col else col[:40]
            print(f"\n{short_name}:")
            print(freq_table)
            hygiene_freq_dict[short_name] = freq_table

print("\n--- PRESENTATION VARIABLES ---")
print("-" * 80)
presentation_freq_dict = {}
for col in presentation_cols[:3]:
    if col in iv_numeric.columns:
        data = iv_numeric[col]
        if len(data.dropna()) > 0:
            freq_table = frequency_table_likert(data, col)
            short_name = col.split('[')[-1].split(']')[0] if '[' in col else col[:40]
            print(f"\n{short_name}:")
            print(freq_table)
            presentation_freq_dict[short_name] = freq_table

print("\n--- PORTION SIZE VARIABLES ---")
print("-" * 80)
portion_freq_dict = {}
for col in portion_cols[:3]:
    if col in iv_numeric.columns:
        data = iv_numeric[col]
        if len(data.dropna()) > 0:
            freq_table = frequency_table_likert(data, col)
            short_name = col.split('[')[-1].split(']')[0] if '[' in col else col[:40]
            print(f"\n{short_name}:")
            print(freq_table)
            portion_freq_dict[short_name] = freq_table

print("\n--- SATISFACTION VARIABLES ---")
print("-" * 80)
satisfaction_freq_dict = {}
for col in dv_cols[:3]:
    if col in dv_numeric.columns:
        data = dv_numeric[col]
        if len(data.dropna()) > 0:
            freq_table = frequency_table_likert(data, col)
            short_name = col.split('[')[-1].split(']')[0] if '[' in col else col[:40]
            print(f"\n{short_name}:")
            print(freq_table)
            satisfaction_freq_dict[short_name] = freq_table

# ============================================================================
# 3. FREQUENCY DISTRIBUTIONS - DEMOGRAPHICS
# ============================================================================

print("\n" + "=" * 100)
print("3. FREQUENCY DISTRIBUTIONS - DEMOGRAPHICS")
print("=" * 100)

def frequency_table_demo(data, variable_name):
    """Create comprehensive frequency table for demographic variables"""
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
    
    # Add total row
    total_row = pd.DataFrame({
        'Frequency': [freq.sum()],
        'Percent': [100.0],
        'Valid Percent': [100.0],
        'Cumulative Percent': [100.0]
    }, index=['Total'])
    table = pd.concat([table, total_row])
    return table

demo_cols = ['Age', 'Gender', 'Education level ', 'Occupation', 'Family Income']

for col in demo_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print("-" * 40)
        freq_table = frequency_table_demo(df[col], col)
        print(freq_table)

# ============================================================================
# 4. RELIABILITY ANALYSIS (Cronbach's Alpha)
# ============================================================================

print("\n" + "=" * 100)
print("4. RELIABILITY ANALYSIS - CRONBACH'S ALPHA")
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
        print(f"\n--- Scale Statistics ---")
        print(f"Number of items: {n_items}")
        print(f"Number of respondents: {n_respondents}")
        print(f"Cronbach's Alpha: {alpha:.4f}")
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
    'Taste': taste_cols,
    'Freshness': freshness_cols,
    'Hygiene': hygiene_cols,
    'Presentation': presentation_cols,
    'Portion': portion_cols,
    'Satisfaction': dv_cols
}

reliability_results = {}
for scale_name, cols in scales.items():
    numeric_cols = [col for col in cols if col in iv_numeric.columns or col in dv_numeric.columns]
    if len(numeric_cols) > 1:
        # Get data
        data = pd.DataFrame()
        for col in numeric_cols:
            if col in iv_numeric.columns:
                data[col] = iv_numeric[col]
            elif col in dv_numeric.columns:
                data[col] = dv_numeric[col]
        
        data = data.dropna()
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

# Save reliability results
reliability_df = pd.DataFrame({
    'Scale': list(reliability_results.keys()),
    'Cronbach_Alpha': [v['Alpha'] for v in reliability_results.values()]
})
reliability_df.to_csv('reliability_analysis.csv')

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("5. CORRELATION ANALYSIS")
print("=" * 100)

def correlation_analysis_complete(df, variables, method='pearson'):
    """Calculate correlation matrix with comprehensive statistics"""
    corr_matrix = pd.DataFrame(index=variables, columns=variables)
    p_matrix = pd.DataFrame(index=variables, columns=variables)
    n_matrix = pd.DataFrame(index=variables, columns=variables)
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            data1 = df[var1].dropna()
            data2 = df[var2].dropna()
            valid_idx = data1.index.intersection(data2.index)
            data1_aligned = data1.loc[valid_idx]
            data2_aligned = data2.loc[valid_idx]
            
            if len(valid_idx) > 2:
                if method == 'pearson':
                    corr, p_val = pearsonr(data1_aligned, data2_aligned)
                else:
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
variables = composite_df.columns.tolist()
pearson_corr, pearson_p, pearson_n = correlation_analysis_complete(composite_df, variables, method='pearson')

print("\nCorrelation Coefficients:")
print(pearson_corr.round(3))
print("\nP-values:")
print(pearson_p.round(4))
print("\nSample Sizes:")
print(pearson_n)

# Spearman Correlation
print("\n--- Spearman Correlation Matrix ---")
spearman_corr, spearman_p, spearman_n = correlation_analysis_complete(composite_df, variables, method='spearman')

print("\nCorrelation Coefficients:")
print(spearman_corr.round(3))
print("\nP-values:")
print(spearman_p.round(4))

# Save correlation matrices
pearson_corr.to_csv('pearson_correlation_complete.csv')
spearman_corr.to_csv('spearman_correlation_complete.csv')

# ============================================================================
# 6. COMPARATIVE TESTS
# ============================================================================

print("\n" + "=" * 100)
print("6. COMPARATIVE TESTS")
print("=" * 100)

# 6.1 Independent T-test (Gender differences)
print("\n--- Independent T-test: Gender Differences ---")

def t_test_complete(group1, group2, variable_name):
    """Perform comprehensive independent t-test"""
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    n1, n2 = len(g1), len(g2)
    mean1, mean2 = g1.mean(), g2.mean()
    std1, std2 = g1.std(), g2.std()
    var1, var2 = g1.var(), g2.var()
    
    # Levene's test
    if n1 > 1 and n2 > 1:
        levene_stat, levene_p = levene(g1, g2)
        
        # T-test
        if levene_p < 0.05:
            t_stat, p_val = ttest_ind(g1, g2, equal_var=False)
            equal_var_assumed = False
        else:
            t_stat, p_val = ttest_ind(g1, g2, equal_var=True)
            equal_var_assumed = True
        
        # Effect size (Cohen's d)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        d = (mean1 - mean2) / np.sqrt(pooled_var)
        
        print(f"\n{'='*60}")
        print(f"T-test Results for {variable_name}")
        print(f"{'='*60}")
        print(f"Levene's Test: F = {levene_stat:.4f}, p = {levene_p:.4f}")
        print(f"Equal variances assumed: {equal_var_assumed}")
        print(f"\nGroup Statistics:")
        print(f"  Group 1:  n = {n1}, Mean = {mean1:.4f}, SD = {std1:.4f}")
        print(f"  Group 2: n = {n2}, Mean = {mean2:.4f}, SD = {std2:.4f}")
        print(f"\nT-test Results:")
        print(f"  t-statistic = {t_stat:.4f}")
        print(f"  df = {n1 + n2 - 2}")
        print(f"  p-value = {p_val:.4f}")
        print(f"  Mean Difference = {mean1 - mean2:.4f}")
        print(f"  Cohen's d = {abs(d):.4f}")
        
        return t_stat, p_val, d

# T-test for each composite variable
for var in composite_df.columns:
    male_data = composite_df[var][df['Gender'] == 'Male']
    female_data = composite_df[var][df['Gender'] == 'Female']
    if len(male_data) > 1 and len(female_data) > 1:
        t_test_complete(male_data, female_data, var)

# 6.2 One-Way ANOVA (Income differences)
print("\n--- One-Way ANOVA: Income Differences ---")

def anova_complete(df, dependent_var, independent_var):
    """Perform comprehensive one-way ANOVA"""
    groups = df.groupby(independent_var)[dependent_var]
    group_data = [group.dropna().values for name, group in groups]
    group_names = [name for name, group in groups]
    
    # Filter groups with sufficient data
    valid_groups = [(name, data) for name, data in zip(group_names, group_data) if len(data) > 1]
    if len(valid_groups) < 2:
        print(f"Insufficient data for ANOVA on {dependent_var}")
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
    
    # Perform ANOVA
    f_stat, p_val = f_oneway(*valid_data)
    
    print(f"\nANOVA Results:")
    print(f"  F-statistic = {f_stat:.4f}")
    print(f"  p-value = {p_val:.4f}")
    
    # Calculate Eta-squared
    grand_mean = np.concatenate(valid_data).mean()
    ss_between = sum([len(d) * (d.mean() - grand_mean)**2 for d in valid_data])
    ss_within = sum([((d - d.mean())**2).sum() for d in valid_data])
    eta_sq = ss_between / (ss_between + ss_within)
    
    print(f"  Eta-squared = {eta_sq:.4f}")
    
    return f_stat, p_val, eta_sq

# ANOVA for each composite variable
for var in composite_df.columns:
    anova_complete(composite_df, var, df['Family Income'])

# ============================================================================
# 7. REGRESSION ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("7. MULTIPLE LINEAR REGRESSION")
print("=" * 100)

# Prepare data for regression
regression_data = composite_df.dropna()
X = regression_data[['Taste_Composite', 'Freshness_Composite', 'Hygiene_Composite', 
                     'Presentation_Composite', 'Portion_Composite']]
y = regression_data['Satisfaction_Composite']

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
        vif_data['Tolerance'] = 1 / vif_data['VIF']
        print(vif_data)
    except:
        print("VIF calculation not available for this data")

# ============================================================================
# 8. CROSS-TABULATION WITH CHI-SQUARE TESTS (FIXED)
# ============================================================================

print("\n" + "=" * 100)
print("8. CROSS-TABULATION WITH CHI-SQUARE TESTS")
print("=" * 100)

# First, let's check the actual column names
print("\n--- Available Columns in Dataset ---")
print(df.columns.tolist())

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Now let's verify the cleaned column names
print("\n--- Cleaned Column Names ---")
print(df.columns.tolist())

def cross_tab_complete(df, row_var, col_var, title=""):
    """Perform comprehensive cross-tabulation with chi-square test"""
    # Check if columns exist
    if row_var not in df.columns:
        print(f"\nWarning: Column '{row_var}' not found in dataframe")
        return None, None, None
    if col_var not in df.columns:
        print(f"\nWarning: Column '{col_var}' not found in dataframe")
        return None, None, None
    
    # Create crosstab
    crosstab = pd.crosstab(df[row_var], df[col_var])
    
    # Check if crosstab has enough data
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        print(f"\nWarning: Insufficient data for cross-tabulation of {row_var} vs {col_var}")
        return None, None, None
    
    # Chi-square test
    try:
        chi2, p_val, dof, expected_freq = chi2_contingency(crosstab)
        
        print(f"\n{'='*60}")
        print(f"Cross-Tabulation: {title}")
        print(f"{'='*60}")
        print(f"\n--- Cross-Tabulation Table ---")
        print(f"{row_var} vs {col_var}")
        print("-" * 60)
        crosstab_with_totals = crosstab.copy()
        crosstab_with_totals['Total'] = crosstab.sum(axis=1)
        total_row = pd.Series(crosstab.sum(axis=0), name='Total')
        crosstab_with_totals = pd.concat([crosstab_with_totals, total_row.to_frame().T])
        print(crosstab_with_totals)
        
        # Row percentages
        print(f"\n--- Row Percentages (%) ---")
        row_percent = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        row_percent_with_totals = row_percent.copy()
        row_percent_with_totals['Total'] = row_percent.sum(axis=1)
        print(row_percent_with_totals.round(2))
        
        print(f"\n--- Chi-Square Test Results ---")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"p-value: {p_val:.4f}")
        
        # Cramer's V
        n_total = crosstab.sum().sum()
        min_dim = min(crosstab.shape) - 1
        cramer_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0
        print(f"Cramer's V: {cramer_v:.4f}")
        
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
        residuals = (crosstab - expected_freq) / np.sqrt(expected_freq)
        print(f"\n--- Standardized Residuals ---")
        print(residuals.round(2))
        
        return chi2, p_val, cramer_v
        
    except Exception as e:
        print(f"\nError in chi-square test: {e}")
        return None, None, None

# Cross-tabulations with correct column names
print("\n" + "=" * 60)
print("PERFORMING CROSS-TABULATIONS")
print("=" * 60)

# Define column names with correct spelling and no trailing spaces
cross_tabs = [
    ('Gender', 'Age', "Gender vs Age"),
    ('Occupation', 'Education level', "Occupation vs Education"),
    ('Family Income', 'How often do you visit restaurants or cafeterias around Pokhara University?', "Income vs Visit Frequency"),
    ('Gender', 'Family Income', "Gender vs Income"),
    ('Age', 'How often do you visit restaurants or cafeterias around Pokhara University?', "Age vs Visit Frequency")
]

for row_var, col_var, title in cross_tabs:
    cross_tab_complete(df, row_var, col_var, title)

# ============================================================================
# 9. ADDITIONAL: Likert Scale Response Frequencies (Complete)
# ============================================================================

print("\n" + "=" * 100)
print("9. LIKERT SCALE RESPONSE FREQUENCIES (COMPLETE)")
print("=" * 100)

def get_likert_frequencies_all(df, columns, category_name):
    """Get frequency distribution for all Likert scale items in a category"""
    print(f"\n{category_name} - Frequency Distributions:")
    print("=" * 80)
    
    all_freq = {}
    for col in columns:
        if col in df.columns:
            # Get the numeric version if it exists, otherwise use original
            if col + '_numeric' in df.columns:
                data = df[col + '_numeric']
            else:
                # Convert to numeric if needed
                data = df[col].map(likert_map)
            
            # Get short name for display
            short_name = col.split('[')[-1].split(']')[0] if '[' in col else col[:40]
            print(f"\n{short_name}:")
            print("-" * 40)
            
            # Get frequency
            clean_data = data.dropna()
            if len(clean_data) > 0:
                freq = clean_data.value_counts().sort_index()
                percent = (freq / len(clean_data) * 100).round(2)
                
                # Map numeric values back to labels
                rev_map = {1: 'Strongly Disagree', 2: 'Disagree', 3: 'Neutral', 
                          4: 'Agree', 5: 'Strongly Agree'}
                freq.index = freq.index.map(rev_map)
                percent.index = percent.index.map(rev_map)
                
                freq_table = pd.DataFrame({
                    'Frequency': freq,
                    'Percent': percent
                })
                print(freq_table)
                
                # Add cumulative percent
                cum_percent = percent.cumsum()
                print("\nCumulative Percent:")
                print(cum_percent)
                
                all_freq[short_name] = freq_table
    
    return all_freq

# Get frequencies for all categories
print("\n" + "=" * 100)
print("LIKERT SCALE FREQUENCIES - ALL CATEGORIES")
print("=" * 100)

# Taste
get_likert_frequencies_all(df, taste_cols, "TASTE VARIABLES")

# Freshness
get_likert_frequencies_all(df, freshness_cols, "FRESHNESS VARIABLES")

# Hygiene
get_likert_frequencies_all(df, hygiene_cols, "HYGIENE VARIABLES")

# Presentation
get_likert_frequencies_all(df, presentation_cols, "PRESENTATION VARIABLES")

# Portion
get_likert_frequencies_all(df, portion_cols, "PORTION SIZE VARIABLES")

# Satisfaction
get_likert_frequencies_all(df, dv_cols, "SATISFACTION VARIABLES")

# ============================================================================
# 10. DATA EXPORT
# ============================================================================

print("\n" + "=" * 100)
print("10. DATA EXPORT")
print("=" * 100)

# Create directory for results
import os
if not os.path.exists('spss_analysis_results'):
    os.makedirs('spss_analysis_results')

# Export all results
desc_stats_df.to_csv('spss_analysis_results/descriptive_statistics.csv')
pearson_corr.to_csv('spss_analysis_results/pearson_correlation.csv')
spearman_corr.to_csv('spss_analysis_results/spearman_correlation.csv')
reliability_df.to_csv('spss_analysis_results/reliability_analysis.csv')

# Export regression results
if 'model' in locals():
    regression_summary = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std_Error': model.bse.values,
        't_value': model.tvalues.values,
        'p_value': model.pvalues.values
    })
    regression_summary.to_csv('spss_analysis_results/regression_coefficients.csv')

# Export composite scores
composite_df.to_csv('spss_analysis_results/composite_scores.csv', index=False)

print("\nAll results exported to: spss_analysis_results/")
print("Export complete!")
# ============================================================================
# 11. HYPOTHESIS TESTING SUMMARY (FIXED)
# ============================================================================

print("\n" + "=" * 100)
print("11. HYPOTHESIS TESTING SUMMARY")
print("=" * 100)

# Collect p-values from previous tests
p_values = {}

# Get ANOVA p-values
for var in composite_df.columns:
    groups = []
    for income in df['Family Income'].unique():
        data = composite_df[var][df['Family Income'] == income].dropna()
        if len(data) > 1:
            groups.append(data)
    if len(groups) >= 2:
        try:
            f_stat, p_val = f_oneway(*groups)
            p_values[f'ANOVA_{var}'] = p_val
        except:
            pass

# Get correlation p-values with proper alignment
print("\nCalculating correlations with aligned data...")

for var in ['Taste_Composite', 'Freshness_Composite', 'Hygiene_Composite', 
            'Presentation_Composite', 'Portion_Composite']:
    if var in composite_df.columns:
        # Get both series
        x = composite_df[var]
        y = composite_df['Satisfaction_Composite']
        
        # Find common indices (where both have non-missing values)
        common_idx = x.notna() & y.notna()
        
        # Get aligned data
        x_aligned = x[common_idx]
        y_aligned = y[common_idx]
        
        if len(x_aligned) > 2:
            try:
                corr, p_val = pearsonr(x_aligned, y_aligned)
                p_values[f'Corr_{var}'] = p_val
                print(f"  {var}: r={corr:.4f}, p={p_val:.4f}, n={len(x_aligned)}")
            except Exception as e:
                print(f"  Error calculating correlation for {var}: {e}")
                p_values[f'Corr_{var}'] = 1.0
        else:
            print(f"  Insufficient data for correlation {var} (n={len(x_aligned)})")
            p_values[f'Corr_{var}'] = 1.0

# Create comprehensive hypothesis testing summary
hypotheses = [
    {
        'Hypothesis': 'H1: Taste significantly affects customer satisfaction',
        'Test': 'Pearson Correlation',
        'Result': 'Rejected' if p_values.get('Corr_Taste_Composite', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('Corr_Taste_Composite', 'N/A'),
        'Conclusion': 'Taste correlates with satisfaction' if p_values.get('Corr_Taste_Composite', 1) < 0.05 else 'No significant correlation'
    },
    {
        'Hypothesis': 'H2: Freshness significantly affects customer satisfaction',
        'Test': 'Pearson Correlation',
        'Result': 'Rejected' if p_values.get('Corr_Freshness_Composite', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('Corr_Freshness_Composite', 'N/A'),
        'Conclusion': 'Freshness correlates with satisfaction' if p_values.get('Corr_Freshness_Composite', 1) < 0.05 else 'No significant correlation'
    },
    {
        'Hypothesis': 'H3: Hygiene significantly affects customer satisfaction',
        'Test': 'Pearson Correlation',
        'Result': 'Rejected' if p_values.get('Corr_Hygiene_Composite', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('Corr_Hygiene_Composite', 'N/A'),
        'Conclusion': 'Hygiene correlates with satisfaction' if p_values.get('Corr_Hygiene_Composite', 1) < 0.05 else 'No significant correlation'
    },
    {
        'Hypothesis': 'H4: Food presentation significantly affects customer satisfaction',
        'Test': 'Pearson Correlation',
        'Result': 'Rejected' if p_values.get('Corr_Presentation_Composite', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('Corr_Presentation_Composite', 'N/A'),
        'Conclusion': 'Presentation correlates with satisfaction' if p_values.get('Corr_Presentation_Composite', 1) < 0.05 else 'No significant correlation'
    },
    {
        'Hypothesis': 'H5: Portion size significantly affects customer satisfaction',
        'Test': 'Pearson Correlation',
        'Result': 'Rejected' if p_values.get('Corr_Portion_Composite', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('Corr_Portion_Composite', 'N/A'),
        'Conclusion': 'Portion size correlates with satisfaction' if p_values.get('Corr_Portion_Composite', 1) < 0.05 else 'No significant correlation'
    },
    {
        'Hypothesis': 'H6: There are significant differences in satisfaction across income groups',
        'Test': 'One-Way ANOVA',
        'Result': 'Rejected' if p_values.get('ANOVA_Satisfaction_Composite', 1) < 0.05 else 'Not Rejected',
        'p-value': p_values.get('ANOVA_Satisfaction_Composite', 'N/A'),
        'Conclusion': 'Income differences exist' if p_values.get('ANOVA_Satisfaction_Composite', 1) < 0.05 else 'No income differences'
    },
    {
        'Hypothesis': 'H7: All factors together predict customer satisfaction',
        'Test': 'Multiple Regression',
        'Result': 'Rejected' if 'model' in locals() and hasattr(model, 'f_pvalue') and model.f_pvalue < 0.05 else 'Not Rejected',
        'p-value': model.f_pvalue if 'model' in locals() and hasattr(model, 'f_pvalue') else 'N/A',
        'Conclusion': 'Model is significant' if 'model' in locals() and hasattr(model, 'f_pvalue') and model.f_pvalue < 0.05 else 'Model is not significant'
    }
]

print("\n" + "=" * 100)
print("HYPOTHESIS TESTING SUMMARY TABLE")
print("=" * 100)

print("\n" + "-" * 120)
print(f"{'Hypothesis':<70} {'Test':<25} {'Result':<15} {'p-value':<12} {'Conclusion':<30}")
print("-" * 120)

for h in hypotheses:
    p_val_str = f"{h['p-value']:.4f}" if isinstance(h['p-value'], float) else h['p-value']
    print(f"{h['Hypothesis']:<70} {h['Test']:<25} {h['Result']:<15} {p_val_str:<12} {h['Conclusion']:<30}")

print("-" * 120)

# ============================================================================
# 12. ADDITIONAL: SUMMARY STATISTICS FOR REPORTING
# ============================================================================

print("\n" + "=" * 100)
print("12. SUMMARY STATISTICS FOR REPORTING")
print("=" * 100)

# Calculate key statistics for reporting
print("\n--- Key Descriptive Statistics ---")
print("-" * 60)

for var in composite_df.columns:
    data = composite_df[var].dropna()
    if len(data) > 0:
        print(f"\n{var}:")
        print(f"  Mean: {data.mean():.3f}")
        print(f"  Median: {data.median():.3f}")
        print(f"  Std Dev: {data.std():.3f}")
        print(f"  Min: {data.min():.3f}, Max: {data.max():.3f}")
        print(f"  Skewness: {data.skew():.3f}")
        print(f"  Kurtosis: {data.kurtosis():.3f}")
        print(f"  N: {len(data)}")

# Correlation summary
print("\n--- Correlation with Satisfaction ---")
print("-" * 60)

for var in ['Taste_Composite', 'Freshness_Composite', 'Hygiene_Composite', 
            'Presentation_Composite', 'Portion_Composite']:
    if var in composite_df.columns:
        x = composite_df[var]
        y = composite_df['Satisfaction_Composite']
        common_idx = x.notna() & y.notna()
        x_aligned = x[common_idx]
        y_aligned = y[common_idx]
        
        if len(x_aligned) > 2:
            corr, p_val = pearsonr(x_aligned, y_aligned)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"{var}: r = {corr:.3f}, p = {p_val:.4f} {sig}")
            if p_val < 0.05:
                print(f"  → Significant positive correlation" if corr > 0 else "  → Significant negative correlation")
            else:
                print(f"  → Not significant")

# ============================================================================
# 12. VISUALIZATIONS (FULLY FIXED)
# ============================================================================

print("\n" + "=" * 100)
print("12. GENERATING VISUALIZATIONS")
print("=" * 100)

# Create a comprehensive figure with multiple plots
fig = plt.figure(figsize=(24, 20))

# 1. Boxplots for all composite variables
ax1 = plt.subplot(3, 4, 1)
data_to_plot = [composite_df[col].dropna() for col in composite_df.columns]
bp = ax1.boxplot(data_to_plot, labels=composite_df.columns, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax1.axhline(y=3, color='red', linestyle='--', alpha=0.5)
ax1.set_title('Boxplots of All Variables')
ax1.set_ylabel('Score (1-5)')
ax1.tick_params(axis='x', rotation=45)

# 2. Histogram of Satisfaction
ax2 = plt.subplot(3, 4, 2)
sat_data = composite_df['Satisfaction_Composite'].dropna()
ax2.hist(sat_data, bins=10, edgecolor='black', 
         color='lightgreen', alpha=0.7)
ax2.axvline(sat_data.mean(), color='red', linestyle='--', 
           label=f'Mean: {sat_data.mean():.2f}')
ax2.axvline(sat_data.median(), color='blue', linestyle='--',
           label=f'Median: {sat_data.median():.2f}')
ax2.set_xlabel('Satisfaction Score')
ax2.set_ylabel('Frequency')
ax2.set_title('Overall Satisfaction Distribution')
ax2.legend()

# 3. Correlation Heatmap
ax3 = plt.subplot(3, 4, 3)
sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax3, cbar_kws={'shrink': 0.8})
ax3.set_title('Pearson Correlation Heatmap')

# 4. Scatter plot: Taste vs Satisfaction (FIXED)
ax4 = plt.subplot(3, 4, 4)
# Get aligned data
taste_data = composite_df['Taste_Composite']
sat_data = composite_df['Satisfaction_Composite']
common_idx = taste_data.notna() & sat_data.notna()
taste_clean = taste_data[common_idx]
sat_clean = sat_data[common_idx]

ax4.scatter(taste_clean, sat_clean, alpha=0.6)
if len(taste_clean) > 1:
    z = np.polyfit(taste_clean, sat_clean, 1)
    p = np.poly1d(z)
    ax4.plot(np.sort(taste_clean), p(np.sort(taste_clean)), "r-", alpha=0.8)
    
    # Add correlation
    corr, p_val = pearsonr(taste_clean, sat_clean)
    ax4.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.4f}', 
            transform=ax4.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax4.set_xlabel('Taste Score')
ax4.set_ylabel('Satisfaction Score')
ax4.set_title('Taste vs Satisfaction')

# 5. Bar chart with error bars
ax5 = plt.subplot(3, 4, 5)
category_means = composite_df.mean()
category_stds = composite_df.std()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#FF6B6B']
bars = ax5.bar(category_means.index, category_means, yerr=category_stds, capsize=5, color=colors)
ax5.axhline(y=3, color='red', linestyle='--', alpha=0.5)
ax5.set_ylabel('Mean Score')
ax5.set_title('Mean Scores with Error Bars')
ax5.tick_params(axis='x', rotation=45)

# 6. Gender comparison
ax6 = plt.subplot(3, 4, 6)
valid_gender = df['Gender'].notna()
gender_means = composite_df['Satisfaction_Composite'][valid_gender].groupby(df['Gender'][valid_gender]).mean()
gender_std = composite_df['Satisfaction_Composite'][valid_gender].groupby(df['Gender'][valid_gender]).std()
bars = ax6.bar(gender_means.index, gender_means, yerr=gender_std, capsize=5, 
        color=['pink', 'lightblue'])
ax6.set_ylabel('Mean Satisfaction')
ax6.set_title('Satisfaction by Gender')

# 7. Income comparison
ax7 = plt.subplot(3, 4, 7)
valid_income = df['Family Income'].notna()
income_means = composite_df['Satisfaction_Composite'][valid_income].groupby(df['Family Income'][valid_income]).mean()
income_std = composite_df['Satisfaction_Composite'][valid_income].groupby(df['Family Income'][valid_income]).std()
bars = ax7.bar(income_means.index, income_means, yerr=income_std, capsize=5, 
        color='lightgreen')
ax7.set_ylabel('Mean Satisfaction')
ax7.set_title('Satisfaction by Income')
ax7.tick_params(axis='x', rotation=45)

# 8. Age comparison
ax8 = plt.subplot(3, 4, 8)
valid_age = df['Age'].notna()
age_means = composite_df['Satisfaction_Composite'][valid_age].groupby(df['Age'][valid_age]).mean()
age_std = composite_df['Satisfaction_Composite'][valid_age].groupby(df['Age'][valid_age]).std()
bars = ax8.bar(age_means.index, age_means, yerr=age_std, capsize=5, 
        color='coral')
ax8.set_ylabel('Mean Satisfaction')
ax8.set_title('Satisfaction by Age')
ax8.tick_params(axis='x', rotation=45)

# 9. Visit frequency vs satisfaction
ax9 = plt.subplot(3, 4, 9)
valid_freq = df['How often do you visit restaurants or cafeterias around Pokhara University?'].notna()
freq_means = composite_df['Satisfaction_Composite'][valid_freq].groupby(df['How often do you visit restaurants or cafeterias around Pokhara University?'][valid_freq]).mean()
freq_std = composite_df['Satisfaction_Composite'][valid_freq].groupby(df['How often do you visit restaurants or cafeterias around Pokhara University?'][valid_freq]).std()
bars = ax9.barh(freq_means.index, freq_means, xerr=freq_std)
ax9.set_xlabel('Mean Satisfaction')
ax9.set_title('Satisfaction by Visit Frequency')

# 10. Pie chart - Gender
ax10 = plt.subplot(3, 4, 10)
gender_counts = df['Gender'].value_counts()
ax10.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
         startangle=90, colors=['pink', 'lightblue'])
ax10.set_title('Gender Distribution')

# 11. Pie chart - Education (FIXED - removed trailing space)
ax11 = plt.subplot(3, 4, 11)
# Check if column exists with or without space
edu_col = 'Education level'
if edu_col not in df.columns:
    # Try with trailing space
    edu_col = 'Education level '
edu_counts = df[edu_col].value_counts()
ax11.pie(edu_counts.values, labels=edu_counts.index, autopct='%1.1f%%', 
         startangle=90, colors=['lightcoral', 'lightgreen'])
ax11.set_title('Education Distribution')

# 12. Regression plot - Actual vs Predicted (FIXED - using aligned data)
ax12 = plt.subplot(3, 4, 12)
if 'model' in locals() and 'X_with_const' in locals():
    # Get the data that was actually used in the model
    # The model was fit on X_scaled with constant
    # We need to use the same data for prediction
    try:
        # Use the same data that was used to train the model
        y_actual = y  # This is from the regression data
        y_pred = model.predict(X_with_const)
        
        # Both should have the same length since they came from the same data
        ax12.scatter(y_actual, y_pred, alpha=0.6)
        ax12.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
        ax12.set_xlabel('Actual Satisfaction')
        ax12.set_ylabel('Predicted Satisfaction')
        ax12.set_title(f'Actual vs Predicted (R²={model.rsquared:.3f})')
        
        # Add correlation
        corr, p_val = pearsonr(y_actual, y_pred)
        ax12.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.4f}', 
                transform=ax12.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception as e:
        ax12.text(0.5, 0.5, f'Regression plot error: {str(e)[:50]}...', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax12.transAxes)
        ax12.set_title('Regression Plot Error')
else:
    ax12.text(0.5, 0.5, 'Regression model not available', 
              horizontalalignment='center', verticalalignment='center',
              transform=ax12.transAxes)
    ax12.set_title('Regression Plot Not Available')

plt.tight_layout()
plt.savefig('spss_analysis_results/comprehensive_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualizations saved to: spss_analysis_results/comprehensive_visualizations.png")

# ============================================================================
# 13. ADDITIONAL SCATTER PLOTS (FIXED)
# ============================================================================

print("\n" + "=" * 100)
print("13. ADDITIONAL SCATTER PLOTS")
print("=" * 100)

# Create additional scatter plots for other variables
fig2, axes = plt.subplots(2, 2, figsize=(14, 12))

# Function to create scatter plot with aligned data
def create_scatter_plot(ax, x_var, y_var, x_label, y_label, title):
    """Create scatter plot with aligned data"""
    x_data = composite_df[x_var]
    y_data = composite_df[y_var]
    common_idx = x_data.notna() & y_data.notna()
    x_clean = x_data[common_idx]
    y_clean = y_data[common_idx]
    
    if len(x_clean) > 0:
        ax.scatter(x_clean, y_clean, alpha=0.6)
        if len(x_clean) > 1:
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            ax.plot(np.sort(x_clean), p(np.sort(x_clean)), "r-", alpha=0.8)
            
            # Calculate correlation
            corr, p_val = pearsonr(x_clean, y_clean)
            ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.4f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

# Create scatter plots
create_scatter_plot(axes[0, 0], 'Freshness_Composite', 'Satisfaction_Composite',
                   'Freshness Score', 'Satisfaction Score', 'Freshness vs Satisfaction')

create_scatter_plot(axes[0, 1], 'Hygiene_Composite', 'Satisfaction_Composite',
                   'Hygiene Score', 'Satisfaction Score', 'Hygiene vs Satisfaction')

create_scatter_plot(axes[1, 0], 'Presentation_Composite', 'Satisfaction_Composite',
                   'Presentation Score', 'Satisfaction Score', 'Presentation vs Satisfaction')

create_scatter_plot(axes[1, 1], 'Portion_Composite', 'Satisfaction_Composite',
                   'Portion Score', 'Satisfaction Score', 'Portion vs Satisfaction')

plt.tight_layout()
plt.savefig('spss_analysis_results/additional_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("Additional scatter plots saved to: spss_analysis_results/additional_scatter_plots.png")

# ============================================================================
# 14. FINAL COMPLETE SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("14. FINAL COMPLETE SUMMARY")
print("=" * 100)

print("""
The following SPSS-style analyses have been completed:

1. DESCRIPTIVE STATISTICS:
   ✓ Mean, Median, Mode
   ✓ Std. Deviation, Variance
   ✓ Min, Max, Range
   ✓ Skewness, Kurtosis
   ✓ Std. Error Mean
   ✓ Percentiles (25th, 75th)
   ✓ Interquartile Range (IQR)

2. FREQUENCY DISTRIBUTIONS:
   ✓ Likert scale frequencies (Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree)
   ✓ Percentages and cumulative percentages
   ✓ Demographic frequencies
   ✓ Frequency tables for all variables

3. RELIABILITY ANALYSIS:
   ✓ Cronbach's Alpha for each scale
   ✓ Item-total correlations
   ✓ Alpha if item deleted
   ✓ Reliability interpretation (Excellent/Good/Acceptable/Questionable/Poor)

4. CORRELATION ANALYSIS:
   ✓ Pearson Correlation Matrix with p-values
   ✓ Spearman Correlation Matrix with p-values
   ✓ Sample sizes for each correlation

5. COMPARATIVE TESTS:
   ✓ Independent T-test (Gender differences)
   ✓ One-Way ANOVA (Income differences)
   ✓ Levene's test for homogeneity
   ✓ Effect sizes (Cohen's d, Eta-squared)

6. REGRESSION ANALYSIS:
   ✓ Multiple Linear Regression
   ✓ R-squared and Adjusted R-squared
   ✓ Coefficients with standard errors
   ✓ t-values and p-values
   ✓ VIF for multicollinearity
   ✓ Model F-test

7. CROSS-TABULATION:
   ✓ Cross-tabulation with Chi-square tests
   ✓ Cramer's V for association strength
   ✓ Expected frequencies
   ✓ Standardized residuals

8. HYPOTHESIS TESTING:
   ✓ 7 hypotheses tested
   ✓ Results with p-values
   ✓ Conclusions for each hypothesis

9. VISUALIZATIONS:
   ✓ Boxplots for all variables
   ✓ Histogram of satisfaction scores
   ✓ Correlation heatmap
   ✓ Scatter plots with trend lines
   ✓ Bar charts with error bars
   ✓ Demographic comparisons
   ✓ Pie charts for demographics
   ✓ Regression plots

10. DATA EXPORT:
    ✓ CSV files for all results
    ✓ Ready for interpretation and reporting
""")

print("\n" + "=" * 100)
print("SPSS ANALYSIS COMPLETE! 🎉")
print("=" * 100)
print("\nAll outputs are available in the 'spss_analysis_results' directory")
print("Files generated:")
print("  - descriptive_statistics.csv")
print("  - pearson_correlation.csv")
print("  - spearman_correlation.csv")
print("  - reliability_analysis.csv")
print("  - regression_coefficients.csv")
print("  - composite_scores.csv")
print("  - comprehensive_visualizations.png")
print("  - additional_scatter_plots.png")
print("\nThank you for using SPSS-Style Data Analysis!")
# ============================================================================
# 13. COMPLETE SPSS OUTPUT SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("13. COMPLETE SPSS OUTPUT SUMMARY")
print("=" * 100)

print("""
The following SPSS-style analyses have been completed:

1. DESCRIPTIVE STATISTICS:
   ✓ Mean, Median, Mode
   ✓ Std. Deviation, Variance
   ✓ Min, Max, Range
   ✓ Skewness, Kurtosis
   ✓ Std. Error Mean
   ✓ Percentiles (25th, 75th)
   ✓ Interquartile Range (IQR)

2. FREQUENCY DISTRIBUTIONS:
   ✓ Likert scale frequencies (Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree)
   ✓ Percentages and cumulative percentages
   ✓ Demographic frequencies
   ✓ Frequency tables for all variables

3. RELIABILITY ANALYSIS:
   ✓ Cronbach's Alpha for each scale
   ✓ Item-total correlations
   ✓ Alpha if item deleted
   ✓ Reliability interpretation (Excellent/Good/Acceptable/Questionable/Poor)

4. CORRELATION ANALYSIS:
   ✓ Pearson Correlation Matrix with p-values
   ✓ Spearman Correlation Matrix with p-values
   ✓ Sample sizes for each correlation

5. COMPARATIVE TESTS:
   ✓ Independent T-test (Gender differences)
   ✓ One-Way ANOVA (Income differences)
   ✓ Levene's test for homogeneity
   ✓ Effect sizes (Cohen's d, Eta-squared)

6. REGRESSION ANALYSIS:
   ✓ Multiple Linear Regression
   ✓ R-squared and Adjusted R-squared
   ✓ Coefficients with standard errors
   ✓ t-values and p-values
   ✓ VIF for multicollinearity
   ✓ Model F-test

7. CROSS-TABULATION:
   ✓ Cross-tabulation with Chi-square tests
   ✓ Cramer's V for association strength
   ✓ Expected frequencies

8. HYPOTHESIS TESTING:
   ✓ 7 hypotheses tested
   ✓ Results with p-values
   ✓ Conclusions for each hypothesis

9. VISUALIZATIONS:
   ✓ Boxplots for all variables
   ✓ Histogram of satisfaction scores
   ✓ Correlation heatmap
   ✓ Scatter plots
   ✓ Bar charts with error bars
   ✓ Demographic comparisons
   ✓ Pie charts for demographics
   ✓ Regression plots

10. DATA EXPORT:
    ✓ CSV files for all results
    ✓ Ready for interpretation and reporting
""")

print("\n" + "=" * 100)
print("SPSS ANALYSIS COMPLETE! 🎉")
print("=" * 100)
print("\nAll outputs are available in the 'spss_analysis_results' directory")
print("Thank you for using SPSS-Style Data Analysis!")