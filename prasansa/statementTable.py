import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Read the CSV file
df = pd.read_csv('prasansa.csv')

# Clean column names (remove extra spaces and special characters)
df.columns = df.columns.str.strip()

# Define the Likert scale mapping for the independent and dependent variables
likert_mapping = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

# Handle misspelled 'Neutral' variations
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

# Function to convert Likert scale responses to numeric values
def convert_likert_to_numeric(series):
    return series.map(likert_mapping)

# Define the variables to analyze
demographic_vars = ['Age', 'Gender', 'Marital Status', 'Educational Level', 'Year of study', 
                   'Employment Status', 'Family Monthly income']

career_vars = ['Have you participated in any internship or industrial visit?', 
               'Why did you choose to study BBA?',
               'Have you decided on your future career?', 
               'Are you currently a BBA student under Pokhara University?',
               'Which careeer are you most interested in after graduation?',
               'Have you attended any career counseling or career guidance program?',
               'What is your main source of career information?']

# Independent variables - Parental Education
parental_education_vars = [
    'Independent Variables\n1. Parental Education [My parents encourage me to continue my education]',
    'Independent Variables\n1. Parental Education [My parents discuss career opportunities with me.]',
    'Independent Variables\n1. Parental Education [My parents educational background motivates me to achieve higher goals.]',
    'Independent Variables\n1. Parental Education [ I seek career advice from my parents.]'
]

# Independent variables - Family Income Level
family_income_vars = [
    '2. Family Income Level [Family financial condition affect my educational opportunities.]',
    '2. Family Income Level [I consider the cost of further education when planning my career.]',
    '2. Family Income Level [I prefer a career that offers long-term financial security.]',
    '2. Family Income Level [Financial responsibilities influene my career plans.]',
    '2. Family Income Level [I may change my career choices due to financial limitations.]'
]

# Independent variables - Peer Influence
peer_influence_vars = [
    '3. Peer Influence [I discuss career plans with my friends.]',
    '3. Peer Influence [My friends introduce me to different careeer success.]',
    '3. Peer Influence [I feel motivated when my friends achieve career success]',
    '3. Peer Influence [I value suggestion from my friends when making career decisions.]'
]

# Independent variables - Job Market Awareness
job_market_vars = [
    '4. Job Market Awareness [I regularly follow information about employment opportunities]',
    '4. Job Market Awareness [I am aware of the skill employers expect from graduates ]',
    '4. Job Market Awareness [I keep myself updated on current labor market trends.]',
    '4. Job Market Awareness [I believe understanding the job market helps me make better career decision]'
]

# Independent variables - Personality Type
personality_vars = [
    '5. Personality Type [I prefer a career that matches my personal strengths]',
    '5. Personality Type [I enjoy work that reflect my interest.]',
    '5. Personality Type [I believe my personality fits my preferred profession]',
    '5. Personality Type [My values are important when deciding on a career]'
]

# Dependent variables
dependent_vars = [
    'Dependent Variable: Career preferencee pattern [I have a clear defined career preference.]',
    'Dependent Variable: Career preferencee pattern [I am confident in pursuing my preferred careeer path.]',
    'Dependent Variable: Career preferencee pattern [I intend to pursue the career that best matches my interests, skill, and values.]',
    'Dependent Variable: Career preferencee pattern [I am committed to achieving my career despite potential challenges.]',
    'Dependent Variable: Career preferencee pattern [My preferred career aligns with my long term professional goals.]'
]

all_likert_vars = parental_education_vars + family_income_vars + peer_influence_vars + job_market_vars + personality_vars + dependent_vars

# Function to clean and prepare the data
def prepare_data(df):
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Drop rows with all NaN values (like the duplicate header rows)
    df_clean = df_clean.dropna(how='all')
    
    # For each likert variable, clean the values
    for var in all_likert_vars:
        if var in df_clean.columns:
            df_clean[var] = df_clean[var].apply(clean_likert)
    
    return df_clean

# Main analysis function
def analyze_survey_data(df):
    df_clean = prepare_data(df)
    
    results = {}
    
    # 1. DEMOGRAPHIC FREQUENCY TABLES
    print("\n" + "="*80)
    print("DEMOGRAPHIC FREQUENCY DISTRIBUTIONS")
    print("="*80)
    
    demographic_results = {}
    for var in demographic_vars:
        if var in df_clean.columns:
            freq = df_clean[var].value_counts(dropna=False)
            percent = df_clean[var].value_counts(normalize=True, dropna=False) * 100
            demo_table = pd.DataFrame({
                'Frequency': freq,
                'Percentage': percent.round(2)
            })
            demographic_results[var] = demo_table
            
            print(f"\n{var}:")
            print(demo_table)
            print("-"*50)
    
    # 2. CAREER CHOICE FREQUENCY TABLES
    print("\n" + "="*80)
    print("CAREER CHOICE FREQUENCY DISTRIBUTIONS")
    print("="*80)
    
    career_results = {}
    for var in career_vars:
        if var in df_clean.columns:
            freq = df_clean[var].value_counts(dropna=False)
            percent = df_clean[var].value_counts(normalize=True, dropna=False) * 100
            career_table = pd.DataFrame({
                'Frequency': freq,
                'Percentage': percent.round(2)
            })
            career_results[var] = career_table
            
            print(f"\n{var}:")
            print(career_table)
            print("-"*50)
    
    # 3. LIKERT SCALE DESCRIPTIVE STATISTICS
    print("\n" + "="*80)
    print("LIKERT SCALE DESCRIPTIVE STATISTICS")
    print("="*80)
    
    # Prepare numeric data for Likert variables
    likert_data = {}
    for var in all_likert_vars:
        if var in df_clean.columns:
            # Convert to numeric
            numeric_vals = df_clean[var].map(likert_mapping)
            likert_data[var] = numeric_vals
    
    likert_df = pd.DataFrame(likert_data)
    
    # Calculate descriptive statistics
    descriptive_stats = []
    for var in all_likert_vars:
        if var in likert_df.columns:
            series = likert_df[var].dropna()
            if len(series) > 0:
                stats_dict = {
                    'Variable': var[:50] + '...' if len(var) > 50 else var,
                    'N': len(series),
                    'Mean': series.mean(),
                    'Median': series.median(),
                    'Mode': series.mode().iloc[0] if len(series.mode()) > 0 else np.nan,
                    'Std Dev': series.std(),
                    'Min': series.min(),
                    'Max': series.max(),
                    'Skewness': series.skew(),
                    'Kurtosis': series.kurtosis()
                }
                descriptive_stats.append(stats_dict)
    
    stats_df = pd.DataFrame(descriptive_stats)
    print("\nDescriptive Statistics for Likert Scale Items:")
    print(stats_df.to_string(index=False))
    
    # 4. FREQUENCY DISTRIBUTIONS FOR LIKERT ITEMS
    print("\n" + "="*80)
    print("LIKERT SCALE FREQUENCY DISTRIBUTIONS")
    print("="*80)
    
    likert_freq_results = {}
    for var in all_likert_vars:
        if var in df_clean.columns:
            freq = df_clean[var].value_counts(dropna=False)
            percent = df_clean[var].value_counts(normalize=True, dropna=False) * 100
            freq_table = pd.DataFrame({
                'Frequency': freq,
                'Percentage': percent.round(2)
            })
            likert_freq_results[var] = freq_table
            
            print(f"\n{var[:60]}...")
            print(freq_table)
            print("-"*50)
    
    # 5. GROUPED STATISTICS BY DEMOGRAPHIC VARIABLES
    print("\n" + "="*80)
    print("GROUPED ANALYSIS - MEAN SCORES BY DEMOGRAPHIC VARIABLES")
    print("="*80)
    
    grouped_results = {}
    
    # Create composite scores for each factor
    factor_groups = {
        'Parental Education': parental_education_vars,
        'Family Income': family_income_vars,
        'Peer Influence': peer_influence_vars,
        'Job Market Awareness': job_market_vars,
        'Personality Type': personality_vars,
        'Career Preference': dependent_vars
    }
    
    # Calculate composite scores
    for factor_name, var_list in factor_groups.items():
        valid_vars = [v for v in var_list if v in likert_df.columns]
        if valid_vars:
            # Calculate mean score for each respondent
            df_clean[f'{factor_name}_Composite'] = likert_df[valid_vars].mean(axis=1)
    
    # Group by demographic variables
    demo_vars_for_grouping = ['Age', 'Gender', 'Marital Status', 'Educational Level', 'Year of study', 'Employment Status']
    
    for demo_var in demo_vars_for_grouping:
        if demo_var in df_clean.columns:
            print(f"\n--- Grouped by: {demo_var} ---")
            for factor_name in factor_groups.keys():
                composite_col = f'{factor_name}_Composite'
                if composite_col in df_clean.columns:
                    grouped = df_clean.groupby(demo_var)[composite_col].agg(['mean', 'median', 'std', 'count'])
                    print(f"\n{factor_name} scores by {demo_var}:")
                    print(grouped.round(2))
                    print("-"*40)
    
    # 6. CORRELATION ANALYSIS
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Calculate composite scores for correlation
    composite_scores = pd.DataFrame()
    for factor_name, var_list in factor_groups.items():
        valid_vars = [v for v in var_list if v in likert_df.columns]
        if valid_vars:
            composite_scores[factor_name] = likert_df[valid_vars].mean(axis=1)
    
    # Correlation matrix
    corr_matrix = composite_scores.corr()
    print("\nCorrelation Matrix of Factor Scores:")
    print(corr_matrix.round(3))
    
    # 7. RELIABILITY ANALYSIS (Cronbach's Alpha)
    print("\n" + "="*80)
    print("RELIABILITY ANALYSIS (Cronbach's Alpha)")
    print("="*80)
    
    def cronbach_alpha(data):
        """Calculate Cronbach's alpha for a set of items"""
        if data.shape[1] < 2:
            return np.nan
        items = data.shape[1]
        variances = data.var(axis=0, ddof=1)
        total_variance = data.sum(axis=1).var(ddof=1)
        alpha = (items / (items - 1)) * (1 - (variances.sum() / total_variance))
        return alpha
    
    for factor_name, var_list in factor_groups.items():
        valid_vars = [v for v in var_list if v in likert_df.columns]
        if len(valid_vars) >= 2:
            item_data = likert_df[valid_vars].dropna()
            if len(item_data) > 0:
                alpha = cronbach_alpha(item_data)
                print(f"\n{factor_name}: Cronbach's Alpha = {alpha:.3f}")
                print(f"Number of items: {len(valid_vars)}, Sample size: {len(item_data)}")
    
    # Return all results for further use
    return {
        'demographic': demographic_results,
        'career': career_results,
        'likert_stats': stats_df,
        'likert_freq': likert_freq_results,
        'grouped': grouped_results,
        'correlation': corr_matrix
    }

# Run the analysis
results = analyze_survey_data(df)

# Export results to CSV files
print("\n" + "="*80)
print("EXPORTING RESULTS TO CSV FILES")
print("="*80)

# Export descriptive statistics
results['likert_stats'].to_csv('likert_descriptive_stats.csv', index=False)
print("✓ Likert descriptive statistics saved to 'likert_descriptive_stats.csv'")

# Export correlation matrix
results['correlation'].to_csv('correlation_matrix.csv')
print("✓ Correlation matrix saved to 'correlation_matrix.csv'")

# Export demographic frequencies
for var, table in results['demographic'].items():
    filename = f"demographic_{var.replace(' ', '_')}.csv"
    table.to_csv(filename)
    print(f"✓ {var} frequencies saved to '{filename}'")

# Export career frequencies
for var, table in results['career'].items():
    filename = f"career_{var.replace(' ', '_')[:30]}.csv"
    table.to_csv(filename)
    print(f"✓ {var[:30]}... frequencies saved to '{filename}'")

# Export Likert frequency distributions
for i, (var, table) in enumerate(results['likert_freq'].items()):
    if i < 10:  # Limit to first 10 to avoid too many files
        filename = f"likert_freq_{i+1}.csv"
        table.to_csv(filename)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)