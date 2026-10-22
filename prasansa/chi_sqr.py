import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Read the CSV file
df = pd.read_csv('prasansa.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Clean the data
df_clean = df.copy()
df_clean = df_clean.dropna(how='all')

# ============================================================================
# CHI-SQUARE ANALYSIS FUNCTION
# ============================================================================

def chi_square_test(var1, var2, data):
    """
    Perform Chi-Square test between two categorical variables
    """
    # Create contingency table
    contingency_table = pd.crosstab(data[var1], data[var2], margins=False)
    
    # Remove any rows/columns with all zeros
    contingency_table = contingency_table.loc[(contingency_table.sum(axis=1) > 0), 
                                             (contingency_table.sum(axis=0) > 0)]
    
    if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return None, None, None, None, None, None
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramer's V
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        cramer_v = np.sqrt(phi2 / min(k-1, r-1))
        
        return chi2, p_value, dof, expected, cramer_v, contingency_table
    except:
        return None, None, None, None, None, None

# ============================================================================
# DEFINE CATEGORICAL VARIABLES FOR ANALYSIS
# ============================================================================

# Demographic variables
demographic_vars = ['Age', 'Gender', 'Marital Status', 'Educational Level', 
                    'Year of study', 'Employment Status', 'Family Monthly income']

# Career-related variables
career_vars = ['Have you participated in any internship or industrial visit?',
               'Why did you choose to study BBA?',
               'Have you decided on your future career?',
               'Are you currently a BBA student under Pokhara University?',
               'Which careeer are you most interested in after graduation?',
               'Have you attended any career counseling or career guidance program?',
               'What is your main source of career information?']

# Likert variables (converted to categorical)
likert_vars = [
    'Independent Variables\n1. Parental Education [My parents encourage me to continue my education]',
    'Independent Variables\n1. Parental Education [My parents discuss career opportunities with me.]',
    'Independent Variables\n1. Parental Education [My parents educational background motivates me to achieve higher goals.]',
    'Independent Variables\n1. Parental Education [ I seek career advice from my parents.]',
    '2. Family Income Level [Family financial condition affect my educational opportunities.]',
    '2. Family Income Level [I consider the cost of further education when planning my career.]',
    '2. Family Income Level [I prefer a career that offers long-term financial security.]',
    '2. Family Income Level [Financial responsibilities influene my career plans.]',
    '2. Family Income Level [I may change my career choices due to financial limitations.]',
    '3. Peer Influence [I discuss career plans with my friends.]',
    '3. Peer Influence [My friends introduce me to different careeer success.]',
    '3. Peer Influence [I feel motivated when my friends achieve career success]',
    '3. Peer Influence [I value suggestion from my friends when making career decisions.]',
    '4. Job Market Awareness [I regularly follow information about employment opportunities]',
    '4. Job Market Awareness [I am aware of the skill employers expect from graduates ]',
    '4. Job Market Awareness [I keep myself updated on current labor market trends.]',
    '4. Job Market Awareness [I believe understanding the job market helps me make better career decision]',
    '5. Personality Type [I prefer a career that matches my personal strengths]',
    '5. Personality Type [I enjoy work that reflect my interest.]',
    '5. Personality Type [I believe my personality fits my preferred profession]',
    '5. Personality Type [My values are important when deciding on a career]',
    'Dependent Variable: Career preferencee pattern [I have a clear defined career preference.]',
    'Dependent Variable: Career preferencee pattern [I am confident in pursuing my preferred careeer path.]',
    'Dependent Variable: Career preferencee pattern [I intend to pursue the career that best matches my interests, skill, and values.]',
    'Dependent Variable: Career preferencee pattern [I am committed to achieving my career despite potential challenges.]',
    'Dependent Variable: Career preferencee pattern [My preferred career aligns with my long term professional goals.]'
]

# Clean Likert values
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

for var in likert_vars:
    if var in df_clean.columns:
        df_clean[var] = df_clean[var].apply(clean_likert)

# ============================================================================
# CHI-SQUARE TEST: DEMOGRAPHIC vs CAREER CHOICE
# ============================================================================

print("="*140)
print("CHI-SQUARE ANALYSIS: DEMOGRAPHIC VARIABLES vs CAREER CHOICE")
print("="*140)

chi_square_results = []

# Define career choice variable
career_choice_var = 'Which careeer are you most interested in after graduation?'

# Test demographic variables against career choice
for var in demographic_vars:
    if var in df_clean.columns and career_choice_var in df_clean.columns:
        # Remove rows with missing values
        clean_data = df_clean[[var, career_choice_var]].dropna()
        
        if len(clean_data) > 0:
            chi2, p_value, dof, expected, cramer_v, contingency = chi_square_test(var, career_choice_var, clean_data)
            
            if chi2 is not None:
                result = {
                    'Variable 1': var,
                    'Variable 2': career_choice_var,
                    'Chi-Square': round(chi2, 3),
                    'df': dof,
                    'p-value': round(p_value, 4),
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Cramer\'s V': round(cramer_v, 3),
                    'Effect Size': 'Large' if cramer_v >= 0.3 else 'Medium' if cramer_v >= 0.1 else 'Small'
                }
                chi_square_results.append(result)
                
                print(f"\n{var} vs {career_choice_var}:")
                print(f"  Chi-Square = {chi2:.3f}, df = {dof}, p-value = {p_value:.4f}")
                print(f"  Cramer's V = {cramer_v:.3f} ({result['Effect Size']} effect)")
                print(f"  Significant: {result['Significant']}")
                if contingency is not None:
                    print("\nContingency Table:")
                    print(contingency)
                print("-"*100)

# ============================================================================
# CHI-SQUARE TEST: DEMOGRAPHIC vs CAREER COUNSELING
# ============================================================================

print("\n" + "="*140)
print("CHI-SQUARE ANALYSIS: DEMOGRAPHIC VARIABLES vs CAREER COUNSELING")
print("="*140)

counseling_var = 'Have you attended any career counseling or career guidance program?'

for var in demographic_vars:
    if var in df_clean.columns and counseling_var in df_clean.columns:
        clean_data = df_clean[[var, counseling_var]].dropna()
        
        if len(clean_data) > 0:
            chi2, p_value, dof, expected, cramer_v, contingency = chi_square_test(var, counseling_var, clean_data)
            
            if chi2 is not None:
                result = {
                    'Variable 1': var,
                    'Variable 2': counseling_var,
                    'Chi-Square': round(chi2, 3),
                    'df': dof,
                    'p-value': round(p_value, 4),
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Cramer\'s V': round(cramer_v, 3),
                    'Effect Size': 'Large' if cramer_v >= 0.3 else 'Medium' if cramer_v >= 0.1 else 'Small'
                }
                chi_square_results.append(result)
                
                print(f"\n{var} vs {counseling_var}:")
                print(f"  Chi-Square = {chi2:.3f}, df = {dof}, p-value = {p_value:.4f}")
                print(f"  Cramer's V = {cramer_v:.3f} ({result['Effect Size']} effect)")
                print(f"  Significant: {result['Significant']}")
                if contingency is not None:
                    print("\nContingency Table:")
                    print(contingency)
                print("-"*100)

# ============================================================================
# CHI-SQUARE TEST: DEMOGRAPHIC vs CAREER DECISION
# ============================================================================

print("\n" + "="*140)
print("CHI-SQUARE ANALYSIS: DEMOGRAPHIC VARIABLES vs CAREER DECISION")
print("="*140)

decision_var = 'Have you decided on your future career?'

for var in demographic_vars:
    if var in df_clean.columns and decision_var in df_clean.columns:
        clean_data = df_clean[[var, decision_var]].dropna()
        
        if len(clean_data) > 0:
            chi2, p_value, dof, expected, cramer_v, contingency = chi_square_test(var, decision_var, clean_data)
            
            if chi2 is not None:
                result = {
                    'Variable 1': var,
                    'Variable 2': decision_var,
                    'Chi-Square': round(chi2, 3),
                    'df': dof,
                    'p-value': round(p_value, 4),
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Cramer\'s V': round(cramer_v, 3),
                    'Effect Size': 'Large' if cramer_v >= 0.3 else 'Medium' if cramer_v >= 0.1 else 'Small'
                }
                chi_square_results.append(result)
                
                print(f"\n{var} vs {decision_var}:")
                print(f"  Chi-Square = {chi2:.3f}, df = {dof}, p-value = {p_value:.4f}")
                print(f"  Cramer's V = {cramer_v:.3f} ({result['Effect Size']} effect)")
                print(f"  Significant: {result['Significant']}")
                if contingency is not None:
                    print("\nContingency Table:")
                    print(contingency)
                print("-"*100)

# ============================================================================
# CHI-SQUARE TEST: LIKERT ITEMS vs CAREER CHOICE
# ============================================================================

print("\n" + "="*140)
print("CHI-SQUARE ANALYSIS: LIKERT ITEMS vs CAREER CHOICE")
print("="*140)

# Select key Likert items for analysis
key_likert_items = [
    'Independent Variables\n1. Parental Education [My parents encourage me to continue my education]',
    '3. Peer Influence [I discuss career plans with my friends.]',
    '4. Job Market Awareness [I regularly follow information about employment opportunities]',
    '5. Personality Type [I prefer a career that matches my personal strengths]',
    'Dependent Variable: Career preferencee pattern [I have a clear defined career preference.]',
    'Dependent Variable: Career preferencee pattern [I am confident in pursuing my preferred careeer path.]'
]

for var in key_likert_items:
    if var in df_clean.columns and career_choice_var in df_clean.columns:
        clean_data = df_clean[[var, career_choice_var]].dropna()
        
        # Only include if both variables have categories
        if len(clean_data) > 0 and clean_data[var].nunique() >= 2:
            chi2, p_value, dof, expected, cramer_v, contingency = chi_square_test(var, career_choice_var, clean_data)
            
            if chi2 is not None:
                result = {
                    'Variable 1': var[:60] + '...' if len(var) > 60 else var,
                    'Variable 2': career_choice_var,
                    'Chi-Square': round(chi2, 3),
                    'df': dof,
                    'p-value': round(p_value, 4),
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Cramer\'s V': round(cramer_v, 3),
                    'Effect Size': 'Large' if cramer_v >= 0.3 else 'Medium' if cramer_v >= 0.1 else 'Small'
                }
                chi_square_results.append(result)
                
                print(f"\n{var[:60]}... vs {career_choice_var}:")
                print(f"  Chi-Square = {chi2:.3f}, df = {dof}, p-value = {p_value:.4f}")
                print(f"  Cramer's V = {cramer_v:.3f} ({result['Effect Size']} effect)")
                print(f"  Significant: {result['Significant']}")
                if contingency is not None:
                    print("\nContingency Table:")
                    print(contingency)
                print("-"*100)

# ============================================================================
# CHI-SQUARE TEST: INTERNSHIP vs CAREER CHOICE
# ============================================================================

print("\n" + "="*140)
print("CHI-SQUARE ANALYSIS: INTERNSHIP vs OTHER VARIABLES")
print("="*140)

internship_var = 'Have you participated in any internship or industrial visit?'

# Test internship against career choice
if internship_var in df_clean.columns and career_choice_var in df_clean.columns:
    clean_data = df_clean[[internship_var, career_choice_var]].dropna()
    if len(clean_data) > 0:
        chi2, p_value, dof, expected, cramer_v, contingency = chi_square_test(internship_var, career_choice_var, clean_data)
        
        if chi2 is not None:
            print(f"\n{internship_var} vs {career_choice_var}:")
            print(f"  Chi-Square = {chi2:.3f}, df = {dof}, p-value = {p_value:.4f}")
            print(f"  Cramer's V = {cramer_v:.3f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
            if contingency is not None:
                print("\nContingency Table:")
                print(contingency)
            print("-"*100)

# Test internship against career counseling
if internship_var in df_clean.columns and counseling_var in df_clean.columns:
    clean_data = df_clean[[internship_var, counseling_var]].dropna()
    if len(clean_data) > 0:
        chi2, p_value, dof, expected, cramer_v, contingency = chi_square_test(internship_var, counseling_var, clean_data)
        
        if chi2 is not None:
            print(f"\n{internship_var} vs {counseling_var}:")
            print(f"  Chi-Square = {chi2:.3f}, df = {dof}, p-value = {p_value:.4f}")
            print(f"  Cramer's V = {cramer_v:.3f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
            if contingency is not None:
                print("\nContingency Table:")
                print(contingency)
            print("-"*100)

# ============================================================================
# CHI-SQUARE TEST: REASON FOR BBA vs CAREER CHOICE
# ============================================================================

print("\n" + "="*140)
print("CHI-SQUARE ANALYSIS: REASON FOR BBA vs CAREER CHOICE")
print("="*140)

reason_var = 'Why did you choose to study BBA?'

if reason_var in df_clean.columns and career_choice_var in df_clean.columns:
    clean_data = df_clean[[reason_var, career_choice_var]].dropna()
    if len(clean_data) > 0:
        chi2, p_value, dof, expected, cramer_v, contingency = chi_square_test(reason_var, career_choice_var, clean_data)
        
        if chi2 is not None:
            print(f"\n{reason_var} vs {career_choice_var}:")
            print(f"  Chi-Square = {chi2:.3f}, df = {dof}, p-value = {p_value:.4f}")
            print(f"  Cramer's V = {cramer_v:.3f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
            if contingency is not None:
                print("\nContingency Table:")
                print(contingency)
            print("-"*100)

# ============================================================================
# SUMMARY TABLE OF ALL CHI-SQUARE RESULTS
# ============================================================================

print("\n" + "="*140)
print("SUMMARY TABLE: ALL CHI-SQUARE TEST RESULTS")
print("="*140)

# Create summary dataframe
results_df = pd.DataFrame(chi_square_results)

if len(results_df) > 0:
    # Reorder columns
    column_order = ['Variable 1', 'Variable 2', 'Chi-Square', 'df', 'p-value', 
                    'Significant', "Cramer's V", 'Effect Size']
    results_df = results_df[column_order]
    
    # Sort by significance and p-value
    results_df = results_df.sort_values(['Significant', 'p-value'], ascending=[False, True])
    
    print("\nAll Chi-Square Test Results:")
    print(results_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS:")
    print(f"Total Tests Conducted: {len(results_df)}")
    print(f"Significant Results (p < 0.05): {len(results_df[results_df['Significant'] == 'Yes'])}")
    print(f"Non-Significant Results (p >= 0.05): {len(results_df[results_df['Significant'] == 'No'])}")
    
    if len(results_df[results_df['Significant'] == 'Yes']) > 0:
        print(f"\nSignificant Relationships Found:")
        sig_results = results_df[results_df['Significant'] == 'Yes']
        for _, row in sig_results.iterrows():
            print(f"  • {row['Variable 1']} ↔ {row['Variable 2']}: χ² = {row['Chi-Square']:.3f}, p = {row['p-value']:.4f}, V = {row["Cramer's V"]:.3f}")
    
    # Save to CSV
    results_df.to_csv('chi_square_results.csv', index=False)
    print("\n✓ Chi-square results saved to 'chi_square_results.csv'")
else:
    print("No valid chi-square tests could be performed.")

# ============================================================================
# ADDITIONAL ANALYSIS: MULTIPLE COMPARISONS
# ============================================================================

print("\n" + "="*140)
print("ADDITIONAL INSIGHTS FROM CHI-SQUARE ANALYSIS")
print("="*140)

# Calculate which variables have the strongest associations
if len(results_df) > 0:
    # Top 5 strongest associations (by Cramer's V)
    top_associations = results_df.nlargest(5, "Cramer's V")
    print("\nTop 5 Strongest Associations:")
    for _, row in top_associations.iterrows():
        print(f"  • {row['Variable 1']} ↔ {row['Variable 2']}: V = {row["Cramer's V"]:.3f} ({row['Effect Size']})")
    
    # Significant relationships by variable
    print("\nSignificant Relationships Summary:")
    sig_vars = results_df[results_df['Significant'] == 'Yes']
    if len(sig_vars) > 0:
        for var in demographic_vars:
            var_results = sig_vars[sig_vars['Variable 1'] == var]
            if len(var_results) > 0:
                print(f"\n{var}:")
                for _, row in var_results.iterrows():
                    print(f"  ↔ {row['Variable 2']}: χ² = {row['Chi-Square']:.3f}, p = {row['p-value']:.4f}")

print("\n" + "="*140)
print("CHI-SQUARE ANALYSIS COMPLETE!")
print("="*140)