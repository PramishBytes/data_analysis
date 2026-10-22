import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

file_path = '/home/pramiz/Desktop/prasansa/prasansa.csv'
df = pd.read_csv(file_path, encoding='utf-8')
df.columns = df.columns.str.strip()

# Build new column names dynamically
new_names = []
demo_names = ['Timestamp', 'Name', 'Age', 'Gender', 'Marital_Status', 'Education_Level',
              'Year_of_Study', 'Employment_Status', 'Family_Income', 'Internship',
              'Reason_BBA', 'Future_Career_Decided', 'Pokhara_University_Student',
              'Career_Interest', 'Career_Counseling', 'Career_Info_Source']

for i in range(df.shape[1]):
    if i < len(demo_names):
        new_names.append(demo_names[i])
    else:
        orig = df.columns[i]
        if 'Parental' in orig:
            new_names.append(f'Parental_{i}')
        elif 'Family Income' in orig:
            new_names.append(f'FamilyIncome_{i}')
        elif 'Peer' in orig:
            new_names.append(f'Peer_{i}')
        elif 'Job Market' in orig:
            new_names.append(f'JobMarket_{i}')
        elif 'Personality' in orig:
            new_names.append(f'Personality_{i}')
        elif 'Career preference' in orig or 'Career Preference' in orig:
            new_names.append(f'CareerPref_{i}')
        else:
            new_names.append(f'Col_{i}')

df.columns = new_names

# ============================================================================
# LIKERT MAPPING & COMPOSITE SCORES
# ============================================================================

likert_map = {'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4,
              'Strongly Agree': 5, 'Neeutral': 3}

likert_cols = df.columns[16:]
for col in likert_cols:
    df[col + '_num'] = pd.to_numeric(df[col].map(likert_map), errors='coerce')

parental_items = [c for c in likert_cols if c.startswith('Parental_')]
income_items = [c for c in likert_cols if c.startswith('FamilyIncome_')]
peer_items = [c for c in likert_cols if c.startswith('Peer_')]
jobmarket_items = [c for c in likert_cols if c.startswith('JobMarket_')]
personality_items = [c for c in likert_cols if c.startswith('Personality_')]
careerpref_items = [c for c in likert_cols if c.startswith('CareerPref_')]

df['Parental'] = df[[c + '_num' for c in parental_items]].mean(axis=1)
df['Income'] = df[[c + '_num' for c in income_items]].mean(axis=1)
df['Peer'] = df[[c + '_num' for c in peer_items]].mean(axis=1)
df['JobMarket'] = df[[c + '_num' for c in jobmarket_items]].mean(axis=1)
df['Personality'] = df[[c + '_num' for c in personality_items]].mean(axis=1)
df['CareerPref'] = df[[c + '_num' for c in careerpref_items]].mean(axis=1)

output_dir = '/home/pramiz/Desktop/prasansa/SPSS_Tables'
os.makedirs(output_dir, exist_ok=True)

print("=" * 90)
print("SPSS ANALYSIS - CAREER PREFERENCE STUDY")
print("=" * 90)

# ============================================================================
# TABLE 1: MULTIPLE REGRESSION
# ============================================================================

print("\n" + "=" * 90)
print("TABLE 1: MULTIPLE LINEAR REGRESSION")
print("DV: Career Preference | IVs: Parental, Income, Peer, JobMarket, Personality")
print("=" * 90)

X = df[['Parental', 'Income', 'Peer', 'JobMarket', 'Personality']].dropna()
y = df.loc[X.index, 'CareerPref']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_const = sm.add_constant(X_scaled)
model = sm.OLS(y, X_const).fit()

reg_table = pd.DataFrame({
    'Variable': model.params.index,
    'B': np.round(model.params.values, 4),
    'Std.Error': np.round(model.bse.values, 4),
    't': np.round(model.tvalues.values, 4),
    'p': np.round(model.pvalues.values, 4),
    '95% CI Lower': np.round(model.conf_int()[0].values, 4),
    '95% CI Upper': np.round(model.conf_int()[1].values, 4)
})
reg_table['Sig.'] = reg_table['p'].apply(
    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
)

print("\n--- Coefficients ---")
print(reg_table.to_string(index=False))
reg_table.to_csv(f'{output_dir}/Table1_Regression_Coefficients.csv', index=False)

model_summary = pd.DataFrame({
    'Metric': ['R²', 'Adjusted R²', 'F-statistic', 'p-value (F)', 'N', 'Durbin-Watson'],
    'Value': [round(model.rsquared, 4), round(model.rsquared_adj, 4),
              round(model.fvalue, 4), round(model.f_pvalue, 4),
              int(model.nobs), round(sm.stats.durbin_watson(model.resid), 4)]
})
print("\n--- Model Summary ---")
print(model_summary.to_string(index=False))
model_summary.to_csv(f'{output_dir}/Table1_Regression_Model_Summary.csv', index=False)

print("\n--- SUMMARY ---")
print(f"• Model: {'SIGNIFICANT' if model.f_pvalue < 0.05 else 'NOT SIGNIFICANT'} (F = {model.fvalue:.4f}, p = {model.f_pvalue:.4f})")
print(f"• R² = {model.rsquared:.4f} → {model.rsquared*100:.2f}% variance explained")
sig_preds = [v for v in X.columns if model.pvalues[v] < 0.05]
print(f"• Significant predictors: {', '.join(sig_preds) if sig_preds else 'None'}")
print("• Note: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

# ============================================================================
# TABLE 2: CORRELATION MATRIX (FIXED)
# ============================================================================

print("\n\n" + "=" * 90)
print("TABLE 2: PEARSON CORRELATION MATRIX")
print("=" * 90)

corr_vars = ['Parental', 'Income', 'Peer', 'JobMarket', 'Personality', 'CareerPref']

corr_table = pd.DataFrame(index=corr_vars, columns=corr_vars, dtype=float)
pval_table = pd.DataFrame(index=corr_vars, columns=corr_vars, dtype=float)

for i in corr_vars:
    for j in corr_vars:
        valid = df[[i, j]].dropna()
        if len(valid) > 2:
            r, p = pearsonr(valid[i].values, valid[j].values)
            # FIX: force scalar
            r = float(np.asarray(r).flatten()[0])
            p = float(np.asarray(p).flatten()[0])
        else:
            r, p = np.nan, np.nan
        corr_table.loc[i, j] = round(r, 3) if not np.isnan(r) else np.nan
        pval_table.loc[i, j] = round(p, 3) if not np.isnan(p) else np.nan

print("\n--- Correlation Coefficients (r) ---")
print(corr_table.to_string())
corr_table.to_csv(f'{output_dir}/Table2_Correlation_Coefficients.csv')

print("\n--- P-values ---")
print(pval_table.to_string())
pval_table.to_csv(f'{output_dir}/Table2_Correlation_Pvalues.csv')

# Combined with significance markers
combined = corr_table.copy().astype(object)
for i in corr_vars:
    for j in corr_vars:
        if i == j:
            combined.loc[i, j] = "1.000"
        else:
            r = corr_table.loc[i, j]
            p = pval_table.loc[i, j]
            if pd.isna(r) or pd.isna(p):
                combined.loc[i, j] = "-"
            else:
                marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                combined.loc[i, j] = f"{r}{marker}"

print("\n--- Combined (with sig. markers) ---")
print(combined.to_string())
combined.to_csv(f'{output_dir}/Table2_Correlation_Combined.csv')

print("\n--- SUMMARY ---")
max_corr, max_pair = 0, ''
for i in range(len(corr_vars)):
    for j in range(i + 1, len(corr_vars)):
        r = corr_table.iloc[i, j]
        if not pd.isna(r) and abs(r) > max_corr:
            max_corr = abs(r)
            max_pair = f"{corr_vars[i]} ↔ {corr_vars[j]}"
print(f"• Strongest correlation: {max_pair} (r = {max_corr:.3f})")
print("• Correlations with Career Preference:")
for v in corr_vars[:-1]:
    r = corr_table.loc[v, 'CareerPref']
    p = pval_table.loc[v, 'CareerPref']
    if pd.isna(r):
        print(f"   - {v}: N/A")
    else:
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"   - {v}: r = {r}, p = {p} ({sig})")
print("• Note: *** p<0.001, ** p<0.01, * p<0.05")

# ============================================================================
# TABLE 3: CHI-SQUARE TESTS
# ============================================================================

print("\n\n" + "=" * 90)
print("TABLE 3: CHI-SQUARE TESTS OF INDEPENDENCE")
print("=" * 90)

def chi_square_table(df, row_var, col_var, title):
    sub = df[[row_var, col_var]].dropna()
    crosstab = pd.crosstab(sub[row_var], sub[col_var])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    cramer_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))

    print(f"\n--- {title} ---")
    print("\nObserved Frequencies:")
    print(crosstab.to_string())

    print("\nExpected Frequencies:")
    print(pd.DataFrame(expected, index=crosstab.index, columns=crosstab.columns).round(2).to_string())

    stats_table = pd.DataFrame({
        'Statistic': ['Chi-Square', 'df', 'p-value', 'N', "Cramer's V", 'Significant'],
        'Value': [round(chi2, 4), dof, round(p, 4), n, round(cramer_v, 4),
                  'Yes' if p < 0.05 else 'No']
    })
    print("\nChi-Square Statistics:")
    print(stats_table.to_string(index=False))

    return crosstab, stats_table

# Run tests
c1, s1 = chi_square_table(df, 'Gender', 'Career_Interest', 'Gender vs Career Interest')
c1.to_csv(f'{output_dir}/Table3_ChiSquare_Gender_CareerInterest_Crosstab.csv')
s1.to_csv(f'{output_dir}/Table3_ChiSquare_Gender_CareerInterest_Stats.csv', index=False)

c2, s2 = chi_square_table(df, 'Gender', 'Future_Career_Decided', 'Gender vs Future Career Decided')
c2.to_csv(f'{output_dir}/Table3_ChiSquare_Gender_CareerDecided_Crosstab.csv')
s2.to_csv(f'{output_dir}/Table3_ChiSquare_Gender_CareerDecided_Stats.csv', index=False)

c3, s3 = chi_square_table(df, 'Internship', 'Career_Counseling', 'Internship vs Career Counseling')
c3.to_csv(f'{output_dir}/Table3_ChiSquare_Internship_Counseling_Crosstab.csv')
s3.to_csv(f'{output_dir}/Table3_ChiSquare_Internship_Counseling_Stats.csv', index=False)

# Summary
print("\n--- CHI-SQUARE SUMMARY ---")
summary_chi = pd.DataFrame({
    'Test': ['Gender vs Career Interest', 'Gender vs Future Career Decided',
             'Internship vs Career Counseling'],
    'Chi-Square': [s1['Value'][0], s2['Value'][0], s3['Value'][0]],
    'df': [s1['Value'][1], s2['Value'][1], s3['Value'][1]],
    'p-value': [s1['Value'][2], s2['Value'][2], s3['Value'][2]],
    "Cramer's V": [s1['Value'][4], s2['Value'][4], s3['Value'][4]],
    'Significant': [s1['Value'][5], s2['Value'][5], s3['Value'][5]]
})
print(summary_chi.to_string(index=False))
summary_chi.to_csv(f'{output_dir}/Table3_ChiSquare_Summary.csv', index=False)

print("\n--- SUMMARY ---")
for _, row in summary_chi.iterrows():
    sig = "SIGNIFICANT" if row['Significant'] == 'Yes' else "NOT SIGNIFICANT"
    v = row["Cramer's V"]
    strength = "weak" if v < 0.3 else "moderate" if v < 0.5 else "strong"
    print(f"• {row['Test']}: χ² = {row['Chi-Square']}, p = {row['p-value']} → {sig} ({strength})")

print("\n" + "=" * 90)
print(f"All tables saved to: {output_dir}/")
print("=" * 90)