import pandas as pd
import numpy as np

# Your data
data = [
    [1, "Parental Encouragement for Education", 162, 3.926, 4.000, 5.000, 1.101, 1.000, 5.000, -0.814, -0.237, 38.272, 32.099, 16.049, 11.111, 2.469],
    [2, "Parents Discuss Career Opportunities", 162, 3.741, 4.000, 4.000, 0.916, 1.000, 5.000, -0.639, 0.109, 18.519, 49.383, 20.988, 9.877, 1.235],
    [3, "Parental Education as Motivation", 162, 3.772, 4.000, 4.000, 1.017, 1.000, 5.000, -0.639, -0.088, 25.926, 39.506, 22.840, 9.259, 2.469],
    [4, "Seek Career Advice from Parents", 162, 3.704, 4.000, 4.000, 1.002, 1.000, 5.000, -0.462, -0.363, 23.457, 37.654, 26.543, 10.494, 1.852],
    [5, "Family Finances Affect Education", 162, 3.321, 3.000, 4.000, 1.113, 1.000, 5.000, -0.061, -0.986, 16.667, 29.630, 25.926, 24.691, 3.086],
    [6, "Consider Cost in Career Planning", 162, 3.821, 4.000, 4.000, 0.912, 1.000, 5.000, -0.682, 0.474, 22.840, 45.679, 24.074, 5.556, 1.852],
    [7, "Prefer Long-term Financial Security", 162, 4.154, 4.000, 5.000, 0.916, 1.000, 5.000, -0.950, 0.560, 43.827, 33.333, 18.519, 3.086, 1.235],
    [8, "Financial Responsibilities Affect Career", 162, 4.043, 4.000, 4.000, 0.961, 1.000, 5.000, -1.065, 1.065, 36.420, 40.741, 16.049, 4.321, 2.469],
    [9, "May Change Career Due to Finances", 162, 3.710, 4.000, 4.000, 1.067, 1.000, 5.000, -0.577, -0.204, 26.543, 33.951, 27.160, 8.642, 3.704],
    [10, "Discuss Career with Friends", 162, 4.012, 4.000, 4.000, 0.939, 1.000, 5.000, -0.890, 0.655, 34.568, 40.123, 19.136, 4.321, 1.852],
    [11, "Friends Introduce Career Success", 162, 3.710, 4.000, 4.000, 0.745, 1.000, 5.000, -0.749, 1.956, 10.494, 54.938, 31.481, 1.235, 1.852],
    [12, "Motivated by Friends' Success", 162, 3.932, 4.000, 4.000, 0.998, 1.000, 5.000, -0.851, 0.067, 31.481, 43.827, 12.346, 11.111, 1.235],
    [13, "Value Friends' Suggestions", 162, 3.870, 4.000, 4.000, 1.016, 1.000, 5.000, -0.563, -0.473, 32.716, 33.333, 23.457, 9.259, 1.235],
    [14, "Follow Employment Information", 162, 3.883, 4.000, 4.000, 0.866, 1.000, 5.000, -0.525, 0.284, 25.309, 43.210, 27.160, 3.086, 1.235],
    [15, "Aware of Employer Expectations", 162, 3.951, 4.000, 4.000, 0.862, 1.000, 5.000, -0.612, 0.424, 28.395, 43.210, 24.691, 2.469, 1.235],
    [16, "Updated on Labor Market Trends", 162, 4.062, 4.000, 4.000, 0.832, 1.000, 5.000, -0.707, 0.411, 32.716, 45.062, 18.519, 3.086, 0.617],
    [17, "Job Market Understanding Helps", 162, 4.068, 4.000, 4.000, 0.812, 1.000, 5.000, -0.759, 0.698, 31.481, 48.148, 16.667, 3.086, 0.617],
    [18, "Career Matches Personal Strengths", 162, 4.093, 4.000, 5.000, 0.911, 1.000, 5.000, -0.884, 0.533, 38.889, 37.654, 18.519, 3.704, 1.235],
    [19, "Enjoy Interest-Matching Work", 162, 4.068, 4.000, 4.000, 0.812, 1.000, 5.000, -0.970, 1.658, 30.247, 51.235, 14.815, 2.469, 1.235],
    [20, "Personality Fits Preferred Profession", 162, 4.056, 4.000, 4.000, 0.879, 1.000, 5.000, -0.664, -0.009, 35.802, 38.889, 20.988, 3.704, 0.617],
    [21, "Values Important in Career Choice", 162, 4.025, 4.000, 4.000, 0.870, 1.000, 5.000, -0.850, 0.838, 31.481, 45.679, 17.901, 3.704, 1.235],
    [22, "Clear Defined Career Preference", 162, 3.951, 4.000, 4.000, 0.944, 1.000, 5.000, -0.528, -0.464, 33.951, 34.568, 24.691, 6.173, 0.617],
    [23, "Confident in Career Path", 162, 3.827, 4.000, 4.000, 0.853, 1.000, 5.000, -0.632, 0.568, 20.370, 49.383, 24.074, 4.938, 1.235],
    [24, "Pursue Career Matching Interests", 162, 4.025, 4.000, 4.000, 0.870, 1.000, 5.000, -0.793, 0.697, 32.099, 43.827, 19.753, 3.086, 1.235],
    [25, "Committed to Career Goals", 162, 3.988, 4.000, 4.000, 0.878, 1.000, 5.000, -0.813, 0.722, 29.630, 46.296, 18.519, 4.321, 1.235],
    [26, "Career Aligns with Long-term Goals", 160, 3.962, 4.000, 4.000, 0.882, 1.000, 5.000, -0.761, 0.592, 28.750, 45.625, 20.000, 4.375, 1.250]
]

# Create DataFrame with column order: Statement → N → Mean → Median → Mode → Std.Dev → Min → Max → Skewness → Kurtosis → Strongly Agree → Agree → Neutral → Disagree → Strongly Disagree
columns = [
    'No', 'Statement', 'N', 'Mean', 'Median', 'Mode', 'Std. Dev', 'Min', 'Max', 
    'Skewness', 'Kurtosis',
    '% Strongly Agree', '% Agree', '% Neutral', '% Disagree', '% Strongly Disagree'
]

df = pd.DataFrame(data, columns=columns)

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.3f}'.format)

# Print the reorganized table
print("="*180)
print("DESCRIPTIVE STATISTICS FOR LIKERT SCALE ITEMS (N=162)")
print("="*180)
print("\nNote: Percentages represent the proportion of respondents selecting each response option")
print("="*180)

# Display with proper formatting
print(df.to_string(index=False))

# Calculate and display summary statistics
print("\n" + "="*180)
print("SUMMARY STATISTICS")
print("="*180)

# Group by factor
factor_groups = {
    'Parental Education': ['Parental Encouragement for Education', 'Parents Discuss Career Opportunities', 
                          'Parental Education as Motivation', 'Seek Career Advice from Parents'],
    'Family Income': ['Family Finances Affect Education', 'Consider Cost in Career Planning', 
                     'Prefer Long-term Financial Security', 'Financial Responsibilities Affect Career',
                     'May Change Career Due to Finances'],
    'Peer Influence': ['Discuss Career with Friends', 'Friends Introduce Career Success', 
                      'Motivated by Friends\' Success', 'Value Friends\' Suggestions'],
    'Job Market': ['Follow Employment Information', 'Aware of Employer Expectations', 
                  'Updated on Labor Market Trends', 'Job Market Understanding Helps'],
    'Personality': ['Career Matches Personal Strengths', 'Enjoy Interest-Matching Work', 
                   'Personality Fits Preferred Profession', 'Values Important in Career Choice'],
    'Career Preference': ['Clear Defined Career Preference', 'Confident in Career Path', 
                         'Pursue Career Matching Interests', 'Committed to Career Goals',
                         'Career Aligns with Long-term Goals']
}

print("\nFactor-wise Mean Scores:")
print("-"*80)
for factor, items in factor_groups.items():
    factor_data = df[df['Statement'].isin(items)]
    if len(factor_data) > 0:
        mean_score = factor_data['Mean'].mean()
        print(f"{factor:25s}: {mean_score:.3f}")

# Save to Excel
with pd.ExcelWriter('career_preference_descriptive_stats.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Descriptive_Stats', index=False)
    
    # Add summary
    summary_data = []
    for factor, items in factor_groups.items():
        factor_data = df[df['Statement'].isin(items)]
        if len(factor_data) > 0:
            summary_data.append({
                'Factor': factor,
                'Number of Items': len(items),
                'Mean Score': factor_data['Mean'].mean(),
                'Median Score': factor_data['Median'].mean(),
                'Std Dev': factor_data['Std. Dev'].mean(),
                '% Strongly Agree': factor_data['% Strongly Agree'].mean(),
                '% Agree': factor_data['% Agree'].mean(),
                '% Neutral': factor_data['% Neutral'].mean(),
                '% Disagree': factor_data['% Disagree'].mean(),
                '% Strongly Disagree': factor_data['% Strongly Disagree'].mean()
            })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Factor_Summary', index=False)

print("\n" + "="*180)
print("✓ Results exported to 'career_preference_descriptive_stats.xlsx'")
print("="*180)