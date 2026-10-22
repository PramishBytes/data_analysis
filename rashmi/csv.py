import os
import csv

# Create a folder
os.makedirs("rashmi_tables", exist_ok=True)

tables = {
    "Table_1.csv": """Statistic,Taste_Composite,Freshness_Composite,Hygiene_Composite,Presentation_Composite,Portion_Composite,Satisfaction_Composite
N,165,,,,166
Mean,3.213,,,,,3.227
Median,3.000,,,,,3.000
Mode,3.000,,,,,3.000
Std_Deviation,0.723,,,,,0.738
Variance,0.523,,,,,0.545
Minimum,1.000,,,,,1.000
Maximum,5.000,,,,,5.000
Range,4.000,,,,,4.000
Skewness,-0.117,,,,,-0.216
Kurtosis,0.901,,,,,0.883
Std_Error_Mean,0.056,,,,,0.057
Sum,530.200,,,,,535.600
Percentile_25,3.000,,,,,3.000
Percentile_75,3.800,,,,,3.600
IQR,0.800,,,,,0.600""",

    "Table_2_Taste.csv": """Statement,Strongly_Disagree,Disagree,Neutral,Agree,Strongly_Agree,Total
The food served has good taste,17,21,44,60,23,165
The food is flavourful and well seasoned,16,28,44,57,20,165
The taste is consistent,22,34,42,45,22,165
Food meets expectations in terms of taste,18,31,42,52,22,165
I enjoy the taste of the food,17,25,42,55,26,165""",

    "Table_3_Freshness.csv": """Statement,Strongly_Disagree,Disagree,Neutral,Agree,Strongly_Agree,Total
The food served is fresh,21,32,43,48,21,165
Ingredients appear fresh,18,34,45,52,16,165
Served at appropriate temperature,18,37,40,48,22,165
Appears freshly prepared,19,26,42,58,20,165
I am satisfied with freshness,22,31,40,48,24,165""",

    "Table_4_Hygiene.csv": """Statement,Strongly_Disagree,Disagree,Neutral,Agree,Strongly_Agree,Total
Prepared in hygienic manner,19,28,53,42,22,164
Dining area clean,19,29,48,51,19,166
Plates/glasses/utensils clean,21,21,41,60,22,165
Staff maintain personal hygiene,21,24,42,56,22,165
Confident about cleanliness/safety,20,31,46,48,20,165""",

    "Table_5_Presentation.csv": """Statement,Strongly_Disagree,Disagree,Neutral,Agree,Strongly_Agree,Total
Served in attractive manner,21,20,57,44,23,165
Appearance appealing,19,30,40,57,19,165
Presented neatly,21,24,41,56,24,166
Increases appetite,22,21,48,53,21,165
Satisfied with presentation,22,17,52,51,23,165""",

    "Table_6_Portion.csv": """Statement,Strongly_Disagree,Disagree,Neutral,Agree,Strongly_Agree,Total
Portion size adequate,25,27,36,52,25,165
Quantity worth price,19,32,34,58,22,165
Portion consistent on every visit,25,21,41,52,25,164
Amount sufficient to satisfy hunger,21,29,34,56,26,166
Satisfied with portion size,26,24,36,53,26,165""",

    "Table_7_Satisfaction.csv": """Statement,Strongly_Disagree,Disagree,Neutral,Agree,Strongly_Agree,Total
Overall satisfied with food quality,22,22,42,54,25,165
Meets my expectations,15,28,47,57,18,165
Satisfied with overall dining experience,19,25,51,50,21,166
Will visit again,15,26,43,58,23,165
Would recommend,20,21,46,53,25,165""",

    "Table_8_Demographics.csv": """Variable,Category,Frequency,Percent,Valid_Percent,Cumulative_Percent
Age,20-25,103,62.05,62.05,62.05
Age,25-30,19,11.45,11.45,73.50
Age,Above 30,15,9.04,9.04,82.54
Age,Below 20,29,17.47,17.47,100.01
Gender,Female,86,51.81,51.81,51.81
Gender,Male,79,47.59,47.59,99.40
Occupation,Others,18,10.84,10.84,10.84
Occupation,Student,115,69.28,69.28,80.12
Occupation,Teacher_Faculty,18,10.84,10.84,90.96
Occupation,University_staff,15,9.04,9.04,100.00
Family_Income,50000-100000,67,40.36,40.36,40.36
Family_Income,Above_100000,49,29.52,29.52,69.88
Family_Income,Below_50000,48,28.92,28.92,98.80""",

    "Table_9_Cronbachs_Alpha.csv": """Scale,Number_of_Items,Respondents,Cronbachs_Alpha,Total_Scale_Mean,Total_Scale_Variance,Total_Scale_SD,Interpretation
Taste,5,165,0.5637,16.0667,13.0626,3.6142,Poor reliability
Freshness,5,164,0.5724,15.6646,13.5249,3.6776,Poor reliability
Hygiene,5,164,0.5783,15.8293,13.5044,3.6748,Poor reliability
Presentation,5,165,0.5427,15.9636,12.9499,3.5986,Poor reliability
Portion,5,164,0.6746,15.9390,17.6527,4.2015,Questionable reliability
Satisfaction,5,165,0.5944,16.1697,13.4710,3.6703,Poor reliability""",

    "Table_10_Item_Total.csv": """Scale,Item,Item_Total_Correlation,Alpha_if_Deleted
Taste,Item 1,0.3103,0.5155
Taste,Item 2,0.2968,0.5229
Taste,Item 3,0.3014,0.5215
Taste,Item 4,0.3327,0.5027
Taste,Item 5,0.3761,0.4770
Freshness,Item 1,0.3326,0.5159
Freshness,Item 2,0.3208,0.5226
Freshness,Item 3,0.2967,0.5361
Freshness,Item 4,0.3262,0.5195
Freshness,Item 5,0.3770,0.4894
Hygiene,Item 1,0.3184,0.5325
Hygiene,Item 2,0.3422,0.5195
Hygiene,Item 3,0.3460,0.5172
Hygiene,Item 4,0.3496,0.5151
Hygiene,Item 5,0.3217,0.5308
Presentation,Item 1,0.3147,0.4816
Presentation,Item 2,0.2938,0.4940
Presentation,Item 3,0.3032,0.4886
Presentation,Item 4,0.2709,0.5076
Presentation,Item 5,0.3468,0.4619
Portion,Item 1,0.3800,0.6454
Portion,Item 2,0.4228,0.6262
Portion,Item 3,0.4027,0.6350
Portion,Item 4,0.4652,0.6069
Portion,Item 5,0.4704,0.6042
Satisfaction,Item 1,0.3937,0.5146
Satisfaction,Item 2,0.2910,0.5695
Satisfaction,Item 3,0.3618,0.5332
Satisfaction,Item 4,0.3177,0.5564
Satisfaction,Item 5,0.3849,0.5201""",

    "Table_11_Pearson.csv": """Variable,Taste,Freshness,Hygiene,Presentation,Portion,Satisfaction
Taste_Composite,1.000,,,,
Freshness_Composite,0.540,1.000,,,,
Hygiene_Composite,0.410,,1.000,,,
Presentation_Composite,0.589,,,1.000,,
Portion_Composite,0.547,,,,1.000,
Satisfaction_Composite,0.620,0.681,0.659,0.748,0.768,1.000""",

    "Table_12_Spearman.csv": """Variable,Taste,Freshness,Hygiene,Presentation,Portion,Satisfaction
Taste_Composite,1.000,,,,
Freshness_Composite,0.530,1.000,,,,
Hygiene_Composite,0.385,,1.000,,,
Presentation_Composite,0.505,,,1.000,,
Portion_Composite,0.480,,,,1.000,
Satisfaction_Composite,0.549,0.572,0.575,0.651,0.704,1.000""",

    "Table_13_TTest_Gender.csv": """Dependent_Variable,Group1_N,Group1_Mean,Group1_SD,Group2_N,Group2_Mean,Group2_SD,t,df,p_value,Mean_Difference,Cohens_d
Taste_Composite,78,3.1846,0.8422,86,3.2302,0.5977,-0.4028,162,0.6876,-0.0456,0.0630
Freshness_Composite,79,3.1924,0.7848,86,3.0465,0.6994,1.2626,163,0.2085,0.1459,0.1968
Hygiene_Composite,79,3.2456,0.7961,86,3.0779,0.6662,1.4712,163,0.1432,0.1677,0.2293
Presentation_Composite,79,3.1899,0.7712,86,3.1860,0.6706,0.0341,163,0.9729,0.0038,0.0053
Portion_Composite,79,3.1070,0.9298,86,3.2442,0.7523,-1.0458,163,0.2972,-0.1372,0.1630
Satisfaction_Composite,79,3.1468,0.8075,86,3.2930,0.6670,-1.2718,163,0.2052,-0.1462,0.1982""",

    "Table_14_ANOVA_Income.csv": """Dependent_Variable,Income_Group,N,Mean,SD,F_statistic,p_value,Eta_squared
Taste_Composite,50000-100000,67,3.1851,0.7557,0.1103,0.8957,0.0014
Taste_Composite,Above_100000,48,3.2375,0.6294,,,
Taste_Composite,Below_50000,48,3.2417,0.7648,,,
Freshness_Composite,50000-100000,67,3.1582,0.7145,0.1820,0.8338,0.0023
Freshness_Composite,Above_100000,49,3.0735,0.6536,,,
Freshness_Composite,Below_50000,48,3.1167,0.8572,,,
Hygiene_Composite,50000-100000,67,3.1955,0.6675,0.2221,0.8011,0.0028
Hygiene_Composite,Above_100000,49,3.1061,0.6235,,,
Hygiene_Composite,Below_50000,48,3.1792,0.8984,,,
Presentation_Composite,50000-100000,67,3.2388,0.7365,0.6786,0.5088,0.0084
Presentation_Composite,Above_100000,49,3.2204,0.5470,,,
Presentation_Composite,Below_50000,48,3.0875,0.8306,,,
Portion_Composite,50000-100000,67,3.2687,0.8847,0.6607,0.5179,0.0081
Portion_Composite,Above_100000,49,3.1306,0.7851,,,
Portion_Composite,Below_50000,48,3.1010,0.8228,,,
Satisfaction_Composite,50000-100000,67,3.3134,0.7373,0.9596,0.3852,0.0118
Satisfaction_Composite,Above_100000,49,3.2163,0.6387,,,
Satisfaction_Composite,Below_50000,48,3.1208,0.8091,,,""",

    "Table_15_Regression_Summary.csv": """Statistic,Value
R_squared,0.7366
Adjusted_R_squared,0.7283
F_statistic,88.9094
p_value_F_test,0.0000
Number_of_Observations,165.0""",

    "Table_16_Regression_Coefficients.csv": """Variable,Coefficient,Std_Error,t_value,p_value
const,3.2339,0.0298,108.5618,0.0000
Taste_Composite,0.1026,0.0400,2.5671,0.0112
Freshness_Composite,0.0813,0.0461,1.7632,0.0798
Hygiene_Composite,0.0687,0.0507,1.3560,0.1770
Presentation_Composite,0.2097,0.0525,3.9928,0.0001
Portion_Composite,0.2842,0.0420,6.7644,0.0000""",

    "Table_17_VIF.csv": """Variable,VIF,Tolerance
Taste_Composite,32.688711,0.030592
Freshness_Composite,45.074565,0.022185
Hygiene_Composite,54.157301,0.018465
Presentation_Composite,64.346794,0.015541
Portion_Composite,30.923041,0.032338""",

    "Table_18_Gender_Age.csv": """Gender,20-25,25-30,Above_30,Below_20,Total
Female,55,8,9,14,86
Male,47,11,6,15,79""",

    "Table_19_Occupation_Education.csv": """Occupation,Bachelor,Masters,Total
Others,3,15,18
Student,112,2,114
Teacher_Faculty,0,18,18
University_staff,15,0,15""",

    "Table_20_Income_Visit.csv": """Family_Income,2-3_times_a_week,Total
50000-100000,19,67
Above_100000,10,49
Below_50000,10,48""",

    "Table_21_Gender_Income.csv": """Gender,50000-100000,Above_100000,Below_50000,Total
Female,38,28,20,86
Male,28,21,28,77""",

    "Table_22_Age_Visit.csv": """Age,2-3_times_a_week,Total
20-25,25,103
25-30,5,19
Above_30,3,15
Below_20,6,29""",

    "Table_29_Hypothesis_Summary.csv": """Hypothesis,Test,Result,p_value,Conclusion
H1_Taste_affects_satisfaction,Pearson_Correlation,Rejected,0.0000,Taste correlates with satisfaction
H2_Freshness_affects_satisfaction,Pearson_Correlation,Rejected,0.0000,Freshness correlates with satisfaction
H3_Hygiene_affects_satisfaction,Pearson_Correlation,Rejected,0.0000,Hygiene correlates with satisfaction
H4_Presentation_affects_satisfaction,Pearson_Correlation,Rejected,0.0000,Presentation correlates with satisfaction
H5_Portion_affects_satisfaction,Pearson_Correlation,Rejected,0.0000,Portion correlates with satisfaction
H6_Satisfaction_differs_by_income,One-Way_ANOVA,Not_Rejected,0.3852,No income differences
H7_All_factors_predict_satisfaction,Multiple_Regression,Rejected,0.0000,Model is significant""",

    "Table_30_Summary_Stats.csv": """Variable,Mean,Median,Std_Dev,Minimum,Maximum,Skewness,Kurtosis,N
Taste_Composite,3.213,3.000,0.723,1.000,5.000,-0.117,0.901,165
Freshness_Composite,3.119,3.000,0.741,1.000,5.000,-0.273,0.596,166
Hygiene_Composite,3.161,3.000,0.732,1.000,5.000,-0.131,0.762,166
Presentation_Composite,3.192,3.000,0.718,1.000,5.000,-0.662,2.049,166
Portion_Composite,3.181,3.000,0.840,1.000,5.000,-0.257,0.334,166
Satisfaction_Composite,3.227,3.000,0.738,1.000,5.000,-0.216,0.883,166""",

    "Table_31_Correlation_Satisfaction.csv": """Independent_Variable,Pearson_r,p_value,Significance
Taste_Composite,0.620,0.0000,Significant positive correlation
Freshness_Composite,0.681,0.0000,Significant positive correlation
Hygiene_Composite,0.659,0.0000,Significant positive correlation
Presentation_Composite,0.748,0.0000,Significant positive correlation
Portion_Composite,0.768,0.0000,Significant positive correlation"""
}

for filename, content in tables.items():
    with open(os.path.join("rashmi_tables", filename), 'w', newline='', encoding='utf-8') as f:
        f.write(content)

print(f"✅ Created {len(tables)} CSV files in 'rashmi_tables' folder!")