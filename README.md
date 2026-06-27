# Credit Risk Assessment and Default Prediction using Machine Learning

## 1. Project Overview

Credit risk assessment is an important application of statistics and machine learning in banking, lending, and financial analytics. Banks and credit card companies provide credit facilities to customers with the expectation that repayments will be made on time. However, some customers may fail to repay their dues in the next billing cycle, creating credit risk and possible financial loss for the institution.

This project focuses on predicting whether a credit card customer is likely to default on payment in the next month. The analysis uses customer demographic details, credit limit, repayment history, monthly bill amounts, previous payment amounts, and engineered credit-risk features.

The aim of the project is not only to build a predictive model, but also to understand the customer behavior patterns that are associated with default. Therefore, the project combines statistical thinking, exploratory analysis, machine learning, model evaluation, and business interpretation.

This makes the project relevant for roles such as Data Analyst, Business Analyst, Risk Analyst, Credit Risk Analyst, Banking Analytics, and Consulting Analytics.

---

## 2. Project Title

**Credit Risk Assessment and Default Prediction using Machine Learning**

---

## 3. Problem Statement

Credit card default can lead to financial loss for banks and lending institutions. If high-risk customers can be identified before default occurs, the institution can take preventive actions such as closer monitoring, repayment reminders, credit limit review, or early risk intervention.

The problem addressed in this project is to predict whether a customer will default on credit card payment in the next month based on past repayment behavior, credit usage, bill amounts, payment amounts, and demographic characteristics.

This is a binary classification problem where the target variable is:

- `0` = Customer did not default
- `1` = Customer defaulted

---

## 4. Objective of the Project

The main objective of this project is to develop a machine learning framework for credit card default prediction.

The project aims to:

1. Understand the structure and quality of the credit card dataset.
2. Clean and validate categorical and numerical variables.
3. Perform exploratory data analysis to identify default-related patterns.
4. Engineer meaningful credit-risk features such as utilization ratio.
5. Check multicollinearity using Variance Inflation Factor because Logistic Regression is used as a statistical baseline model.
6. Build and compare Logistic Regression, Random Forest, and XGBoost models.
7. Handle class imbalance using model-based techniques.
8. Evaluate models using Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix, and Feature Importance.
9. Select a final model that is suitable for credit risk prediction.
10. Validate the final model using cross-validation.
11. Perform threshold tuning to understand the business trade-off between Precision and Recall.
12. Provide business recommendations based on the final model and analysis.

---

## 5. Dataset Description

The dataset contains **30,000 customer records** and initially had **25 columns**. The variables include customer demographic details, credit limit, repayment history, monthly bill amounts, monthly payment amounts, and the target variable indicating whether the customer defaulted in the next month.

### Main Categories of Variables

#### Demographic Variables
- Gender
- Education
- Marriage status
- Age

#### Credit and Financial Variables
- Credit limit
- Monthly bill amounts
- Monthly payment amounts

#### Repayment History Variables
- Repayment status from April to September

#### Target Variable
- Default payment next month

The dataset is suitable for a credit risk classification problem because it contains both behavioral and financial indicators that can help predict default risk.

---

## 6. Tools and Libraries Used

The project was implemented in Python using the following libraries:

- **Pandas** for data manipulation and analysis
- **NumPy** for numerical operations
- **Matplotlib** and **Seaborn** for visualization
- **Statsmodels** for VIF analysis
- **Scikit-learn** for preprocessing, model building, and evaluation
- **XGBoost** for boosting-based classification

---

## 7. Project Methodology

The project follows a structured machine learning workflow:

1. Data loading and inspection
2. Data cleaning and categorical validation
3. Exploratory data analysis
4. Feature engineering
5. Correlation analysis
6. Multicollinearity check using VIF
7. Final modeling dataset preparation
8. Train-test split
9. Feature scaling for Logistic Regression
10. Baseline model building
11. Class imbalance handling
12. Balanced model comparison
13. Final model selection
14. Final model evaluation
15. Cross-validation
16. Threshold tuning
17. Business recommendations and conclusion

This workflow keeps the project clean, statistically meaningful, and explainable for placement interviews.

---

## 8. Data Understanding

The dataset was first examined using basic inspection methods such as shape, head, info, and descriptive statistics.

### Key Observations

- The dataset contains **30,000 observations**.
- There are no missing values.
- There are no duplicate records.
- The dataset contains demographic, financial, repayment history, and target variables.
- The target variable is binary.

The initial inspection showed that the dataset was suitable for classification modeling, but categorical values still required validation.

---

## 9. Data Cleaning

Data cleaning was performed to improve consistency and interpretability.

### Missing Values

The dataset was checked for missing values. No missing values were found.

Since there were no missing values, no imputation method was required.

### Duplicate Records

The dataset was checked for duplicate observations. No duplicate records were found.

This ensured that repeated records did not bias the analysis or model training.

### Column Renaming

The original column names were renamed to make them more readable and presentation-friendly.

Examples:

- `LIMIT_BAL` was renamed to `Credit_Limit`
- `SEX` was renamed to `Gender`
- `EDUCATION` was renamed to `Education`
- `MARRIAGE` was renamed to `Marriage`
- `PAY_0`, `PAY_2`, ..., `PAY_6` were renamed as monthly repayment status variables
- `default.payment.next.month` was renamed to `Default`

This step made the notebook easier to understand and improved the interpretation of results.

---

## 10. Categorical Variable Validation

Categorical variables were checked to ensure that their categories were meaningful and valid.

### Gender

Gender had two valid categories:

- `1` = Male
- `2` = Female

No invalid category was found in Gender.

### Education

Education had the following expected categories:

- `1` = Graduate School
- `2` = University
- `3` = High School
- `4` = Others

The dataset also contained values such as `0`, `5`, and `6`, which were not clearly defined. These undefined categories were merged into category `4`, representing Others.

This was done to avoid treating invalid or unclear categories as separate meaningful education levels.

### Marriage

Marriage had the following expected categories:

- `1` = Married
- `2` = Single
- `3` = Others

The dataset also contained value `0`, which was not clearly defined. This value was merged into category `3`, representing Others.

This validation step improved data consistency before modeling.

---

## 11. Target Variable Analysis

The target variable `Default` was analyzed to understand the class distribution.

The dataset contained:

- **23,364 non-defaulters**
- **6,636 defaulters**

This means that approximately:

- **77.9%** of customers did not default
- **22.1%** of customers defaulted

### Interpretation

The dataset is moderately imbalanced because the number of non-defaulters is much higher than the number of defaulters.

In such cases, accuracy alone can be misleading. A model may get high accuracy by predicting most customers as non-defaulters, but this would not be useful for credit risk because the main business concern is identifying actual defaulters.

Therefore, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix were used along with Accuracy.

---

## 12. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify patterns related to default.

### Gender-wise Default Analysis

Default rate was analyzed across gender categories. Percentage-based default rates were used instead of only raw counts because one category may have more observations than another.

### Education-wise Default Analysis

Default behavior was compared across education categories. Education showed some variation in default behavior, but it was not found to be the strongest predictor.

### Marital Status-wise Default Analysis

Default rate was analyzed across marital status categories. This gave demographic insight, but marital status did not dominate default prediction.

### Age Distribution

Age distribution showed that most customers were concentrated in the age group of approximately 25 to 45 years. Age was included as a predictor, but repayment behavior later appeared to be more important.

### Credit Limit Distribution

Credit limit distribution was right-skewed, meaning most customers had relatively lower credit limits while a smaller number of customers had very high credit limits.

The relationship between credit limit and default indicated that customers with higher credit limits generally had lower default risk. This may be because higher credit limits are often assigned to customers with better credit profiles.

### Repayment Status Analysis

Repayment status variables were among the most important variables in the project. Customers with delayed repayment behavior in recent months were more likely to default in the next month.

This finding is important from a business perspective because recent repayment behavior can act as an early warning signal.

### Bill Amount and Payment Amount Distribution

Monthly bill amount and payment amount variables were analyzed. These variables were right-skewed, which is common in financial datasets because a few customers may have very high bill or payment amounts.

These variables provide useful information about customer spending and repayment behavior.

---

## 13. Feature Engineering

Feature engineering was performed to create variables that better represent customer credit behavior.

### Total Bill

`Total_Bill` was created by summing all six monthly bill amount variables.

This represents the total billed amount across the observed months.

### Total Payment

`Total_Payment` was created by summing all six monthly payment amount variables.

This represents the total repayment made by the customer across the observed months.

### Utilization Ratio

`Utilization_Ratio` was created as:

`Total_Bill / Credit_Limit`

This feature measures how much of the available credit limit has been used by the customer.

### Payment-to-Bill Ratio

`Payment_to_Bill_Ratio` was explored to understand repayment behavior relative to billed amount.

### Importance of Feature Engineering

Feature engineering added a business meaning to the model. In particular, `Utilization_Ratio` was retained because it represents credit usage intensity. Customers with high utilization may be under greater financial pressure and may have higher default risk.

---

## 14. Correlation Analysis

Correlation analysis was performed to identify variables associated with default.

### Key Findings

The strongest positive correlations with default were found in repayment status variables:

- `Repayment_Status_Sep`
- `Repayment_Status_Aug`
- `Repayment_Status_Jul`
- `Repayment_Status_Jun`
- `Repayment_Status_May`
- `Repayment_Status_Apr`

This means that delayed repayment behavior in previous months is strongly associated with default in the following month.

`Credit_Limit` showed a negative relationship with default, indicating that customers with higher credit limits were less likely to default.

`Utilization_Ratio` showed a positive association with default, supporting the idea that higher credit usage is linked with higher default risk.

---

## 15. Correlation Heatmap

A correlation heatmap was created for important variables to visually examine relationships among predictors.

### Observation

The heatmap showed that repayment status variables were correlated with each other and were also related to default behavior.

This is expected because repayment behavior often follows a pattern over time. A customer who delays repayment in one month may also delay repayment in the following months.

---

## 16. Multicollinearity Check using VIF

Since Logistic Regression was used as an important statistical baseline model, multicollinearity was checked using Variance Inflation Factor.

VIF helps detect whether independent variables are highly correlated with each other.

### VIF Findings

The VIF analysis showed infinite VIF values for some monthly bill and payment-related variables, especially when total variables were included.

This happened because:

`Total_Bill = BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6`

and

`Total_Payment = PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6`

Therefore, keeping both total variables and monthly variables creates perfect multicollinearity.

### Final Decision

To avoid multicollinearity issues, the following variables were removed from the final modeling dataset:

- `ID`
- `Total_Bill`
- `Total_Payment`
- `Payment_to_Bill_Ratio`
- Label columns used only for visualization, if present

However, `Utilization_Ratio` was retained because:

1. It has strong business interpretation.
2. It captures credit usage behavior.
3. It showed useful association with default.
4. It appeared as an important feature in the final model.

---

## 17. Final Modeling Dataset

After feature selection and multicollinearity handling, the final modeling dataset contained:

- **24 predictor variables**
- **1 target variable**

The final predictors included:

- Credit limit
- Gender
- Education
- Marriage
- Age
- Monthly repayment status variables
- Monthly bill amount variables
- Monthly payment amount variables
- Utilization ratio

The target variable was:

- Default

The final dataset was used for all model building steps.

---

## 18. Train-Test Split

The dataset was divided into training and testing sets using an 80:20 split.

- Training set: **24,000 records**
- Testing set: **6,000 records**

A stratified split was used to preserve the same proportion of defaulters and non-defaulters in both training and testing data.

This was important because the dataset was imbalanced.

---

## 19. Feature Scaling

Feature scaling was applied using StandardScaler for Logistic Regression.

Scaling was important for Logistic Regression because it is affected by the scale of independent variables.

Tree-based models such as Random Forest and XGBoost do not require scaling, so unscaled features were used for those models.

---

## 20. Model Building

Three machine learning models were developed:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Logistic Regression

Logistic Regression was used as the main statistical baseline model. It is suitable for binary classification and is highly interpretable.

This model aligns well with MSc Statistics because it is based on probability, log-odds, and classification theory.

### Random Forest

Random Forest was used as a comparison model. It combines multiple decision trees and can capture non-linear relationships.

It was useful for comparing performance and obtaining model-based feature importance.

### XGBoost

XGBoost was used as an advanced boosting model. It is widely used for structured/tabular data and can capture complex non-linear patterns.

---

## 21. Baseline Model Performance

The baseline models were first trained without explicit imbalance handling.

### Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.809 | 0.697 | 0.244 | 0.362 | 0.709 |
| Random Forest | 0.814 | 0.643 | 0.362 | 0.463 | 0.753 |
| XGBoost | 0.817 | 0.657 | 0.359 | 0.464 | 0.775 |

### Interpretation

The baseline models achieved good accuracy, but their recall values were low.

This means they missed many actual defaulters. In credit risk prediction, this is a serious issue because missing an actual defaulter can lead to financial loss.

Therefore, class imbalance handling was required.

---

## 22. Class Imbalance Handling

The target variable was imbalanced, with only about 22% defaulters.

Imbalance handling was incorporated directly within the models.

### Techniques Used

For Logistic Regression and Random Forest:

- `class_weight='balanced'`

For XGBoost:

- `scale_pos_weight`

These techniques help the models give more importance to the minority class, which is the defaulter class.

This is important because in credit risk analytics, identifying defaulters is usually more important than only maximizing accuracy.

---

## 23. Balanced Model Performance

After handling class imbalance, the models were evaluated again.

### Balanced Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Balanced Logistic Regression | 0.685 | 0.371 | 0.616 | 0.464 | 0.710 |
| Balanced Random Forest | 0.812 | 0.644 | 0.338 | 0.444 | 0.757 |
| Balanced XGBoost | 0.761 | 0.469 | 0.611 | 0.531 | 0.776 |

### Interpretation

After imbalance handling, recall improved significantly for Logistic Regression and XGBoost.

Balanced XGBoost achieved a recall of approximately **61.1%**, meaning it identified a much larger proportion of actual defaulters compared to the baseline XGBoost model.

Although the accuracy decreased compared to baseline XGBoost, this trade-off is acceptable in credit risk because the cost of missing a defaulter is usually higher than the cost of flagging a customer for review.

---

## 24. Final Model Selection

Balanced XGBoost was selected as the final model.

### Reasons for Selecting Balanced XGBoost

Balanced XGBoost was selected because:

1. It achieved the best F1 Score among the balanced models.
2. It maintained a strong ROC-AUC score.
3. It significantly improved recall compared to baseline XGBoost.
4. It handled class imbalance effectively using `scale_pos_weight`.
5. It is suitable for structured financial data.
6. It provided feature importance for business interpretation.

### Final Model Performance on Test Data

| Metric | Value |
|---|---:|
| Accuracy | 0.7608 |
| Precision | 0.4688 |
| Recall | 0.6112 |
| F1 Score | 0.5306 |
| ROC-AUC | 0.7764 |

### Interpretation

The final model is suitable for credit risk monitoring because it captures more actual defaulters than the baseline models.

In this project, Recall, F1 Score, and ROC-AUC were prioritized over Accuracy because the business cost of missing an actual defaulter is high.

---

## 25. Confusion Matrix Interpretation

The final Balanced XGBoost confusion matrix was:

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 3754 | 919 |
| Actual 1 | 516 | 811 |

This means:

- **True Negatives = 3754**: Non-defaulters correctly identified
- **False Positives = 919**: Non-defaulters incorrectly flagged as defaulters
- **False Negatives = 516**: Defaulters missed by the model
- **True Positives = 811**: Defaulters correctly identified

### Interpretation

The model correctly identified **811 actual defaulters** and missed **516 defaulters**.

Compared to baseline models, Balanced XGBoost improved the detection of actual defaulters. Although false positives increased, this trade-off is acceptable in credit risk because reviewing a customer incorrectly flagged as risky is usually less costly than missing an actual defaulter.

---

## 26. ROC Curve Interpretation

The ROC curve was plotted for the final Balanced XGBoost model.

The ROC-AUC score was approximately **0.7764**.

### Interpretation

A ROC-AUC score of 0.7764 means that the model has good ability to distinguish between defaulters and non-defaulters.

The ROC curve was above the diagonal line, meaning the model performed better than random classification.

---

## 27. Feature Importance Interpretation

Feature importance was extracted from the final Balanced XGBoost model.

### Important Predictors

The most important predictors were mainly related to:

- Recent repayment status
- Previous repayment status
- Utilization ratio
- Credit limit
- Payment amount variables
- Bill amount variables

### Interpretation

The final model showed that recent repayment behavior is the strongest indicator of future default risk. `Repayment_Status_Sep` appeared as the most important feature, showing that the most recent repayment status has the highest impact on default prediction.

The engineered feature `Utilization_Ratio` also appeared among the important predictors, which supports the feature engineering step. From a business perspective, customers with delayed repayments and high credit usage are more likely to default.

---

## 28. Cross-Validation of Final Model

Cross-validation was performed to check whether the final Balanced XGBoost model gives stable performance across different subsets of the data.

Since the target variable is imbalanced, Stratified K-Fold Cross-Validation was used. This ensures that each fold contains a similar proportion of defaulters and non-defaulters.

### Cross-Validation Results

| Metric | Value |
|---|---:|
| Mean ROC-AUC | 0.7812 |
| Standard Deviation of ROC-AUC | 0.0052 |
| Mean F1 Score | 0.5390 |
| Standard Deviation of F1 Score | 0.0117 |
| Mean Recall | 0.6231 |
| Mean Precision | 0.4749 |

### Interpretation

The mean ROC-AUC of approximately **0.7812** with a very low standard deviation of **0.0052** shows that the model performance is stable across different folds.

This confirms that the final model is not dependent only on one train-test split.

---

## 29. Threshold Tuning

By default, classification models use a threshold of 0.50. If the predicted probability of default is greater than or equal to 0.50, the customer is classified as a defaulter.

However, in credit risk analytics, the threshold can be adjusted depending on business cost and risk appetite.

### Threshold Tuning Results

| Threshold | Accuracy | Precision | Recall | F1 Score | False Positives | False Negatives |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.4575 | 0.2782 | 0.9111 | 0.4262 | 3137 | 118 |
| 0.30 | 0.5375 | 0.3059 | 0.8598 | 0.4513 | 2589 | 186 |
| 0.35 | 0.6137 | 0.3420 | 0.8086 | 0.4807 | 2064 | 254 |
| 0.40 | 0.6777 | 0.3807 | 0.7302 | 0.5005 | 1576 | 358 |
| 0.45 | 0.7293 | 0.4282 | 0.6669 | 0.5215 | 1182 | 442 |
| 0.50 | 0.7608 | 0.4688 | 0.6112 | 0.5306 | 919 | 516 |
| 0.55 | 0.7833 | 0.5091 | 0.5682 | 0.5370 | 727 | 573 |
| 0.60 | 0.7960 | 0.5400 | 0.5237 | 0.5318 | 592 | 632 |

The best threshold based on F1 Score was **0.55**.

### Interpretation

Threshold tuning shows the trade-off between Precision and Recall.

At threshold **0.50**, the model captures more actual defaulters with Recall of approximately **61.1%**.

At threshold **0.55**, the F1 Score improves slightly to approximately **0.5370**, and Precision improves to approximately **50.9%**, but Recall decreases to approximately **56.8%**.

This means that threshold 0.55 gives a slightly better balance between Precision and Recall, while threshold 0.50 is better when the business wants to identify more defaulters.

In this project, threshold tuning is presented as a business decision tool. The final threshold can be selected based on the bank's risk appetite and cost of false positives versus false negatives.

---

## 30. Business Insights

The project produced the following key business insights.

### Insight 1: Repayment History is the Strongest Risk Indicator

Customers with delayed repayment status in recent months were more likely to default in the following month.

This suggests that banks should closely monitor customers with recent repayment delays.

### Insight 2: Credit Utilization Matters

Customers with high credit utilization showed higher default risk.

This indicates that utilization ratio can be used as an early warning indicator of financial stress.

### Insight 3: Higher Credit Limit is Associated with Lower Default Risk

Customers with higher credit limits generally showed lower default risk.

This may be because higher credit limits are often assigned to customers with better credit profiles.

### Insight 4: Accuracy Alone is Not Enough

Baseline models achieved around 81% accuracy but missed many actual defaulters.

For credit risk, Recall and F1 Score are more important because the cost of missing a defaulter can be high.

### Insight 5: Class Imbalance Handling Improved Risk Detection

Balanced XGBoost improved recall and helped identify more actual defaulters.

This made the model more suitable for credit risk analytics than a model that focuses only on accuracy.

### Insight 6: Threshold Can Be Used as a Business Lever

Threshold tuning showed that different thresholds produce different Precision and Recall values.

A lower threshold can identify more defaulters, while a higher threshold can reduce false alarms.

---

## 31. Business Recommendations

Based on the analysis and model results, the following recommendations can be made.

### 1. Monitor Customers with Recent Payment Delays

Customers with delayed repayment status in recent months should be flagged for closer monitoring.

### 2. Track Credit Utilization Ratio

Customers with high utilization ratios should be considered higher risk because high credit usage may indicate financial stress.

### 3. Use the Model as an Early Warning System

The final Balanced XGBoost model can help identify potentially risky customers before default occurs.

### 4. Use Thresholds Based on Business Objective

If the bank wants to catch more defaulters, it can use a lower threshold. If the bank wants fewer false alarms, it can use a higher threshold.

### 5. Review Credit Limits for High-Risk Customers

Customers with repeated delays and high utilization may require credit limit review or temporary restrictions.

### 6. Provide Preventive Support

Instead of waiting for default, banks can send repayment reminders, offer repayment restructuring options, or provide financial counseling for high-risk customers.

---

## 32. Conclusion

This project developed a complete machine learning framework for credit card default prediction.

The dataset was cleaned, validated, and explored through EDA. Feature engineering was performed to create credit-risk indicators such as utilization ratio. Multicollinearity was checked using VIF before model building.

Logistic Regression, Random Forest, and XGBoost were trained and compared. Logistic Regression was used as the statistical baseline model, while Random Forest and XGBoost were used for performance comparison.

The baseline models achieved good accuracy, but recall was relatively low. Since the target variable was imbalanced, class imbalance handling was applied using `class_weight='balanced'` and `scale_pos_weight`.

Balanced XGBoost was selected as the final model because it achieved the best trade-off between Recall, F1 Score, and ROC-AUC.

The final model achieved:

- Accuracy: **0.7608**
- Precision: **0.4688**
- Recall: **0.6112**
- F1 Score: **0.5306**
- ROC-AUC: **0.7764**

Cross-validation further showed that the model performance was stable, with a mean ROC-AUC of approximately **0.7812**.

Threshold tuning was also performed to understand the trade-off between Precision and Recall from a business perspective.

Overall, the project shows that repayment history, utilization ratio, credit limit, and payment behavior are important indicators of credit default risk.

---


## 33. Future Scope

The following improvements can be explored in future work:

1. **SHAP Explainability**  
   SHAP can be added to explain global and individual customer-level predictions.

2. **Probability Calibration**  
   Calibration methods such as Platt Scaling or Isotonic Regression can be used if the model is deployed for production-level probability of default estimation.

3. **Hyperparameter Tuning**  
   GridSearchCV or RandomizedSearchCV can be applied to further optimize XGBoost parameters.

4. **One-Hot Encoding Refinement**  
   One-hot encoding can be applied to categorical variables such as Gender, Education, and Marriage, especially for a more refined Logistic Regression model.

5. **Cost-Sensitive Threshold Optimization**  
   Business costs can be assigned to false positives and false negatives to select an optimal threshold based on financial impact.

6. **Model Monitoring**  
   If used in production, the model should be monitored over time to check whether customer behavior patterns or default rates change.

---


The project is suitable for placement interviews because it is technically strong, statistically meaningful, and easy to explain clearly.

