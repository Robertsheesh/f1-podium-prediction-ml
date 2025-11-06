# Predicting Formula 1 Podium Finishes Using Machine Learning

This project was completed as a group assignment for the course **Data Science for Business I** at **Aalto University**.

## 1. Introduction

Formula 1 is a highly competitive motorsport where performance is determined by a complex interaction of thousands of variables. Factors such as driver skill, vehicle characteristics, team performance, and race conditions all influence race outcomes. The ability to predict final race positions is valuable for stakeholders including sports bettors, marketers, and investors, as well as for teams seeking to enhance strategic and technical performance. This project focuses on predicting whether a driver will finish on the podium (top three positions) based solely on data available before each race weekend. To achieve this, we applied and compared five machine learning models: logistic regression (with ridge and lasso regularization), Random Forest, and XGBoost.

The dataset used spans Formula 1 seasons from 1950 to 2024. However, our analysis was limited to the 2014–2024 “hybrid era,” following major regulatory changes that reshaped competitive dynamics. Restricting the data to this period ensured model stability and comparability.

## 2. Problem Formulation

This project represents a supervised machine learning task, where both input features and output labels are known. Specifically, the goal is to classify whether a driver finishes on the podium (class = 1) or not (class = 0), making it a binary classification problem. Input features capture various aspects of a Formula 1 race, including contextual, driver-specific, and race-specific variables. These include the driver’s grid position, team (constructor), circuit, year, round, nationality, age, number of laps, and circuit location. These multidimensional features allow the model to learn patterns associated with podium performance.

The data originates from the Formula 1 World Championship dataset on Kaggle, published by Rohan Rao. It consists of 15 interlinked tables derived from the Ergast API. For this project, we used four core tables:


| **Table**        | **Description**                              | **Key Columns**                                | **Rows** |
|------------------|----------------------------------------------|------------------------------------------------|----------:|
| `races.csv`      | Details of each Formula 1 race               | raceId, year, round, circuitId, date           | 1,125 |
| `drivers.csv`    | Information about all drivers since 1950     | driverId, driverRef, code, dob                 | 861 |
| `results.csv`    | Results of each race for each driver         | resultId, raceId, driverId, grid, positionOrder, laps | 26,759 |
| `circuits.csv`   | Details of each circuit                      | circuitId, circuitRef, name, country           | 77 |


These tables were merged through their key identifiers (e.g., raceId, driverId) to produce a unified dataset suitable for supervised learning.

## 3. Methods
### 3.1 Data Preparation and Feature Engineering

After merging the selected tables, we cleaned the dataset by removing incomplete or invalid rows and replacing missing values with NaN. Since the original dataset did not include a podium indicator, we created a binary target column, podium, where 1 indicates a top-three finish and 0 indicates otherwise. 

The resulting dataset contained **26,759 rows** and **19 columns**. Only **12.7%** of records represented podium finishes, highlighting a substantial class imbalance. 

Feature selection was guided by domain knowledge. We computed driver age based on date of birth and race year. Categorical features such as driver nationality and circuit location were excluded to prevent high dimensionality from one-hot encoding, as these variables were expected to have limited predictive power relative to other features. 

To capture driver performance trends, we engineered several rolling average features using historical data over windows of 3, 5, 10, 15, 20, and 50 previous races. For each window, we calculated: 
- Number of previous podiums
- Average grid position
- Average finishing position

For new drivers, these values were initialized as 0 (for podiums) and 20 (for grid/finish averages), reflecting expected back-of-the-grid performance for rookies.

### 3.2 Train–Validation–Test Split

Given the time-dependent nature of the data, we performed a chronological split to avoid data leakage:

- **Training:** 2014–2021
- **Validation:** 2022–2023
- **Test:** 2024

### 3.3 Model Selection and Hyperparameter Tuning

We trained and evaluated five models:
1. Logistic Regression (baseline)
2. Logistic Regression with Lasso (L1) regularization
3. Logistic Regression with Ridge (L2) regularization
4. Random Forest
5. XGBoost

For the logistic models, we applied class_weight="balanced" to mitigate label imbalance and standardized features using StandardScaler(). Optimal regularization strengths (C) were determined via grid search to maximize the F1-score:

- **Lasso:** C = 0.1
- **Ridge:** C = 0.01

For tree-based models, Random Forest captured non-linear relationships, while XGBoost leveraged boosting to iteratively correct prior errors. For XGBoost, we set the scale_pos_weight parameter according to the class ratio.

3.4 Evaluation Metrics

Given the imbalance in class distribution, we prioritized precision, recall, and F1-score over accuracy. Additional evaluation included confusion matrices and feature importance analyses.

## 4. Results

| **Model**              | **Precision (Podium)** | **Recall (Podium)** | **F1-score (Podium)** |
|------------------------|-----------------------:|--------------------:|----------------------:|
| Logistic Regression    | 0.413 | 0.879 | 0.562 |
| Lasso Regression       | 0.415 | 0.886 | 0.565 |
| Ridge Regression       | 0.411 | 0.879 | 0.560 |
| Random Forest          | 0.750 | 0.159 | 0.263 |
| XGBoost                | 0.446 | 0.591 | 0.508 |


Logistic regression variants performed similarly, with Lasso achieving the best F1-score due to its sparsity-inducing regularization. Random Forest overfit the majority class, resulting in low recall. XGBoost achieved a more balanced trade-off, with higher recall than Random Forest and competitive precision.

Model selection depends on context:
- **High precision** suits betting applications where false positives are costly.
- **High recall** benefits team analytics seeking to identify potential podium contenders.

For a balanced evaluation, we emphasized the F1-score. Based on validation results, Lasso Regression and XGBoost were selected for final testing.

## 5. Final Testing and Model Selection

| **Model**              | **Precision (Podium)** | **Recall (Podium)** | **F1-score (Podium)** |
|------------------------|-----------------------:|--------------------:|----------------------:|
| XGBoost                | 0.487 | 0.792 | **0.603** |
| Lasso Regression       | 0.527 | 0.681 | 0.594 |


Both models performed similarly, but XGBoost achieved the highest F1-score, confirming its robustness across datasets.

Feature importance analysis revealed that previous podium finishes were the strongest predictors of future success, followed by rolling averages of grid positions and the race year. Interestingly, the average grid position was more predictive than the average finish position, likely due to random events (e.g., crashes, mechanical failures) skewing final results.

## 6. Conclusion

This project developed a machine learning framework for predicting Formula 1 podium finishes using pre-race data from the 2014–2024 hybrid era. The task was formulated as a binary classification problem and addressed using five supervised learning models.

Among these, XGBoost delivered the best overall performance, offering the most balanced precision and recall and achieving the highest F1-score on the test set. Logistic models performed well as interpretable baselines, while Random Forest suffered from recall deficiencies.

Key takeaways include:

- Feature engineering using rolling averages substantially improved predictive power.
- Regularization stabilized logistic regression performance without overfitting.
- Chronological validation prevented temporal data leakage, ensuring realistic evaluation.

Future work could incorporate external contextual features such as weather conditions, tire strategies, and qualifying performance. Advanced temporal models like recurrent neural networks (RNNs) or transformer-based architectures could further capture driver and team dynamics across seasons.

Overall, the results demonstrate that even with limited pre-race data, machine learning can provide valuable insights into Formula 1 performance and offer a strong foundation for future predictive analytics in motorsports.
