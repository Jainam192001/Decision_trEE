# Decision Tree Practice: Predicting Student Pass/Fail

A beginner-friendly project to understand how decision trees work in practice, using a small hand-made dataset and scikit-learn.

## Goal

Learn how a decision tree picks questions, splits data, and overfits, by building one step by step and inspecting the result.

## Dataset

`students.csv`: 20 students, 5 columns of input and 1 target.

| Column | Type | Role |
|---|---|---|
| StudentID | number | ID (dropped, not a feature) |
| StudyHours | number | Feature |
| SleepHours | number | Feature |
| DoneHomework | Yes/No | Feature |
| Attendance | Good/Poor | Feature |
| Passed | Yes/No | **Target** |

## Concepts covered

- **Decision tree basics:** a flowchart of yes/no questions that splits data into purer groups.
- **Impurity measures:** Gini impurity and entropy (information gain = entropy before split − weighted entropy after).
- **Stopping and overfitting:** limiting `max_depth` (pre-pruning) vs. growing a full tree.
- **Preparing data:** separating features (X) from the target (y), dropping ID columns, one-hot encoding text columns.

I also worked through the classic "Play Tennis" dataset by hand to verify the information gain calculations (Outlook has the highest gain, about 0.247, so it becomes the root).

## Workflow

1. Load the CSV with pandas and inspect it (`head()`, `shape`).
2. Split into features `X` and target `y`; drop `StudentID`.
3. Encode text columns with `pd.get_dummies(X, drop_first=True)`.
4. Train/test split (80/20, `random_state=42`).
5. Train `DecisionTreeClassifier(criterion=..., max_depth=3)`.
6. Evaluate with accuracy and a confusion matrix.
7. Inspect the learned rules with `export_text` and the drawn tree with `plot_tree`.
8. Check feature importances.
9. Compare tree depths to see overfitting.

## Results

With `SleepHours` included as a feature and `max_depth=3`:

| Criterion | Train accuracy | Test accuracy | 5-fold CV accuracy |
|---|---|---|---|
| gini | 0.94 | 0.75 | 0.65 |
| entropy | 0.94 | 0.75 | 0.65 |

The learned tree starts with `SleepHours <= 5.5` (everyone with 5 hours of sleep failed), then uses `StudyHours` and `Attendance`.

## Key takeaways

- **Gini and entropy gave identical results** here. They usually produce very similar trees, so tuning depth matters more than the criterion.
- **Feature choice mattered most.** An earlier run accidentally dropped `SleepHours` instead of `StudentID`, which changed the root question and the accuracy. Changing the criterion at the same time made it look like entropy helped, but it was the feature change. Change one thing at a time when comparing.
- **Overfitting is visible:** training accuracy reaches 1.00 with deeper trees, while test accuracy does not reliably improve.
- **Tiny test sets are noisy.** With 20 rows, the test set has only 4 students, so one prediction moves accuracy by 25%. Cross-validation gives a more honest estimate.
- **Name mismatch gotcha:** after one-hot encoding, column names change (e.g. `Attendance` becomes `Attendance_Poor`). Use the encoded column names when calling `export_text`, `plot_tree`, or building a feature importance table, or the printed rules will be misleading.

## Limitations

- The dataset is very small and hand-made, so results are for learning, not for real conclusions.
- No hyperparameter tuning beyond `max_depth`.

## Next steps

- Try a larger dataset (Titanic, Iris) from Kaggle.
- Use cross-validation and `GridSearchCV` to tune `max_depth` and `min_samples_leaf`.
- Compare with a Random Forest and Gradient Boosting.
- Try post-pruning with `ccp_alpha`.

## Requirements

```
pip install pandas scikit-learn matplotlib
```
