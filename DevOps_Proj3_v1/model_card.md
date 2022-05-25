# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a lightgbm with the default scikit-learn hyperparameters.

## Intended Use

The model was created to predict if the salary of an individual will be more or less than 50k based on the
Census Income Data Set by UCI.

## Training Data

Census Income Data Set by UCI was splitted 80/20. Training was 80%.

## Evaluation Data

Census Income Data Set by UCI was splitted 80/20. Training was 20%.

## Metrics

The metrics used were accuray(~73%), roc(~76%).

## Ethical Considerations

Model bias should be examined further. Due to the features of the model it can result to potential gender,racial and ethnicity discrimination.

## Caveats and Recommendations

Hyper-parameter optimization and more feature engineering is recommended as it could possibly lead to better results.
