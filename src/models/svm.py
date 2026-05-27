import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import configparser
from ..utils.random_seed import set_random_seed
from ..utils.logger import get_logger


def train_svm(input, output):
    """ Trains an SVM model using cross-validation and returns best parameters. """

    set_random_seed()
    logger = get_logger()
    logger.info(f"Starting SVM training on dataset: {input}")

    # Read CSV files
    try:
        df_X = pd.read_csv(input)
        df_Y = pd.read_csv(output)
    except Exception as e:
        logger.error(f"Error reading CSV files: {e}")
        raise ValueError("Error loading input data. Ensure valid CSV format.") 

    # Handle missing 'Pair' column (drop it if exists)
    df_X.drop(columns=['Pair'], inplace=True, errors='ignore')

    # Ensure there are no missing values
    df_X = df_X.dropna()

    # Convert data to NumPy arrays
    try:
        X = df_X.values
        y = df_Y.iloc[:, 0].values
    except Exception as e:
        logger.error(f"Error converting data to numpy arrays: {e}")
        raise ValueError("Dataset contains non-numeric values.")

    # Ensure at least 5 samples for cross-validation
    if len(y) < 5:
        logger.error("Not enough samples for 5-fold cross-validation.")
        raise ValueError("At least 5 samples are required for cross-validation.")

    # Building the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif)),
        ('svc', SVC(probability=True, class_weight='balanced'))
    ])

    # Load SVM hyperparameters from config file
    config = configparser.ConfigParser()
    config.read('config/svm_config.ini')

    # Parsing the list
    def parse_list(value):
        """ Convert config values to lists of numbers or strings """
        items = value.split(", ")
        parsed_items = []
        for x in items:
            try:
                num = float(x)
                parsed_items.append(int(num) if num.is_integer() else num)
            except ValueError:
                parsed_items.append(x)
        return parsed_items

    try:
        select_k = parse_list(config.get('SVM', 'select_k'))
        C = parse_list(config.get('SVM', 'C'))
        kernel = parse_list(config.get('SVM', 'kernel'))
        gamma = parse_list(config.get('SVM', 'gamma'))
        degree = parse_list(config.get('SVM', 'degree'))
    except Exception as e:
        logger.error(f"Invalid config values: {e}")
        raise ValueError("Config file contains invalid values.") # some error handling

    param_grid = {
        'select__k': select_k,
        'svc__C': C,
        'svc__kernel': kernel,
        'svc__gamma': gamma,
        'svc__degree': degree
    }

    # Perform GridSearchCV with 5-fold cross-validation
    scoring = {'f1': make_scorer(f1_score)}
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit='f1',
        cv=5
    )

    logger.info("Starting hyperparameter tuning with GridSearchCV...")
    grid_search.fit(X, y)

    # Store best model results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    results_df = pd.DataFrame(grid_search.cv_results_)

    logger.info(f"Best model: {best_model}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best F1 Score: {best_score}")

    return best_params, best_score, results_df

# Include separate function for multiple datasets handling
def multi_datasets(datasets):
    """ Runs SVM training on multiple datasets and returns summary results. """

    summary_list = []
    logger = get_logger()
    logger.info("Running SVM training on multiple datasets...")

    for (input, output) in datasets:
        try:
            best_params, best_score, results_df = train_svm(input, output)
            columns = {
                'Dataset': input,
                'K Value': best_params.get('select__k', None),
                'C': best_params.get('svc__C', None),
                'Gamma': best_params.get('svc__gamma', None),
                'Kernel': best_params.get('svc__kernel', None),
                'Best F1 Score': best_score
            }
            summary_list.append(columns)

        except Exception as e:
            logger.info(f"Skipping dataset {input} due to error: {e}")

    summary_df = pd.DataFrame(summary_list)

    # Ensure dataset sorting works correctly
    def number(series):
        return series.str.extract(r'(\d+)').fillna(0).astype(int)[0]

    summary_df = summary_df.sort_values(by='Dataset', key=number)

    return summary_df

