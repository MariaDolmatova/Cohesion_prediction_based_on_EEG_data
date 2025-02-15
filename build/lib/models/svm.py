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
  
  set_random_seed()
  logger = get_logger()
  global results_df

  df_X = pd.read_csv(input)
  df_Y = pd.read_csv(output)

  df_X.drop(columns=['Pair'], inplace=True, errors='ignore')
  df_X = df_X.dropna()

  X = df_X.values
  y = df_Y.iloc[:, 0].values

  pipeline = Pipeline([
      ('scaler', StandardScaler()),
      ('select', SelectKBest(score_func=f_classif)),
      ('svc', SVC(probability=True, class_weight='balanced'))
  ])

  config = configparser.ConfigParser()
  config.read('config/svm_config.ini')  

  def parse_list(value):
      items = value.split(", ")  
      parsed_items = []
      for x in items:
          try:
              num = float(x)  
              parsed_items.append(int(num) if num.is_integer() else num)  
          except ValueError:
              parsed_items.append(x)  
      return parsed_items

  select_k = parse_list(config.get('SVM', 'select_k'))
  C = parse_list(config.get('SVM', 'C'))
  kernel = parse_list(config.get('SVM', 'kernel'))
  gamma = parse_list(config.get('SVM', 'gamma'))
  degree = parse_list(config.get('SVM', 'degree'))

  param_grid = {
        'select__k': select_k,
        'svc__C': C,
        'svc__kernel': kernel,
        'svc__gamma': gamma,
        'svc__degree': degree
      }

  scoring = {
      'f1': make_scorer(f1_score)
  }

  grid_search = GridSearchCV(
      estimator=pipeline,
      param_grid=param_grid,
      scoring=scoring,
      refit='f1',
      cv=5,
  )

  grid_search.fit(X, y)

  best_model = grid_search.best_estimator_
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_
  results_df = pd.DataFrame(grid_search.cv_results_)

  logger.info('Best Parameters:', grid_search.best_params_)
  logger.info('Best F1 Score:', grid_search.best_score_)

  return best_model, best_params, best_score, results_df


def multi_datasets(datasets):

    summary_list = []
    logger = get_logger()

    for (input, output) in datasets:
        try:
            best_model, best_params, best_score = train_svm(input, output)
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
            logger.info(f'Check the {input}: {e}')

    summary_df = pd.DataFrame(summary_list)

    def number(series):
      return series.str.extract(r'(\d+)').fillna(0).astype(int)[0]

    summary_df = summary_df.sort_values(by='Dataset', key=number)

    return summary_df