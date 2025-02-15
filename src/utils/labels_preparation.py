import pandas as pd
from ..utils.logger import get_logger


def process_labels(input):
  """Creates 2 data frames out of a csv file: one with average questionnaire score per each
  pair and aother with binary values or achieved or not achieved cohesion with the threshold of 4.5
  """
  logger = get_logger()

  # Check if file is csv
  if not input.endswith(".csv"): 
      logger.error("File format error: Input file is not a CSV.")
      raise ValueError("Invalid file format. Please provide a CSV file.")

  try:
    df_label = pd.read_csv(input)
  except pd.errors.ParserError:
    logger.error("Failed to parse the CSV file.")
    raise ValueError("Error reading the CSV file. Please check its format.")
  
  # Check if there's a needed column
  if "Average cohesion score" not in df_label.columns:
    logger.error("Missing required column: 'Average cohesion score'.")
    raise KeyError("Missing column: 'Average cohesion score'")

  # Check if the column filled with num values
  if not pd.api.types.is_numeric_dtype(df_label['Average cohesion score']):
    logger.error("Non-numeric values found in 'Average cohesion score' column.")
    raise ValueError("Invalid data: 'Average cohesion score' must contain only numeric values.")

  # filling missing vals with mean 
  mean_value = df_label['Average cohesion score'].mean()
  df_label.loc[:, 'Average cohesion score'] = df_label['Average cohesion score'].fillna(mean_value)

  cohesion_score = df_label["Average cohesion score"].values

  # Check that each val has a pair
  if len(cohesion_score) % 2 != 0: 
   raise ValueError("Check the dataset, something wrong with rows - they are not even.") 

  # Reshape it into pairs
  cohesion_pair = cohesion_score.reshape(-1, 2)
  pair_mean = cohesion_pair.mean(axis=1)

  # Set arbitrary threshold for cohesion score
  # 1 = High Cohesive, 0 = Low Cohesive
  cohesion_binary_set = (pair_mean > 4.5).astype(int)

  out_cohesion = pd.DataFrame({
  'pair': range(1, int(len(cohesion_score) / 2 + 1)),
  'Labels': pair_mean,
  'Binary Labels': cohesion_binary_set
  })

  cohesion_binary = pd.DataFrame({
  'Labels': cohesion_binary_set
  })

  cohesion_binary.to_csv('data/labels.csv', index=False)
  logger.info(f"Labels processed")

  return cohesion_binary, out_cohesion
