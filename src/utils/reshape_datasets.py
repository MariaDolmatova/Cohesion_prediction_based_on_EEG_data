import numpy as np
import pandas as pd
from .logger import get_logger

def reshape_input_eeg(input_file, output_file, has_part=True):
    logger = get_logger()
 
    df = pd.read_csv(input_file, header=None)

    if has_part:  # "part" means we dissect the data by time, this is how we name it in the CSV file
        df.columns = ["PairPart", "Band"] + [f"Electrode{i}" for i in range(1, 9)]  # Give new columns name

        pair_part = df["PairPart"].str.extract(r"(pair\s*\d+)\s+part\s*(\d+)", expand=True)
        df["Pair"] = pair_part[0]
        df["Part"] = pair_part[1].astype(int)
        df.drop("PairPart", axis=1, inplace=True)

    else:
        df.columns = ["Band"] + [f"Electrode{i}" for i in range(1, 9)]

        num_rows = len(df)
        bands_per_pair = 5
        if num_rows % bands_per_pair != 0:
            raise ValueError("check the rows!")  # add Pair column for the rows since there is no Pair column in this format
        df["Pair"] = ["pair " + str(i // bands_per_pair + 1) for i in range(len(df))]

    id_columns = ["Pair", "Band"]
    if has_part:
        id_columns.append("Part")

    melted = df.melt(
        id_vars=id_columns,
        value_vars=[f"Electrode{i}" for i in range(1, 9)],
        var_name="Electrode",
        value_name="Correlation"
    )

    # Pivot the table
    if has_part:
        pivoted = melted.pivot_table(
            index="Pair",
            columns=["Band", "Part", "Electrode"],
            values="Correlation",
            sort=False
        )

        pivoted.columns = [
            f"{band}_T{part}_{electrode}"
            for (band, part, electrode) in pivoted.columns
        ]
    else:
        pivoted = melted.pivot_table(
            index="Pair",
            columns=["Band", "Electrode"],
            values="Correlation",
            sort=False
        )

        pivoted.columns = [
            f"{band}_{electrode}"
            for (band, electrode) in pivoted.columns
        ]

    pivoted.reset_index(inplace=True)
    pivoted.drop(pivoted[pivoted["Pair"] == "pair 1"].index, inplace=True)
    pivoted_cleaned = pivoted.dropna()

    pivoted_cleaned.to_csv(output_file, index=False)
    logger.info(f'The reshape csv is done as {output_file}')

    return pivoted_cleaned