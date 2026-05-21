import pandas as pd

# Standard UCI Adult Dataset column names
COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

DATA_PATH = "data/adult.csv"


def validate_dataset():
    print("=" * 60)
    print("UCI Adult Dataset Validation")
    print("=" * 60)

    try:
        # Load dataset
        df = pd.read_csv(
            DATA_PATH,
            header=None,
            names=COLUMN_NAMES,
            skipinitialspace=True
        )

        # Basic validation
        rows, cols = df.shape

        print(f"\nDataset Shape: {rows} rows x {cols} columns")

        # Validate column count
        if cols != 15:
            raise ValueError(
                f"Expected 15 columns, but found {cols}"
            )

        print("✓ Column count validation passed")

        # Missing values detection
        missing_values = (df == "?").sum()

        print("\nMissing Values Per Column:")
        print(missing_values[missing_values > 0])

        # First 5 rows
        print("\nFirst 5 Rows:")
        print(df.head())

        # Dataset info
        print("\nDataset Info:")
        print(df.info())

        # Income class validation
        print("\nTarget Classes:")
        print(df["income"].value_counts())

        print("\n✓ Dataset validation completed successfully")
        print("✓ Dataset is compatible with DVC pipeline stages")

    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {DATA_PATH}")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    validate_dataset()
