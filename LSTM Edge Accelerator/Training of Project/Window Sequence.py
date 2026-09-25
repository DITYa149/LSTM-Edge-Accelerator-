"""
============================================================
CREATE WINDOWED SEQUENCES FOR LSTM
============================================================

LSTM TASK
---------

10 hours of history
        ↓
      LSTM
        ↓
Predict next hour

Input:
    X = (10 timesteps, 4 features)

Output:
    y = (4 features)

Sensors:
    0 = Temperature
    1 = Solar radiation
    2 = Humidity
    3 = CO2

============================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 10

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FEATURES = [
    "temperature_C",
    "solar_radiation_Wm2",
    "humidity_pct",
    "co2_concentration_ppm"
]

INPUT_DIR = Path(".")

NORMALIZED_DATA_FILE = (
    "mars_synthetic_sensor_data_normalized.csv"
)

OUTPUT_DIR = INPUT_DIR / "windowed_data"


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(data, seq_length=10):

    """
    Convert time-series data into sliding windows.

    Example:

        t0  t1  t2 ... t8  t9  t10
        └────── X ────────┘  └ y ┘

        X = [t0 ... t9]
        y = t10

    Parameters
    ----------
    data : np.ndarray
        Shape = (number_of_hours, 4)

    seq_length : int
        Number of historical timesteps.

    Returns
    -------
    X : np.ndarray
        Shape = (number_of_windows, seq_length, 4)

    y : np.ndarray
        Shape = (number_of_windows, 4)
    """

    if len(data) <= seq_length:

        raise ValueError(
            "Not enough data to create a sequence."
        )

    X = []
    y = []

    for i in range(
        len(data) - seq_length
    ):

        # 10 previous hours
        X.append(
            data[
                i:i + seq_length
            ]
        )

        # Next hour
        y.append(
            data[
                i + seq_length
            ]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    file_path = (
        INPUT_DIR /
        NORMALIZED_DATA_FILE
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"{file_path.resolve()}\n\n"
            "Make sure Normalise_Mars_data.py "
            "has been run first."
        )

    print(
        f"Loading:\n"
        f"{file_path.resolve()}"
    )

    df = pd.read_csv(file_path)

    # Check required columns
    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = df[
        FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    return df, data


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("MARS LSTM WINDOW GENERATOR")
    print("=" * 65)


    # --------------------------------------------------------
    # Load normalized data
    # --------------------------------------------------------

    df, data = load_data()

    n_samples = len(data)

    print(
        f"\nTotal hourly samples: "
        f"{n_samples:,}"
    )

    print(
        f"Number of features: "
        f"{data.shape[1]}"
    )


    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    # IMPORTANT:
    #
    # Split RAW time-series BEFORE windowing.
    #
    # This prevents overlapping train/validation/test
    # sequences from sharing almost identical data.

    train_end = int(
        n_samples *
        TRAIN_RATIO
    )

    val_end = int(
        n_samples *
        (TRAIN_RATIO + VAL_RATIO)
    )

    train_data = data[
        :train_end
    ]

    val_data = data[
        train_end:val_end
    ]

    test_data = data[
        val_end:
    ]


    print("\nRAW DATA SPLIT")
    print("-" * 65)

    print(
        f"Train: "
        f"{len(train_data):,} samples"
    )

    print(
        f"Validation: "
        f"{len(val_data):,} samples"
    )

    print(
        f"Test: "
        f"{len(test_data):,} samples"
    )


    # ========================================================
    # CREATE WINDOWS
    # ========================================================

    print(
        f"\nCreating "
        f"{SEQUENCE_LENGTH}-hour windows..."
    )

    X_train, y_train = create_sequences(
        train_data,
        SEQUENCE_LENGTH
    )

    X_val, y_val = create_sequences(
        val_data,
        SEQUENCE_LENGTH
    )

    X_test, y_test = create_sequences(
        test_data,
        SEQUENCE_LENGTH
    )


    # ========================================================
    # PRINT SHAPES
    # ========================================================

    print("\nWINDOWED DATA")
    print("-" * 65)

    print(
        f"X_train : {X_train.shape}"
    )

    print(
        f"y_train : {y_train.shape}"
    )

    print(
        f"X_val   : {X_val.shape}"
    )

    print(
        f"y_val   : {y_val.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    print(
        f"y_test  : {y_test.shape}"
    )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # SAVE NUMPY DATA
    # ========================================================

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "X_val.npy",
        X_val
    )

    np.save(
        OUTPUT_DIR / "y_val.npy",
        y_val
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test
    )


    # ========================================================
    # SAVE SAMPLE FOR HUMAN INSPECTION
    # ========================================================

    N_DISPLAY = min(
        5,
        len(X_train)
    )

    sample_X = X_train[
        :N_DISPLAY
    ]

    sample_y = y_train[
        :N_DISPLAY
    ]

    # Flatten each 10×4 input
    sample_X_flat = (
        sample_X.reshape(
            N_DISPLAY,
            SEQUENCE_LENGTH * len(FEATURES)
        )
    )

    columns = []

    for t in range(
        SEQUENCE_LENGTH
    ):

        for feature in FEATURES:

            columns.append(
                f"X_t{t}_{feature}"
            )

    columns += [
        f"y_{feature}"
        for feature in FEATURES
    ]

    sample_data = np.hstack(
        [
            sample_X_flat,
            sample_y
        ]
    )

    sample_df = pd.DataFrame(
        sample_data,
        columns=columns
    )

    sample_df.to_csv(
        OUTPUT_DIR /
        "sample_sequences.csv",
        index=False
    )


    # ========================================================
    # SAVE DATASET INFORMATION
    # ========================================================

    info = pd.DataFrame({

        "dataset": [
            "train",
            "validation",
            "test"
        ],

        "samples": [
            len(X_train),
            len(X_val),
            len(X_test)
        ],

        "sequence_length": [
            SEQUENCE_LENGTH
        ] * 3,

        "input_features": [
            len(FEATURES)
        ] * 3,

        "target_features": [
            len(FEATURES)
        ] * 3
    })

    info.to_csv(
        OUTPUT_DIR /
        "dataset_info.csv",
        index=False
    )


    # ========================================================
    # VERIFY FIRST WINDOW
    # ========================================================

    print("\n" + "=" * 65)
    print("FIRST TRAINING WINDOW")
    print("=" * 65)

    print(
        "\nX_train[0] = 10 hours:"
    )

    print(
        "\n"
        "Time | Temperature | Solar | Humidity | CO2"
    )

    print("-" * 65)

    for t in range(
        SEQUENCE_LENGTH
    ):

        print(
            f"{t:4d} | "
            f"{X_train[0,t,0]:11.4f} | "
            f"{X_train[0,t,1]:7.4f} | "
            f"{X_train[0,t,2]:8.4f} | "
            f"{X_train[0,t,3]:8.4f}"
        )

    print("\nTarget y_train[0]:")

    print(
        f"Temperature : "
        f"{y_train[0,0]:.4f}"
    )

    print(
        f"Solar       : "
        f"{y_train[0,1]:.4f}"
    )

    print(
        f"Humidity    : "
        f"{y_train[0,2]:.4f}"
    )

    print(
        f"CO2         : "
        f"{y_train[0,3]:.4f}"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("WINDOWING COMPLETE")
    print("=" * 65)

    print(
        "\nLSTM INPUT:"
    )

    print(
        "10 timesteps × 4 features"
    )

    print(
        "\nLSTM OUTPUT:"
    )

    print(
        "4 predicted sensor values"
    )

    print(
        "\nFiles saved in:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print("=" * 65)