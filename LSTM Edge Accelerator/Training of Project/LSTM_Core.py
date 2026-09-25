# ================================================================
# ORIGINAL SINGLE-HEAD LSTM
# ================================================================
#
# Architecture:
#
# Input (10, 4)
#      ↓
# Manual LSTM (16 hidden units)
#      ↓
# Final hidden state h_10
#      ↓
# Dense(16)
#      ↓
# PReLU
#      ↓
# Dropout(0.1) [training only]
#      ↓
# Dense(4)
#      ↓
# [Temperature, Solar Radiation, Humidity, CO2]
#
# Task:
# Multi-output regression
#
# Optimizer:
# Adam
#
# Loss:
# MSE
#
# Initialization:
# LSTM  -> Xavier
# Dense -> He
#
# ================================================================


import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path


# ================================================================
# 1. CONFIGURATION
# ================================================================

SEQUENCE_LENGTH = 10
INPUT_SIZE = 4
HIDDEN_SIZE = 16
DENSE_SIZE = 16
OUTPUT_SIZE = 4

LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1

EPOCHS = 100
BATCH_SIZE = 32

RANDOM_SEED = 42

INPUT_DIR = Path("windowed_data")
OUTPUT_DIR = Path("trained_model_original")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2. SENSOR NAMES
# ================================================================

FEATURE_NAMES = [
    "temperature_C",
    "solar_radiation_Wm2",
    "humidity_pct",
    "co2_concentration_ppm"
]


# ================================================================
# 3. RANDOM SEEDS
# ================================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ================================================================
# 4. INITIALIZATION FUNCTIONS
# ================================================================

def xavier_init(shape):
    """
    Xavier/Glorot uniform initialization.

    limit = sqrt(6 / (fan_in + fan_out))
    """

    fan_in = shape[0]
    fan_out = shape[1]

    limit = np.sqrt(6.0 / (fan_in + fan_out))

    values = np.random.uniform(
        low=-limit,
        high=limit,
        size=shape
    )

    return tf.Variable(
        values,
        dtype=tf.float32,
        trainable=True
    )


def he_init(shape):
    """
    He normal initialization.

    stddev = sqrt(2 / fan_in)
    """

    fan_in = shape[0]

    stddev = np.sqrt(2.0 / fan_in)

    values = np.random.normal(
        loc=0.0,
        scale=stddev,
        size=shape
    )

    return tf.Variable(
        values,
        dtype=tf.float32,
        trainable=True
    )


def zero_init(shape):
    """
    Zero initialization for biases.
    """

    return tf.Variable(
        tf.zeros(shape, dtype=tf.float32),
        trainable=True
    )


# ================================================================
# 5. CREATE MODEL PARAMETERS
# ================================================================

print("\nCreating model parameters...")


# ------------------------------------------------
# LSTM PARAMETERS
# ------------------------------------------------
#
# Four gates:
#
# [forget | input | candidate | output]
#
# Total gate size = 4 × HIDDEN_SIZE
# ------------------------------------------------

W_lstm = xavier_init(
    (INPUT_SIZE, 4 * HIDDEN_SIZE)
)

U_lstm = xavier_init(
    (HIDDEN_SIZE, 4 * HIDDEN_SIZE)
)

b_lstm = zero_init(
    (4 * HIDDEN_SIZE,)
)


# ------------------------------------------------
# DENSE LAYER
# ------------------------------------------------

W_dense = he_init(
    (HIDDEN_SIZE, DENSE_SIZE)
)

b_dense = zero_init(
    (DENSE_SIZE,)
)


# ------------------------------------------------
# PReLU
# ------------------------------------------------

alpha_prelu = tf.Variable(
    tf.fill(
        [DENSE_SIZE],
        tf.constant(0.25, dtype=tf.float32)
    ),
    trainable=True,
    name="alpha_prelu"
)


# ------------------------------------------------
# OUTPUT LAYER
# ------------------------------------------------

W_output = xavier_init(
    (DENSE_SIZE, OUTPUT_SIZE)
)

b_output = zero_init(
    (OUTPUT_SIZE,)
)


# ================================================================
# 6. TRAINABLE VARIABLES
# ================================================================

TRAINABLE_VARIABLES = [
    W_lstm,
    U_lstm,
    b_lstm,

    W_dense,
    b_dense,

    alpha_prelu,

    W_output,
    b_output
]


# ================================================================
# 7. PRINT MODEL INFORMATION
# ================================================================

def count_parameters():

    total = 0

    print("\nParameter shapes:")

    for variable in TRAINABLE_VARIABLES:

        count = int(np.prod(variable.shape))

        total += count

        print(
            f"{variable.name:20s} "
            f"{str(tuple(variable.shape)):15s} "
            f"{count}"
        )

    print("\nTotal trainable parameters:", total)

    return total


TOTAL_PARAMETERS = count_parameters()
# ================================================================
# SAVE / LOAD WEIGHTS
# ================================================================

def save_weights(filepath):

    np.savez(
        filepath,

        W_lstm=W_lstm.numpy(),
        U_lstm=U_lstm.numpy(),
        b_lstm=b_lstm.numpy(),

        W_dense=W_dense.numpy(),
        b_dense=b_dense.numpy(),

        alpha_prelu=alpha_prelu.numpy(),

        W_output=W_output.numpy(),
        b_output=b_output.numpy()
    )


def load_weights(filepath):

    data = np.load(filepath)

    W_lstm.assign(data["W_lstm"])
    U_lstm.assign(data["U_lstm"])
    b_lstm.assign(data["b_lstm"])

    W_dense.assign(data["W_dense"])
    b_dense.assign(data["b_dense"])

    alpha_prelu.assign(data["alpha_prelu"])

    W_output.assign(data["W_output"])
    b_output.assign(data["b_output"])


# ================================================================
# 8. ACTIVATION FUNCTIONS
# ================================================================

def sigmoid(x):
    return tf.math.sigmoid(x)


def tanh(x):
    return tf.math.tanh(x)


# ================================================================
# 9. MANUAL LSTM CELL
# ================================================================

def lstm_cell(x_t, h_prev, c_prev):

    """
    One timestep of the LSTM.

    x_t:
        (batch, INPUT_SIZE)

    h_prev:
        (batch, HIDDEN_SIZE)

    c_prev:
        (batch, HIDDEN_SIZE)

    Returns:
        h_t
        c_t
        f_t
        i_t
        g_t
        o_t
    """

    # ------------------------------------------------------------
    # Linear transformation
    # ------------------------------------------------------------

    gates = (
        tf.matmul(x_t, W_lstm)
        +
        tf.matmul(h_prev, U_lstm)
        +
        b_lstm
    )


    # ------------------------------------------------------------
    # Split the four gates
    # ------------------------------------------------------------

    f_t, i_t, g_t, o_t = tf.split(
        gates,
        num_or_size_splits=4,
        axis=1
    )


    # ------------------------------------------------------------
    # Gate activations
    # ------------------------------------------------------------

    f_t = sigmoid(f_t)

    i_t = sigmoid(i_t)

    g_t = tanh(g_t)

    o_t = sigmoid(o_t)


    # ------------------------------------------------------------
    # Cell state
    # ------------------------------------------------------------

    c_t = (
        f_t * c_prev
        +
        i_t * g_t
    )


    # ------------------------------------------------------------
    # Hidden state
    # ------------------------------------------------------------

    h_t = (
        o_t * tanh(c_t)
    )


    return h_t, c_t, f_t, i_t, g_t, o_t


# ================================================================
# 10. PRELU
# ================================================================

def prelu(x):

    """
    PReLU:

        PReLU(x) =
            x              if x >= 0
            alpha * x      if x < 0
    """

    return tf.maximum(
        x,
        tf.constant(0.0, dtype=tf.float32)
    ) + alpha_prelu * tf.minimum(
        x,
        tf.constant(0.0, dtype=tf.float32)
    )


# ================================================================
# 11. DROPOUT
# ================================================================

def apply_dropout(x, training):

    """
    Dropout is used only during training.

    During inference:
        training = False
        dropout is disabled.
    """

    if training:

        return tf.nn.dropout(
            x,
            rate=DROPOUT_RATE
        )

    return x


# ================================================================
# 12. FORWARD PASS
# ================================================================

@tf.function
def forward_pass(X, training=False):

    """
    X shape:

        (batch, 10, 4)

    Output:

        (batch, 4)
    """

    batch_size = tf.shape(X)[0]


    # ------------------------------------------------------------
    # Initial hidden and cell states
    # ------------------------------------------------------------

    h = tf.zeros(
        (batch_size, HIDDEN_SIZE),
        dtype=tf.float32
    )

    c = tf.zeros(
        (batch_size, HIDDEN_SIZE),
        dtype=tf.float32
    )


    # ------------------------------------------------------------
    # Process sequence
    # ------------------------------------------------------------

    for t in range(SEQUENCE_LENGTH):

        x_t = X[:, t, :]

        h, c, _, _, _, _ = lstm_cell(
            x_t,
            h,
            c
        )


    # ------------------------------------------------------------
    # h is now h_10
    # ------------------------------------------------------------

    dense_output = (
        tf.matmul(h, W_dense)
        +
        b_dense
    )


    # ------------------------------------------------------------
    # PReLU
    # ------------------------------------------------------------

    activated = prelu(
        dense_output
    )


    # ------------------------------------------------------------
    # Dropout
    # ------------------------------------------------------------

    dropped = apply_dropout(
        activated,
        training
    )


    # ------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------

    output = (
        tf.matmul(dropped, W_output)
        +
        b_output
    )


    return output


# ================================================================
# 13. MSE LOSS
# ================================================================

def mse_loss(y_true, y_pred):

    error = y_true - y_pred

    squared_error = tf.square(error)

    return tf.reduce_mean(
        squared_error
    )


# ================================================================
# 14. ADAM OPTIMIZER
# ================================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


# ================================================================
# 15. TRAINING STEP
# ================================================================

@tf.function
def train_step(X_batch, y_batch):

    with tf.GradientTape() as tape:

        predictions = forward_pass(
            X_batch,
            training=True
        )

        loss = mse_loss(
            y_batch,
            predictions
        )


    gradients = tape.gradient(
        loss,
        TRAINABLE_VARIABLES
    )


    optimizer.apply_gradients(
        zip(
            gradients,
            TRAINABLE_VARIABLES
        )
    )


    return loss


# ================================================================
# 16. EVALUATION
# ================================================================

@tf.function
def evaluate_loss(X, y):

    predictions = forward_pass(
        X,
        training=False
    )

    return mse_loss(
        y,
        predictions
    )


# ================================================================
# 17. LOAD DATA
# ================================================================

print("\nLoading dataset...")


X_train = np.load(
    INPUT_DIR / "X_train.npy"
).astype(np.float32)

y_train = np.load(
    INPUT_DIR / "y_train.npy"
).astype(np.float32)


X_val = np.load(
    INPUT_DIR / "X_val.npy"
).astype(np.float32)

y_val = np.load(
    INPUT_DIR / "y_val.npy"
).astype(np.float32)


X_test = np.load(
    INPUT_DIR / "X_test.npy"
).astype(np.float32)

y_test = np.load(
    INPUT_DIR / "y_test.npy"
).astype(np.float32)


print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:  ", X_val.shape)
print("y_val:  ", y_val.shape)

print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)


# ================================================================
# 18. CHECK DATA SHAPES
# ================================================================

assert X_train.ndim == 3
assert X_val.ndim == 3
assert X_test.ndim == 3

assert X_train.shape[1] == SEQUENCE_LENGTH
assert X_train.shape[2] == INPUT_SIZE

assert y_train.shape[1] == OUTPUT_SIZE


# ================================================================
# 19. TRAINING HISTORY
# ================================================================

history = []

best_val_loss = float("inf")

best_epoch = 0


# ================================================================
# 20. TRAINING LOOP
# ================================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


num_train_samples = X_train.shape[0]


for epoch in range(1, EPOCHS + 1):


    # ------------------------------------------------------------
    # Shuffle training data
    # ------------------------------------------------------------

    indices = np.random.permutation(
        num_train_samples
    )

    X_train_shuffled = X_train[
        indices
    ]

    y_train_shuffled = y_train[
        indices
    ]


    # ------------------------------------------------------------
    # Mini-batch training
    # ------------------------------------------------------------

    batch_losses = []


    for start in range(
        0,
        num_train_samples,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            num_train_samples
        )


        X_batch = tf.convert_to_tensor(
            X_train_shuffled[start:end],
            dtype=tf.float32
        )

        y_batch = tf.convert_to_tensor(
            y_train_shuffled[start:end],
            dtype=tf.float32
        )


        loss = train_step(
            X_batch,
            y_batch
        )


        batch_losses.append(
            float(loss.numpy())
        )


    # ------------------------------------------------------------
    # Average training loss
    # ------------------------------------------------------------

    train_loss = float(
        np.mean(batch_losses)
    )


    # ------------------------------------------------------------
    # Validation loss
    # ------------------------------------------------------------

    val_loss = float(
        evaluate_loss(
            X_val,
            y_val
        ).numpy()
    )


    # ------------------------------------------------------------
    # Save best weights
    # ------------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch

        save_weights(
            OUTPUT_DIR / "best_model_weights.npz"
        )


        best_marker = " <-- BEST"

    else:

        best_marker = ""


    # ------------------------------------------------------------
    # Store history
    # ------------------------------------------------------------

    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "val_loss": val_loss

    })


    # ------------------------------------------------------------
    # Print progress
    # ------------------------------------------------------------

    print(
        f"Epoch {epoch:3d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.8f} | "
        f"Val MSE: {val_loss:.8f}"
        f"{best_marker}"
    )


# ================================================================
# 21. LOAD BEST MODEL
# ================================================================

print("\n")
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

load_weights(
    OUTPUT_DIR / "best_model_weights.npz"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation MSE: {best_val_loss:.8f}"
)


# ================================================================
# 22. FINAL EVALUATION
# ================================================================
#
# IMPORTANT:
#
# Test data is used only here.
#
# It was NOT used for model selection.
# ================================================================

print("\n")
print("=" * 70)
print("FINAL EVALUATION")
print("=" * 70)


train_loss_final = float(
    evaluate_loss(
        X_train,
        y_train
    ).numpy()
)


val_loss_final = float(
    evaluate_loss(
        X_val,
        y_val
    ).numpy()
)


test_loss_final = float(
    evaluate_loss(
        X_test,
        y_test
    ).numpy()
)


print(
    f"\nFinal Train MSE: {train_loss_final:.8f}"
)

print(
    f"Final Validation MSE: {val_loss_final:.8f}"
)

print(
    f"Final Test MSE: {test_loss_final:.8f}"
)


# ================================================================
# 23. GENERATE TEST PREDICTIONS
# ================================================================

print("\nGenerating test predictions...")


y_pred_test = forward_pass(
    tf.convert_to_tensor(
        X_test,
        dtype=tf.float32
    ),
    training=False
).numpy()


# ================================================================
# 24. CALCULATE METRICS
# ================================================================

errors = y_test - y_pred_test


mse_overall = np.mean(
    np.square(errors)
)


rmse_overall = np.sqrt(
    mse_overall
)


mae_overall = np.mean(
    np.abs(errors)
)


print("\nOverall test metrics:")

print(
    f"MSE  : {mse_overall:.8f}"
)

print(
    f"RMSE : {rmse_overall:.8f}"
)

print(
    f"MAE  : {mae_overall:.8f}"
)


# ================================================================
# 25. PER-SENSOR METRICS
# ================================================================

print("\n")
print("=" * 70)
print("PER-SENSOR TEST METRICS")
print("=" * 70)


sensor_metrics = {}


for i, name in enumerate(FEATURE_NAMES):

    sensor_error = errors[:, i]


    sensor_mse = np.mean(
        np.square(sensor_error)
    )


    sensor_rmse = np.sqrt(
        sensor_mse
    )


    sensor_mae = np.mean(
        np.abs(sensor_error)
    )


    sensor_metrics[name] = {

        "MSE": float(sensor_mse),

        "RMSE": float(sensor_rmse),

        "MAE": float(sensor_mae)

    }


    print(
        f"\n{name}"
    )

    print(
        f"  MSE  = {sensor_mse:.8f}"
    )

    print(
        f"  RMSE = {sensor_rmse:.8f}"
    )

    print(
        f"  MAE  = {sensor_mae:.8f}"
    )


# ================================================================
# 26. SAVE TRAINING HISTORY
# ================================================================

history_file = (
    OUTPUT_DIR / "training_history.csv"
)


with open(
    history_file,
    "w"
) as f:

    f.write(
        "epoch,train_loss,val_loss\n"
    )

    for row in history:

        f.write(
            f"{row['epoch']},"
            f"{row['train_loss']},"
            f"{row['val_loss']}\n"
        )


print(
    f"\nSaved: {history_file}"
)


# ================================================================
# 27. SAVE FINAL MODEL WEIGHTS
# ================================================================

save_weights(
    OUTPUT_DIR / "model_weights.npz"
)


# ================================================================
# 28. SAVE INDIVIDUAL WEIGHTS
# ================================================================

np.save(
    OUTPUT_DIR / "W_lstm.npy",
    W_lstm.numpy()
)

np.save(
    OUTPUT_DIR / "U_lstm.npy",
    U_lstm.numpy()
)

np.save(
    OUTPUT_DIR / "b_lstm.npy",
    b_lstm.numpy()
)

np.save(
    OUTPUT_DIR / "W_dense.npy",
    W_dense.numpy()
)

np.save(
    OUTPUT_DIR / "b_dense.npy",
    b_dense.numpy()
)

np.save(
    OUTPUT_DIR / "alpha_prelu.npy",
    alpha_prelu.numpy()
)

np.save(
    OUTPUT_DIR / "W_output.npy",
    W_output.numpy()
)

np.save(
    OUTPUT_DIR / "b_output.npy",
    b_output.numpy()
)


# ================================================================
# 29. SAVE TEST PREDICTIONS
# ================================================================

np.savez(
    OUTPUT_DIR / "predictions.npz",

    X_test=X_test,

    y_test=y_test,

    y_pred_test=y_pred_test,

    errors=errors
)


# ================================================================
# 30. SAVE METADATA
# ================================================================

metadata = {

    "architecture": {

        "input_shape": [
            SEQUENCE_LENGTH,
            INPUT_SIZE
        ],

        "lstm_hidden_units": HIDDEN_SIZE,

        "dense_units": DENSE_SIZE,

        "output_units": OUTPUT_SIZE,

        "dropout_rate": DROPOUT_RATE

    },


    "lstm": {

        "gate_order": [
            "forget",
            "input",
            "candidate",
            "output"
        ],

        "forget_activation": "sigmoid",

        "input_activation": "sigmoid",

        "candidate_activation": "tanh",

        "output_activation": "sigmoid",

        "initial_state": "zeros"

    },


    "dense": {

        "activation": "PReLU",

        "initialization": "He"

    },


    "output": {

        "activation": "linear"

    },


    "optimizer": {

        "name": "Adam",

        "learning_rate": LEARNING_RATE

    },


    "loss": "MSE",


    "training": {

        "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "random_seed": RANDOM_SEED,

        "best_epoch": best_epoch

    },


    "parameters": {

        "total_trainable_parameters":
            TOTAL_PARAMETERS

    },


    "features": FEATURE_NAMES,


    "metrics": {

        "train_mse":
            train_loss_final,

        "validation_mse":
            val_loss_final,

        "test_mse":
            test_loss_final,

        "test_rmse":
            rmse_overall,

        "test_mae":
            mae_overall

    },

    "sensor_metrics":
        sensor_metrics

}


with open(
    OUTPUT_DIR / "model_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        default=lambda x: x.item()
    )

# ================================================================
# 31. SAVE WEIGHTS FUNCTION
# ================================================================

# NOTE:
# This function is defined here as well so that the code above
# remains easy to read. Python requires it to exist before the
# training loop actually executes, so in practice it is defined
# below only for reference.
#
# The executable version is handled by moving the function before
# the training loop.
# ================================================================


# ================================================================
# 32. FINISHED
# ================================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nModel directory:"
    f" {OUTPUT_DIR}"
)

print(
    "\nBest validation epoch:"
    f" {best_epoch}"
)

print(
    "\nBest validation MSE:"
    f" {best_val_loss:.8f}"
)

print(
    "\nFinal test MSE:"
    f" {test_loss_final:.8f}"
)

print("\nSaved files:")

print("  best_model_weights.npz")
print("  model_weights.npz")
print("  W_lstm.npy")
print("  U_lstm.npy")
print("  b_lstm.npy")
print("  W_dense.npy")
print("  b_dense.npy")
print("  alpha_prelu.npy")
print("  W_output.npy")
print("  b_output.npy")
print("  predictions.npz")
print("  training_history.csv")
print("  model_metadata.json")

print("\n")