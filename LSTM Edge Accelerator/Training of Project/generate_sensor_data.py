import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# MARS-LIKE SYNTHETIC SENSOR DATASET
# ============================================================
#
# Sensors:
#   1. Temperature       °C
#   2. Solar radiation   W/m²
#   3. Relative humidity %
#   4. CO2 concentration  ppm
#
# ML task:
#   Previous 10 hours -> predict next hour's 4 sensor values
#
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MARTIAN_SOLS = 668
HOURS_PER_SOL = 24.6597

# Approximate number of hourly samples in one Martian year
N_HOURS = int(MARTIAN_SOLS * HOURS_PER_SOL)

START_DATE = "2026-01-01"

SEQUENCE_LENGTH = 10


# ============================================================
# 1. TEMPERATURE
# ============================================================

# Mars has a very cold average surface temperature.
TEMP_MEAN = -63.0

# Strong day/night temperature variation.
TEMP_DAILY_AMPLITUDE = 42.0

# Seasonal variation.
TEMP_SEASONAL_AMPLITUDE = 15.0

TEMP_NOISE_STD = 3.0


# ============================================================
# 2. SOLAR RADIATION
# ============================================================

# Solar irradiance reaching Mars is lower than Earth.
SOLAR_MAX = 590.0

SOLAR_NOISE_STD = 5.0

# Probability of a dust-storm day.
DUST_STORM_PROB = 0.025

# Strong dust storms can severely reduce sunlight.
DUST_ATTENUATION_RANGE = (0.15, 0.55)


# ============================================================
# 3. HUMIDITY
# ============================================================

# Martian atmosphere is extremely dry.
HUMIDITY_BASE = 0.4

HUMIDITY_TEMP_COUPLING = -0.025

HUMIDITY_NOISE_STD = 0.08

HUMIDITY_MIN = 0.0
HUMIDITY_MAX = 5.0


# ============================================================
# 4. CO2 CONCENTRATION
# ============================================================

# Mars atmosphere is approximately 95% CO2.
# 95% = 950,000 ppm.
CO2_BASE = 950000.0

CO2_NOISE_STD = 500.0

# Slow atmospheric variation.
CO2_DRIFT_STD = 40.0

# Occasional local CO2 disturbances.
CO2_EVENT_PROB = 0.002

CO2_EVENT_MAGNITUDE = (2000, 10000)

CO2_EVENT_DECAY = 0.85


# ============================================================
# DATA GENERATOR
# ============================================================

def generate_mars_dataset(n_hours=N_HOURS, seed=SEED):

    rng = np.random.default_rng(seed)

    t = np.arange(n_hours)

    # Fractional Martian sol
    sol_time = (t % HOURS_PER_SOL) / HOURS_PER_SOL

    # Martian year position
    sol_number = t / HOURS_PER_SOL

    # --------------------------------------------------------
    # SOLAR POSITION
    # --------------------------------------------------------

    # Sunrise approximately 06:00
    # Sunset approximately 18:00
    #
    # Solar radiation is approximately zero during night.

    solar_angle = np.pi * (
        (sol_time - 0.25) / 0.5
    )

    daylight = np.clip(
        np.sin(solar_angle),
        0,
        None
    )

    # --------------------------------------------------------
    # SEASONAL EFFECT
    # --------------------------------------------------------

    seasonal_factor = (
        1.0
        + 0.18 *
        np.sin(
            2 * np.pi *
            sol_number / MARTIAN_SOLS
        )
    )

    # --------------------------------------------------------
    # DUST STORMS
    # --------------------------------------------------------

    n_sols = MARTIAN_SOLS + 2

    dust_storm = (
        rng.random(n_sols)
        < DUST_STORM_PROB
    )

    dust_attenuation = np.ones(n_sols)

    dust_attenuation[dust_storm] = (
        rng.uniform(
            DUST_ATTENUATION_RANGE[0],
            DUST_ATTENUATION_RANGE[1],
            dust_storm.sum()
        )
    )

    sol_index = np.minimum(
        np.floor(sol_number).astype(int),
        n_sols - 1
    )

    dust_factor = dust_attenuation[sol_index]

    # --------------------------------------------------------
    # SOLAR RADIATION
    # --------------------------------------------------------

    solar_radiation = (
        SOLAR_MAX
        * daylight
        * seasonal_factor
        * dust_factor
    )

    solar_radiation += rng.normal(
        0,
        SOLAR_NOISE_STD,
        n_hours
    )

    solar_radiation = np.clip(
        solar_radiation,
        0,
        None
    )

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    # Strong Martian day/night temperature cycle.
    daily_temperature = (
        TEMP_DAILY_AMPLITUDE
        * np.sin(
            2 * np.pi *
            (sol_time - 0.38)
        )
    )

    # Seasonal temperature variation.
    seasonal_temperature = (
        TEMP_SEASONAL_AMPLITUDE
        * np.sin(
            2 * np.pi *
            (sol_number - 170)
            / MARTIAN_SOLS
        )
    )

    # Solar heating.
    solar_heating = (
        0.025 *
        solar_radiation
    )

    # Slowly changing weather/dust component.
    weather = np.zeros(n_hours)

    for i in range(1, n_hours):

        weather[i] = (
            0.985 * weather[i - 1]
            + rng.normal(0, 0.35)
        )

    temperature = (
        TEMP_MEAN
        + daily_temperature
        + seasonal_temperature
        + solar_heating
        + weather
        + rng.normal(
            0,
            TEMP_NOISE_STD,
            n_hours
        )
    )

    # --------------------------------------------------------
    # HUMIDITY
    # --------------------------------------------------------

    # Extremely low humidity on Mars.
    #
    # Cooler temperatures -> slightly higher RH.
    # Warmer temperatures -> lower RH.

    humidity_weather = np.zeros(n_hours)

    for i in range(1, n_hours):

        humidity_weather[i] = (
            0.97 * humidity_weather[i - 1]
            + rng.normal(0, 0.025)
        )

    humidity = (
        HUMIDITY_BASE
        + HUMIDITY_TEMP_COUPLING
        * (temperature - TEMP_MEAN)

        + 0.15 *
        np.cos(
            2 * np.pi *
            (sol_time - 0.2)
        )

        + humidity_weather

        + rng.normal(
            0,
            HUMIDITY_NOISE_STD,
            n_hours
        )
    )

    humidity = np.clip(
        humidity,
        HUMIDITY_MIN,
        HUMIDITY_MAX
    )

    # --------------------------------------------------------
    # CO2 CONCENTRATION
    # --------------------------------------------------------

    co2 = np.zeros(n_hours)

    co2[0] = CO2_BASE

    active_event = 0.0

    for i in range(1, n_hours):

        # Slow random atmospheric drift
        drift = rng.normal(
            0,
            CO2_DRIFT_STD
        )

        # Daily atmospheric variation
        daily_variation = (
            1000 *
            np.cos(
                2 * np.pi *
                (sol_time[i] - 0.4)
            )
        )

        # Slowly decay local event
        active_event *= CO2_EVENT_DECAY

        # Occasional local concentration event
        if rng.random() < CO2_EVENT_PROB:

            active_event += rng.uniform(
                CO2_EVENT_MAGNITUDE[0],
                CO2_EVENT_MAGNITUDE[1]
            )

        target = (
            CO2_BASE
            + daily_variation
        )

        # Mean-reverting process
        co2[i] = (
            0.96 * co2[i - 1]
            + 0.04 * target
            + drift
            + active_event * 0.05
            + rng.normal(
                0,
                CO2_NOISE_STD
            )
        )

    co2 = np.clip(
        co2,
        900000,
        1000000
    )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamps = pd.date_range(
        START_DATE,
        periods=n_hours,
        freq="h"
    )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame({

        "timestamp":
            timestamps,

        "temperature_C":
            temperature,

        "solar_radiation_Wm2":
            solar_radiation,

        "humidity_pct":
            humidity,

        "co2_concentration_ppm":
            co2
    })

    # Reasonable precision
    df["temperature_C"] = (
        df["temperature_C"].round(3)
    )

    df["solar_radiation_Wm2"] = (
        df["solar_radiation_Wm2"].round(3)
    )

    df["humidity_pct"] = (
        df["humidity_pct"].round(4)
    )

    df["co2_concentration_ppm"] = (
        df["co2_concentration_ppm"].round(2)
    )

    return df


# ============================================================
# GENERATE DATA
# ============================================================

if __name__ == "__main__":

    df = generate_mars_dataset()

    output_dir = Path(".")

    csv_path = (
        output_dir /
        "mars_synthetic_sensor_data.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print("=" * 60)
    print("MARS SYNTHETIC SENSOR DATASET")
    print("=" * 60)

    print(
        f"\nGenerated {len(df):,} hourly records"
    )

    print(
        f"Approximately {MARTIAN_SOLS} Martian sols"
    )

    print(
        f"\nSaved to:\n{csv_path.resolve()}"
    )

    print("\nFirst 10 records:")
    print(
        df.head(10).to_string(
            index=False
        )
    )

    print("\nStatistics:")
    print(
        df.describe().round(2).to_string()
    )
# ============================================================
# NORMALIZE MARS DATA
# ============================================================

FEATURES = [
    "temperature_C",
    "solar_radiation_Wm2",
    "humidity_pct",
    "co2_concentration_ppm"
]

# ------------------------------------------------------------
# Chronological train / validation / test split
# ------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

n = len(df)

train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]


# ------------------------------------------------------------
# Calculate normalization parameters ONLY from training data
# ------------------------------------------------------------

mean = train_df[FEATURES].mean()
std = train_df[FEATURES].std()

# Prevent division by zero
std = std.replace(0, 1)


# ------------------------------------------------------------
# Normalize
# ------------------------------------------------------------

df_normalized = df.copy()

df_normalized[FEATURES] = (
    df[FEATURES] - mean
) / std


# ------------------------------------------------------------
# Save normalized dataset
# ------------------------------------------------------------

normalized_path = (
    output_dir /
    "mars_synthetic_sensor_data_normalized.csv"
)

df_normalized.to_csv(
    normalized_path,
    index=False
)


# ------------------------------------------------------------
# Save normalization parameters
# ------------------------------------------------------------

normalization_params = pd.DataFrame({
    "sensor": FEATURES,
    "mean": mean.values,
    "std": std.values
})

params_path = (
    output_dir /
    "mars_normalization_parameters.csv"
)

normalization_params.to_csv(
    params_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("NORMALIZATION")
print("=" * 60)

print("\nNormalization parameters:")
print(
    normalization_params.to_string(
        index=False
    )
)

print("\nNormalized data:")
print(
    df_normalized[
        ["timestamp"] + FEATURES
    ].head(10).to_string(
        index=False
    )
)

print("\nNormalized statistics:")
print(
    df_normalized[
        FEATURES
    ].describe().round(3).to_string()
)

print("\nSaved:")
print(normalized_path)
print(params_path)