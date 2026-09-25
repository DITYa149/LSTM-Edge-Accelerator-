import numpy as np


# ============================================================
# ENVIRONMENTAL CONDITION STATES
# ============================================================

STABLE = 0
WORSENING = 1
HAZARDOUS = 2
BARREN = 3


STATE_NAMES = {
    STABLE: "STABLE",
    WORSENING: "WORSENING",
    HAZARDOUS: "HAZARDOUS",
    BARREN: "BARREN"
}


# ============================================================
# THRESHOLDS
# ------------------------------------------------------------
# IMPORTANT:
# These thresholds are in NORMALIZED units.
# They must be tuned using your Mars-like dataset.
# ============================================================

THRESH_TEMP_WARNING = 1.0
THRESH_TEMP_HAZARD = 2.0
THRESH_TEMP_BARREN = 3.0

THRESH_SOLAR_WARNING = 1.0
THRESH_SOLAR_HAZARD = 2.0
THRESH_SOLAR_BARREN = 3.0

THRESH_HUMIDITY_WARNING = 1.0
THRESH_HUMIDITY_HAZARD = 2.0
THRESH_HUMIDITY_BARREN = 3.0

THRESH_GAS_WARNING = 1.0
THRESH_GAS_HAZARD = 2.0
THRESH_GAS_BARREN = 3.0


# ============================================================
# TREND THRESHOLDS
# ============================================================

THRESH_TEMP_DROP = 1.0
THRESH_SOLAR_DROP = 1.0
THRESH_GAS_RISE = 1.0


# ============================================================
# CRISP LOGIC FUNCTION
# ============================================================

def classify_environment(input_window, prediction):
    """
    Classify the predicted environmental condition.

    Parameters
    ----------
    input_window : numpy array
        Shape: (10, 4)

        Feature order:
        [temperature, solar, humidity, gas]

    prediction : numpy array
        Shape: (4,)

        Next-hour predicted values:
        [temperature, solar, humidity, gas]

    Returns
    -------
    state : int
        0 = STABLE
        1 = WORSENING
        2 = HAZARDOUS
        3 = BARREN

    state_name : str
        Human-readable state

    flags : dict
        Internal crisp decisions
    """

    # --------------------------------------------------------
    # Take oldest historical point
    # --------------------------------------------------------

    previous = input_window[0]

    previous_temp = previous[0]
    previous_solar = previous[1]
    previous_humidity = previous[2]
    previous_gas = previous[3]


    # --------------------------------------------------------
    # Predicted next-hour values
    # --------------------------------------------------------

    predicted_temp = prediction[0]
    predicted_solar = prediction[1]
    predicted_humidity = prediction[2]
    predicted_gas = prediction[3]


    # ========================================================
    # TREND CALCULATIONS
    # ========================================================

    temp_drop = previous_temp - predicted_temp

    solar_drop = previous_solar - predicted_solar

    gas_rise = predicted_gas - previous_gas


    # ========================================================
    # TREND FLAGS
    # ========================================================

    sharp_cooling = temp_drop > THRESH_TEMP_DROP

    solar_dimming = solar_drop > THRESH_SOLAR_DROP

    gas_buildup = gas_rise > THRESH_GAS_RISE


    # ========================================================
    # SEVERITY FLAGS
    # ========================================================

    temperature_hazard = (
        predicted_temp < -THRESH_TEMP_HAZARD
    )

    temperature_barren = (
        predicted_temp < -THRESH_TEMP_BARREN
    )


    solar_hazard = (
        predicted_solar < -THRESH_SOLAR_HAZARD
    )

    solar_barren = (
        predicted_solar < -THRESH_SOLAR_BARREN
    )


    humidity_hazard = (
        abs(predicted_humidity) > THRESH_HUMIDITY_HAZARD
    )

    humidity_barren = (
        abs(predicted_humidity) > THRESH_HUMIDITY_BARREN
    )


    gas_hazard = (
        predicted_gas > THRESH_GAS_HAZARD
    )

    gas_barren = (
        predicted_gas > THRESH_GAS_BARREN
    )


    # ========================================================
    # COMBINE SENSOR SEVERITY
    # ========================================================

    hazard_condition = (
        temperature_hazard
        or solar_hazard
        or humidity_hazard
        or gas_hazard
    )


    barren_condition = (
        temperature_barren
        or solar_barren
        or humidity_barren
        or gas_barren
    )


    # ========================================================
    # FINAL CRISP DECISION
    #
    # Priority:
    #
    # BARREN
    #    ↓
    # HAZARDOUS
    #    ↓
    # WORSENING
    #    ↓
    # STABLE
    # ========================================================

    if barren_condition:

        state = BARREN

    elif hazard_condition:

        state = HAZARDOUS

    elif (
        sharp_cooling
        or solar_dimming
        or gas_buildup
    ):

        state = WORSENING

    else:

        state = STABLE


    # ========================================================
    # INTERNAL FLAGS
    # ========================================================

    flags = {

        # Trends
        "sharp_cooling": sharp_cooling,
        "solar_dimming": solar_dimming,
        "gas_buildup": gas_buildup,

        # Severity
        "temperature_hazard": temperature_hazard,
        "temperature_barren": temperature_barren,

        "solar_hazard": solar_hazard,
        "solar_barren": solar_barren,

        "humidity_hazard": humidity_hazard,
        "humidity_barren": humidity_barren,

        "gas_hazard": gas_hazard,
        "gas_barren": gas_barren,

        # Combined
        "hazard_condition": hazard_condition,
        "barren_condition": barren_condition
    }


    return state, STATE_NAMES[state], flags