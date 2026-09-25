# LSTM-Edge-Accelerator-
A hardware-oriented Edge AI accelerator project that combines an LSTM neural network with a rule-based environmental classification block to analyze and predict conditions in a synthetic Mars-like environment.

The project starts with synthetic multi-sensor time-series data representing:

Temperature
Solar radiation
Humidity
CO₂/gas concentration

The sensor data is normalized and converted into 10-hour sliding-window sequences, which are used to train an LSTM-based prediction model. The trained model predicts the normalized values of the four environmental parameters for the next hour.

The neural-network architecture consists of:

10-hour × 4-sensor input
          ↓
      LSTM Trunk
          ↓
    Hidden State (hₜ)
          ↓
      Dense Layer
          ↓
        PReLU
          ↓
     4 Sensor Outputs
          ↓
  Crisp Rule-Based Logic
          ↓
 ┌────────┬────────────┬────────────┐
 │        │            │            │
STABLE  WORSENING   HAZARDOUS     BARREN
Environmental Classification

The final prediction is interpreted using a crisp rule-based decision block with four environmental states:

State	Meaning
STABLE	Environment is suitable for sustaining living organisms.
WORSENING	Environmental conditions have deteriorated and are continuing to become unfavorable.
HAZARDOUS	Conditions are harmful and difficult for living organisms, but life may still be sustained.
BARREN	Conditions are too harsh to sustain life.
Hardware Acceleration

The long-term objective is to convert the trained neural network into a fixed-point FPGA-oriented RTL implementation.

The planned hardware flow is:

Python / TensorFlow
        ↓
Trained LSTM
        ↓
Parameter Extraction
        ↓
Fixed-Point Quantization
        ↓
HEX / ROM Parameters
        ↓
Verilog RTL
        ↓
ModelSim Simulation
        ↓
Golden Reference Comparison

The hardware architecture is being developed around fundamental RTL building blocks such as:

LSTM gates
Multipliers
Adders
Accumulators
Registers
Multiplexers
Control FSM
Parameter ROM
Dense layer
PReLU
Crisp rule-based classifier

A golden reference model will be used to compare the fixed-point Python implementation against the Verilog RTL implementation during ModelSim verification.

Technologies

Software / AI

Python
NumPy
Pandas
TensorFlow
LSTM
PReLU
Fixed-point modeling

Digital / Hardware

Verilog HDL
RTL design
FPGA-oriented architecture
Fixed-point arithmetic
ModelSim simulation
HEX/ROM parameter storage
Project Objective

The goal is to demonstrate the complete transition from a trained neural-network model to a hardware-oriented Edge AI accelerator, while maintaining numerical agreement between the Python reference model and the eventual RTL implementation
