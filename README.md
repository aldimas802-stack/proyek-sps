## Anggota Kelompok 11:
## Dimas Al Faridzi 2042241128
## Faiz Dzikrulloh Akbar 2042241092

# 👃 Electronic Nose (E-Nose) System for Shampoo Aroma Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Arduino](https://img.shields.io/badge/Arduino-UNO%20R4-teal?style=flat&logo=arduino)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Overview

This project is a prototype of an **Electronic Nose (E-Nose)** designed to detect, visualize, and classify the aroma profiles of five different shampoo brands (**Dove, Sunsilk, Lifebuoy, Head & Shoulders, and Pantene**).

The system integrates an **Arduino UNO R4 WiFi** (for hardware control and sensor reading) with a **Python Desktop Application** (for real-time visualization, data logging, and Machine Learning classification).

### Key Features
* **Automated Sampling Cycle:** Uses a Finite State Machine (FSM) on Arduino to control airflow (Intake/Exhaust) automatically.
* **Real-Time Visualization:** Live plotting of sensor data (MQ-4, MQ-135, MQ-6, MQ-7).
* **Auto-Logging:** Automatically saves sensor data to **Edge Impulse-compatible CSV files** upon stopping the sampling process.
* **Local Classification:** Implements a **K-Nearest Neighbors (KNN)** algorithm locally to predict the shampoo brand based on sensor signatures.

---

## 🛠️ Hardware Architecture

### Components
1.  **Microcontroller:** Arduino UNO R4 WiFi
2.  **Gas Sensors:**
    * **MQ-4:** Methane/Natural Gas
    * **MQ-135:** Air Quality/Ammonia/Benzene/Alcohol
    * **MQ-6:** LPG/Isobutane/Propane
    * **MQ-7:** Carbon Monoxide (CO)
3.  **Actuators (Pneumatic System):**
    * 3x DC Motors/Pumps (Motor A: Intake, Motor B: Exhaust, Motor C: Auxiliary)
    * L298N Motor Driver

### Pin Configuration (Wiring)

| Component | Arduino Pin | Description |
| :--- | :--- | :--- |
| **MQ-4** | A0 | Analog Input |
| **MQ-135** | A1 | Analog Input |
| **MQ-6** | A2 | Analog Input |
| **MQ-7** | A3 | Analog Input |
| **Motor A** | 10 (PWM), 12, 13 | Intake Fan |
| **Motor B** | 11 (PWM), 8, 9 | Exhaust Fan |
| **Motor C** | 3 (PWM), 4, 5 | Aux Pump |

---

## 💻 Software & Dependencies

### 1. Arduino Firmware
The firmware implements a **Finite State Machine (FSM)** to manage the sampling lifecycle:
* **PRE-CONDITIONING (15s):** Heats up sensors, intake motor ON.
* **RAMP-UP (2s):** Gradually increases airflow.
* **HOLD (8s):** Stable sampling phase. **Data is sent to Python only during this phase.**
* **PURGE (15s):** Cleans the chamber (All motors ON, high speed).
* **RECOVERY (5s):** Idle state before next run.

### 2. Python Application
The desktop app is built using **CustomTkinter**.

**Required Libraries:**
To run the application, install the dependencies using pip:
```bash
pip install pyserial customtkinter matplotlib numpy pandas scikit-learn
