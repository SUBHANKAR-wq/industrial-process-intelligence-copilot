# Industrial Process Intelligence Copilot  
### Real-Time Anomaly Detection & Agentic AI Decision System

An end-to-end **industrial monitoring and decision-support system** that combines:

- Unsupervised anomaly detection (Autoencoder)
- Real-time sensor monitoring
- Event-driven agent orchestration (LLM Agents)
- Root cause analysis & operational recommendations
- Interactive dashboard visualization

This project simulates a real industrial environment using both **synthetic sensor data** and **NASA C-MAPSS FD001** data to build a scalable predictive maintenance pipeline.

---

## Project Overview

Industrial systems generate continuous multi-sensor data, but traditional monitoring systems rely on static thresholds or labeled failures.

This project solves that by building a system that:

- Detects anomalies without labeled data
- Identifies dominant sensors causing failure
- Generates probable root causes using AI agents
- Recommends operational actions automatically
- Works in a real-time streaming simulation

---

## Core Features

### 🔹 Real-Time Monitoring Pipeline
- Continuous sensor stream processing
- Timestamp-based simulation
- Reconstruction error tracking
- State detection:
  - Normal
  - Drift
  - Anomaly

---

### 🔹 Autoencoder-Based Anomaly Detection
- Built using TensorFlow/Keras
- Trained on normal behavior only
- Uses reconstruction error as anomaly signal
- Sensor-wise reconstruction error attribution

---

### 🔹 Event-Driven Agentic AI System

Agents are triggered **only during confirmed anomalies**:

1. **Sensor Analyst Agent**
   - Finds dominant sensors
   - Calculates severity and confidence

2. **Root Cause Agent**
   - Uses LLM reasoning
   - Generates probable physical causes
   - Suggests inspection focus

3. **Optimization Agent**
   - Recommends action:
     - monitor
     - schedule maintenance
     - shutdown

---


## System Architecture

            Sensor Stream (Synthetic + NASA)
                        ↓
                Data Pipeline Layer
                        ↓
                Autoencoder Inference
                        ↓
            Reconstruction Error Engine
                        ↓
                State Classification
            (Normal / Drift / Anomaly)
                        ↓
                    Event Gate
                        ↓
            Agent Orchestration Layer
