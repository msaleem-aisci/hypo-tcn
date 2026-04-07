# Hypo-TCN

## 1. Abstract
Hypo-TCN is a deep learning architecture designed to predict hypotensive events in intensive care units (ICUs) prior to clinical manifestation. Moving beyond reactive bedside alarms, this system forecasts critical drops in Mean Arterial Pressure (MAP < 65 mmHg) across three distinct temporal horizons: 3-hour, 6-hour, and 12-hour windows. By providing early warning alerts, the model aims to mitigate clinical alarm fatigue and facilitate proactive interventions such as fluid resuscitation or vasopressor administration.

## 2. Dataset and Preprocessing
The model is trained on the PhysioNet 2019 critical care dataset. The pipeline handles extreme class imbalance and irregular sampling rates inherent to Electronic Health Records (EHR).

* **Informative Missingness:** Missing clinical variables are not simply discarded or imputed. The architecture utilizes "Missingness Masks" (binary indicators) to capture the clinical intuition that the frequency of lab orders correlates with patient stability.
* **Temporal Forward-Filling:** Variables are forward-filled at the patient level to simulate a real-time monitoring environment, with residual gaps filled using standard physiological baselines.
* **Windowing:** Data is structured into 12-hour sliding windows to capture local temporal dynamics.

## 3. Model Architecture
The architecture processes 22 input features (physiological signals, demographic data, and missingness masks) through a dual-mechanism temporal network.

* **Residual Causal TCN:** The core temporal feature extractor utilizes 1D Causal Convolutions. The strict causal padding ensures Zero Future Leakage, guaranteeing the model only relies on historical data at step $t$ to predict $t+n$. Residual connections are employed to stabilize gradient flow across deep layers.
* **Multi-Head Self-Attention:** TCN outputs are passed into a Multi-Head Self-Attention block to learn global dependencies across the input window, allowing the network to dynamically weigh specific hours of the sequence that are highly predictive of an impending crash.

## 4. Objective Function
Given the extreme rarity of hypotensive crashes compared to stable periods, standard Cross-Entropy optimization leads to high-accuracy, low-recall models. Hypo-TCN minimizes a Weighted Focal Loss:

$$
FL(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)
$$

This objective forces the network to focus gradient updates on hard-to-predict minority instances (crashes) while down-weighting the loss contribution from the vast majority of stable, easily classified temporal windows.

## 5. Validation and Results
The dataset is split using a strict Patient-Level Group Shuffle Split to prevent data leakage. The model achieves strong discriminative performance, validating its clinical utility in differentiating true emergencies from standard physiological variance.

* **Evaluation Metric:** Area Under the Receiver Operating Characteristic Curve (AUROC).
* **Performance:** The architecture achieves a peak AUROC of **0.896**, maintaining robust predictive power across the 3-hour, 6-hour, and 12-hour forecasting horizons.
