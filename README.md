# Race-Agnostic Age Classifier
## Project Overview

This project implements a **privacy-preserving race-agnostic age classification system** for retail analytics. 
The goal is to estimate visitor age groups from facial images while ensuring that no identity recognition or personal data storage occurs.

The system focuses on **responsible AI development**, including fairness analysis, explainable AI (Grad-CAM), and privacy-by-design principles aligned with GDPR and the EU AI Act.

## Features

- Age group classification using deep learning
- Privacy-preserving design (no facial recognition)
- Fairness analysis across demographic groups
- Explainable AI using Grad-CAM
- Confusion matrix and model evaluation
- Responsible AI considerations (GDPR & EU AI Act)

## Dataset

Two datasets were explored in this project:

### UTKFace
Used for training and evaluation of the age classification model.

### FairFace
Used during exploratory data analysis (EDA) to analyze demographic balance and fairness considerations.


## Project Architecture

<img width="1408" height="768" alt="efficitentv2_s" src="https://github.com/user-attachments/assets/c5d475d8-f07f-45e2-b1aa-c2d2dc4b1591" />


## Model Architecture

The model uses transfer learning with lightweight CNN architectures suitable for edge deployment.

Models experimented:
- MobileNetV2 (baseline acc: 50%)
- MobileNetV3 (baseline acc: 55%)
- EfficientNetV2_S (baseline acc: 59%) (used is further process)
- EfficientNetV2_S (Fine tune acc: 72%)

Input: 224×224 RGB image  
Output: Age group classification

Age groups:

0–12
13–24
25–39
40–59
60+

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Macro F1 Score
- Confusion Matrix
- Fairness analysis across demographic groups

## Fairness Analysis

- Baseline Analysis
Fairness by race:
race_unified	count	accuracy	macro_f1	accuracy_pct	macro_f1_pct
0	Asian	510	0.641176	0.577362	64.12	57.74
1	Black	652	0.575153	0.429674	57.52	42.97
2	Indian	625	0.560000	0.487134	56.00	48.71
3	Other	247	0.570850	0.456209	57.09	45.62
4	White	1522	0.559790	0.542248	55.98	54.22


Fairness by gender:
gender_norm	count	accuracy	macro_f1	accuracy_pct	macro_f1_pct
0	Female	1688	0.589455	0.519924	58.95	51.99
1	Male	1868	0.562099	0.521521	56.21	52.15


Fairness gap summary:
metric	                value	            value_pct_points
0	race_accuracy_gap  	  0.081387	            8.14
1	race_macro_f1_gap	    0.147688	            14.77
2	gender_accuracy_gap	  0.027356	            2.74
3	gender_macro_f1_gap	  0.001598	            0.16


- ## Finetune Model Fairness



## Explainable AI

Grad-CAM was used to visualize which regions of the face influenced the model’s predictions. 
This helps ensure that the model focuses on relevant facial features instead of background artifacts.

## Privacy & Ethics

This project follows **privacy-by-design principles**.

Key safeguards:

- No facial recognition
- No identity tracking
- No image storage
- Edge processing only
- Aggregated outputs

The system aligns with:

- GDPR principles (Article 5,6,9,25)
- EU AI Act guidelines  (Limited Risk AI classification , Transparency , Human Oversight, Fairness and Bias Monitoring)
- Responsible AI development practices


## Limitations

- Age estimation from faces is inherently uncertain
- Adjacent age groups may be difficult to distinguish
- Dataset imbalance may affect performance

<img width="1470" height="956" alt="Screenshot 2026-03-08 at 13 10 57" src="https://github.com/user-attachments/assets/a04862b8-0026-46a0-9747-3c04ac7eb7cb" />


