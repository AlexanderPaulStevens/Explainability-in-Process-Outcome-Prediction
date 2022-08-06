# Explainable Predictive Process Monitoring: Evaluation Metrics and Guidelines for Process Outcome Prediction
Complementary code to reproduce the work of "Explainable Predictive Process Monitoring: Evaluation Metrics and Guidelines for Process Outcome Prediction"

![Guidelines for XAI drawio-1](https://user-images.githubusercontent.com/75080516/183253235-2a255ae8-7be9-4552-a80b-cc655335979e.png)
_This figure contains the guideline to obtain eXplainable Models for Outcome Prediction (X-MOP)_
An overview of the files and folders:

### labeled_logs_csv_processed

This folder contains cleaned and preprocessed event logs that are made available by this GitHub repository: [Benchmark for outcome-oriented predictive process monitoring](https://github.com/irhete/predictive-monitoring-benchmark). They provide 22 event logs, and we have selected 13 of them. The authors of this work an GitHub repository provide a [Google drive link](https://drive.google.com/open?id=154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR) to download these event logs.

### Preprocessing files 

The preprocessing and hyperoptimalisation are derivative work based on the code provided by [Outcome-Oriented Predictive Process Monitoring: Review and Benchmark](https://github.com/irhete/predictive-monitoring-benchmark).
We would like to thank the authors for the high quality code that allowed to fastly reproduce the provided work.
- dataset_confs.py
- DatasetManager.py
- EncoderFactory.py

### Hyperoptimalisation of parameters
- Hyperopt_ML.py
- Hyperopt_DL (GC).ipynb

### Training of the Machine Learning Models
*Logistic Regression (LR), Logit Leaf Model (LLM), Generalized Logistic Rule Regression (GLRM), Random Forest (RF) and XGBoost (XGB)*
- Experiment_ML.py

### Training of the Deep Learning Models (with Google Colab)
*Long short-term memory neural networks (LSTM) and Convolutional Neural Network( CNN)*
- experiment_DL (GC).ipynb

We acknowledgde the work provided by [Building accurate and interpretable models for predictive process analytics](https://github.com/renuka98/interpretable_predictive_processmodel) for their attention-based bidirectional LSTM architecture to create the long short-term neural networks with attention layers visualisations.

Finally, the folder Figures contain the high-resolution figures (PDF format) that have been used in the paper.
