# Explainable Predictive Process Monitoring: Evaluation Metrics and Guidelines for Process Outcome Prediction
Complementary code to reproduce the work of "Explainable Predictive Process Monitoring: Evaluation Metrics and Guidelines for Process Outcome Prediction"

An overview of the files:

### Preprocessing files 

The preprocessing and hyperoptimalisation are derivative work based on the code provided by https://github.com/irhete/predictive-monitoring-benchmark. 
We would like to thank the authors for the high quality code that allowed to fastly reproduce the provided work.
- dataset_confs
- DatasetManager
- EncoderFactory

### Hyperoptimalisation of parameters
- Hyperopt_LogitLeafModel
- Hyperopt_MachineLearningModels

### Training of the Machine Learning Models
*Logistic Regression (LR), Logit Leaf Model (LLM), Generalized Logistic Rule Regression (GLRM), Random Forest (RF) and XGBoost (XGB)*
- Experiment_ML.py

### Training of the Deep Learning Models (with Google Colab)
*Long short-term memory neural networks (LSTM) and Convolutional Neural Network( CNN)*
- experiment_DL(GC).ipynb

We acknowledgde the work provided by [Building accurate and interpretable models for predictive process analytics](https://github.com/renuka98/interpretable_predictive_processmodel) for their attention-based bidirectional LSTM architecture to create the long short-term neural networks with attention layers visualisations.

Finally, the folders contain additional figures and plots that have (not) been used in the paper.
