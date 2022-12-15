# Explainability in Process Outcome Prediction: Guidelines to Obtain Interpretable and Faithful Models
Complementary code to reproduce the work of "Explainable Predictive Process Monitoring: Evaluation Metrics and Guidelines for Process Outcome Prediction"

![Guidelines for accurate and eXplainable Models for process Outcome Prediction](https://user-images.githubusercontent.com/75080516/207905213-524fee98-da13-4bdc-b36e-34ccf3d40200.png)

_This figure contains the guideline to obtain eXplainable Models for Outcome Prediction (X-MOP)_
An overview of the files and folders:

##<sub>The different folders</sub>

### figures
The folder figures contain the high-resolution figures (PDF format) that have been used in the paper.

### labeled_logs_csv_processed

This folder contains cleaned and preprocessed event logs that are made available by this GitHub repository: [Benchmark for outcome-oriented predictive process monitoring](https://github.com/irhete/predictive-monitoring-benchmark). They provide 22 event logs, and we have selected 13 of them. The authors of this work an GitHub repository provide a [Google drive link](https://drive.google.com/open?id=154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR) to download these event logs.

### metrics
The metrics introduced in this paper are created in a seperate class (and .py) file. Hopefully, this will support reproducibility. The metrics are intended to work with all the benchmark models as mentioned in the paper. If there are any problems or questions, feel free to contact me (corresponding author and email related to the experiment of this study can be found on this [site](https://alexanderpaulstevens.github.io/).

### models
This folder contains only one file (LLM.py), where LLM model (originally written in R) is translated to Python. Note that there are some different design choices made compared to the original paper (link: [here](https://www.sciencedirect.com/science/article/abs/pii/S0377221718301243)).

### params_dir
The different hyperparameter files (in .pickle and .csv)

### results_dir
The different result files (in .csv) for the ML models (*results_dir_ML*) and the DL models (*results_dir_DL*).

### transformers
- contains the files to perform the aggregation sequence encoding (code stems from [Teinemaa et. al. (2019)](https://dl.acm.org/doi/abs/10.1145/3301300?casa_token=xiS8Iicds4sAAAAA:M-Wh_zwWGlsdj3QyD2GiK3uS66R484zPvbZJcsWke-UPkWMH3VYJKE0wx035cOlRn0-ux3J-hArmSCo)).

##<sub>The different .py files</sub>

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
