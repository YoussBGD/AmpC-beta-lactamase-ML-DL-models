Machine Learning Models Execution Guide

This guide provides detailed instructions for running Feedforward Neural Network (FFNN), Random Forest (RF), and Support Vector Machine (SVM) models included in this repository.These models are designed to predict non-covalent inhibitors of the AmpC beta-lactamase enzyme.



Repository Structure ------------------------------------------------------------------------------------------------
FFNN_model/: Contains the FFNN model "FFNN_model.hdf5", test set "test_set.csv", execution script "test_FFNN_model.ipynb" and necessary files for runing the model (mean.pkl, std.pkl).

RF_model/: Contains the RF model, test set, and execution script (randomforest-best_model.rds, test_set.csv, Test_model.R).
SVM_model/: Contains the SVM model, test set, and execution script (svm_best_model.rds, Test_model.R, test_set.csv).
-----------------------------------------------------------------------------------------------------------------

Requirements ----------------------------------------------------------------------------------------------------

Python 3.x for FFNN
R version 3.4.4 or higher for RF and SVM
----------------------------------------------------------------------------------------------------------------



Dependency Installation-----------------------------------------------------------------------------------------

FFNN (Python)
Install the necessary dependencies:
    pip install pandas numpy tensorflow scikit-learn

RF and SVM (R)
The required packages:
    install.packages(c("caret", "randomForest", "e1071", "pROC", "ROCR"))
----------------------------------------------------------------------------------------------------------------


Codes execution ------------------------------------------------------------------------------------------------

FFNN
Navigate to the FFNN_model/ directory and execute the FFNN model python script.

    python FFNN_model/test_FFNN_model.py

RF
Open an R console or RStudio, navigate to the RF_model/ directory, and execute the provided R script Test_RF_model.R.

SVM
In an R console or RStudio, navigate to the SVM_model/ directory and execute the provided R script Test_SVM_model.R.
----------------------------------------------------------------------------------------------------------------

Outputs --------------------------------------------------------------------------------------------------------
Each execution script generates a CSV file as output, which contains the computed performance metrics: sensitivity, specificity, balanced accuracy, MCC (Matthews Correlation Coefficient), and AUC (Area Under the Curve).
----------------------------------------------------------------------------------------------------------------
Support
For any questions or issues, feel free to open an issue in the repository or contact bagdad.youcef.ybg@gmail.com.
