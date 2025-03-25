# Estimating Time-to-Total Knee Replacement on Radiographs and MRI: A Multi-Modal Approach Using Self-Supervised Deep Learning 
# Introduction
This repository contains the implementation of a deep learning-based model for predicting the expected duration of time until total knee replacement (time-to-TKR), developed as part of our research described in the paper: [Estimating Time-to-Total Knee Replacement on Radiographs and MRI: A Multi-Modal Approach Using Self-Supervised Deep Learning](https://academic.oup.com/radadv/advance-article/doi/10.1093/radadv/umae030/7901227?searchresult=1). This study combines features from MR images, clinical variables (e.g., age, BMI, and pain score) and image assessment measurements (e.g., Kellgren-Lawrence grade, joint space narrowing, bone marrow lesion size, and cartilage morphology) from multiple modalities to enhance precision in time-based predictions. With this implementation, you can train new models or utilize our pretrained models to predict time-to-TKR within a 9-year timeframe.

The pre-trained models used in our study are available at [https://doi.org/10.5281/zenodo.14947445](https://doi.org/10.5281/zenodo.14947445).

# Instructions
1. Extract radiograph image features using `models/ResNet18/` folder.
    To extract radiograph image features using a pretrained ResNet18 model, please run the feature extraction script:
    ```
    python GetFeaturesFromResNetXray.py  
    ```
    This script uses a pretrained ResNet18 model to extract features and saves the extracted radiograph image features to `models\ResNet18\Pretrained2DResnet18_XRay.csv`. The csv file looks like
    ```
    TrainValTest, File_Name, XRay0, XRay1, XRay2, ... , XRay510, XRay511
    test, 9002430_12m_RIGHT_KNEE.hdf5, 0.61005235, 0.32386965, 0.17083426, ... ,0.7932356, 0.7447767
    ```
    The `TrainValTest` column indicates the data split (train, validation, test). The `File_Name` column contains the radiograph name in the HDF5 file, where `9002430` represents the data ID, `12m` indicates the follow-up month, and`RIGHT_KNEE` specifies the knee side. The `XRay0-XRay511` columns represent the extracted 512 features from the input of the last fully connected layer of the ResNet18 model.
2.	Extract MR image features using `models/TWIST_pretraining/` folder.
    To extract MRI features with TWIST pre-trained model using PyTorch Lightning, please run
    ```
    python TWIST_lighting_inference.py  
    ``` 
    Please refer to `models/TWIST_pretraining/ReadMe.md` for details on the model training.
3.	Extract image readings and clinical variables using `Data_Preparation` folder.
    To combine all quantitative and semi-quantitative image readings from radiographs and MR images, please run
    ``` 
    python GetVariables.py
    ```
    This script generates a combined csv file, `Datasets/DESS_TSE_XRAY_Quantitative_Moaks_Radiographic_Clinical.csv`, including deep learning-extracted radiograph and MR image features, their readings, and clinical data. The csv file look like 
    ``` 
    TrainValTest, File_Name, timeVar, status, V00CBMFPD, V00WMTCTS, AGE, ..., XRay511
    test, 9002430_12m_RIGHT_KNEE.hdf5, 1355, 1, 8.946, 0.662, ..., 69, 0.57695794
    ```
    The `timeVar` column represents the number of days to TKR. A `status` of 1 indicates that the subject underwent TKR within 9 years (within the OAI study period), while a value of 0 indicates that the subject did not undergo TKR during this period. The columns `V00CBMFPD, V00WMTCTS, AGE, ..., XRay511` represent the combined features from the images, their readings, and clinical variables. 
4.	Please refer to `requirements.txt` to install all dependencies for random survival forests (RSF) model.
5.	Once the data is ready, please use the combined csv file and run     
    ```
    LassoCox_Feaure_Selection.R
    ```
    This script selects discriminative features using Lasso Cox feature selection, ranks them by importance, and saves the results as a csv file in `Data_Preparation/Datasets/Variable_Importance.csv`. Additionally, it saves the selected features as a csv file in `Data_Preparation/Datasets/SSL_Selected_TWIST_DESS_TSE_Xray_MOAKS_Quantitative_Radiographic_Imputed.csv`. 
9.	After selecting the features, in order to train RSF model, please run 
    ```
    main.R 
    ```
    This script saves the trained model and inference results within the folder named `models/ RSF_models/…'`
7.	To obtain a time-to-TKR prediction using our pretrained model, please provide the subject's information in a csv file `Data_Preparation/Datasets/SSL_TWIST_Selected_DESS_TSE_Xray_MOAKS_Quantitative_Radiographic_Imputed.csv` 
    The CSV file should be formatted as follows:
    ``` 
    TrainValTest, File_Name, timeVar, status, V00CBMFPD, V00WMTCTS, AGE, ..., XRay511
    test, 9002430_12m_RIGHT_KNEE.hdf5, 1355, 1, 8.946, 0.662, ..., 69, 0.57695794
    ```
    For the full list of 95 required features and their order in the csv file, please refer to **Table 5** in the **[Supplementary data](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/radadv/1/4/10.1093_radadv_umae030/3/umae030_supplementary_data.pdf?Expires=1743617500&Signature=frXPsi8~QAWCiA~GqYQuDCdJsiMWjC8jrpbSu5LreMRmvJQtKOwcbJjAP5UG08G8LcvOrkg~agAx~dh5Zonzs6NRQnTblBMz8etYNhctQp7Rt4MGH9x45CY7s7MgtzJYTURmo-Umjw4cVzIbNDtbOsCf4z7gLOPcniMO9PZiVmZ-GktvD5ZUiw1fECJRff2OxmA9jbSqShZciM0FxDtkDXqGzIb-dE6VNuR-GkLIh819HcZ~5ewHORMhoQ9BNvIGeQVAEFK2wJ2nFtIBFLhcqy70ku3uvnxI5YCaukUlUey3lnd9Ni1GjEu0RxsMK5wWf06FnwWUS406aXU0tHGyFQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)**. 
    Once the csv file is prepared, please run:
    ```
    inference.R
    ```
    This script will provide accuracy, C-Index, macro-AUC with 95% CI, and the integrated Brier Score for time-to-TKR prediction.

# Repository Structure
### Top-Level Scripts
•	`autoSurv_TKR.R`: Trains and evaluates the RSF model, enabling survival analysis and model optimization.
•	`predict_autoSurv_TKR.R`: Performs inference, generating predicted survival probabilities and assessing predictive performance on test datasets.
•	`LassoCox_Feature_Selection.R`: Implements the Lasso Cox feature selection approach to identify the most discriminative features.
•	`main.R`: The main RSF model script for training, validating, and testing.
•	`Inference.R`: Script for inference using the trained RSF model.
### Directories
#### `Data_Preparation/`
Contains instructions and codes for preparing data for model training and evaluation.
##### `Models/`
Directory organized into three subfolders for different purposes:
##### o	 `ResNet18/`
* Includes the feature extraction code for radiograph images.
* When the code is executed, the extracted features are saved as csv files in this folder.
##### o	`TWIST_pretraining/`
* Includes pretrained model weights and checkpoints for initializing models in transfer learning and fine-tuning tasks.
* `./saved_models/`: Contains the pretrained model weights and checkpoints.
##### o	`RSF_models/`
* Stores the saved RSF models used in this study for various input datasets.
* These models are optimized for survival analysis tasks.

<!-- 
# Extracting Radiograph Image Features
To extract radiograph image features using a pretrained ResNet18 model, please use `models/ResNet18/` folder for details.
# Extracting MR Image Features 
To extract MR image features using the TWIST model, please use `models/TWIST_pretraining/` folder for details. -->
# Random Survival Forests Model
This repository leverages `autoSurv`, a survival prediction framework inspired by the work of Suresh [1], to train and evaluate the random survival forests model for time-to-TKR prediction.
Key Features of `autoSurv`:
•	Implements the continuous-time RSF model within a discrete-time framework.
•	Offers automated hyperparameter tuning using Bayesian optimization to enhance predictive performance.
•	Facilitates model evaluation with time-dependent metrics such as AUC and Brier score for handling right-censored data.
•	Enables predictions for new datasets, allowing the assessment of predictive performance on independent test sets.
For a comprehensive description of `autoSurv` and its full capabilities, refer to the original implementation by Suresh [1].
Functions
This repository utilizes two main functions based on `autoSurv`:
•	`autoSurv_TKR()`: Trains and evaluates the RSF model, enabling survival analysis and model optimization.
•	`predict_autoSurv_TKR()`: Focuses on inference, generating predicted survival probabilities and assessing predictive performance on test datasets.
These functions provide a workflow for training, inference, and evaluating survival prediction models for time-to-TKR analysis.

# License
This repository is licensed under the terms of the GNU AGPLv3 license.
# References
[1] Suresh K, Severn C, Ghosh D. Survival prediction models: an introduction to discrete-time modeling. BMC Med Res Methodol. 2022;22(1):207. doi:10.1186/s12874-022-01679-6.

# Citation
If you found this code useful, please cite our paper:

*Estimating Time-to-Total Knee Replacement on Radiographs and MRI: A Multi-Modal Approach Using Self-Supervised Deep Learning*. Ozkan Cigdem, Shengjia Chen, Chaojie Zhang, Kyunghyun Cho, Richard Kijowski, and Cem M. Deniz. Radiology Advances 2024.
```
@article{cigdem2024ra,
    author = {Cigdem, Ozkan and Chen, Shengjia and Zhang, Chaojie and Cho, Kyunghyun and Kijowski, Richard and Deniz, Cem M},
    title = {Estimating Time-to-Total Knee Replacement on Radiographs and MRI: A Multi-Modal Approach Using Self-Supervised Deep Learning},
    journal = {Radiology Advances},
    pages = {umae030},
    year = {2024},
    month = {11},
    issn = {2976-9337},
    doi = {10.1093/radadv/umae030},
    url = {https://doi.org/10.1093/radadv/umae030},
    eprint = {https://academic.oup.com/radadv/advance-article-pdf/doi/10.1093/radadv/umae030/60686238/umae030.pdf},
}

```