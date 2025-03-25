### Instructions for Data Preparation

1. **Organize the Data**:
   - **MRI Image Readings**:  
     Place all follow-up **quantitative** and **semi-quantitative** MRI image readings in:  
     ```
     Datasets/MRI_ImageReadVariables/
     ```
   - **Radiograph Image Readings**:  
     Place all follow-up **quantitative**, **semi-quantitative**, and **angle** radiograph image readings in:  
     ```
    Datasets/Radiograph_ImageReadVariables/
     ```
   - **Clinical Variables**:  
     Place all follow-up clinical variables in:  
     ```
     Datasets/AllClinical/
     ```

2. **Combine Features**:  
   - Run the `GetVariables.py` script to generate a combined CSV file containing all features.  
   - The combined feature CSV file will include variables from MRI, radiograph images and their readings, as well as clinical data.
---

