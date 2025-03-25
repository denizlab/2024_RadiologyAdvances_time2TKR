# ==============================================================================
# Copyright (C) 2024 Ozkan Cigdem, Shengjia Chen, Chaojie Zhang, 
# Kyunghyun Cho, Richard Kijowski, Cem M Deniz

# This file is part of 2024_RadiologyAdvances_time2TKR
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
#####################################################################################################
#####################################################################################################
# 1. Remove unnecessary columns:ID, readprf, Side, ...
# 2. Get Clinical Variables (be careful with comnbining columns, check their names and data from different years 12m, 24m, 48m)
# 3. Remove the variables with 0 flag in Variable.csv and apply threshold of 10%.
# 4. Combine clinical with image readings 
# 5. Define Quantitative and Categorical variables. 
# 6. Floor the categorical variables before imputation
# 7. Impute categorical variables with Mode and quantitative variables with single layer MLP
# 8. Save both before imputation and after imputation csv files. 
# 9. Check for the order of the variables: 00m, 12m, 24, 36m, and 48m.
#####################################################################################################
#####################################################################################################
#######################################################################################################################
## Here we combine Quantitative Eckstein MRI Readings for different months in a csv file for different months
#######################################################################################################################
import os
import pandas as pd
import numpy as np
import csv 
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.impute import SimpleImputer
#######################################################################################################################
directory_path = 'Datasets/MRI_ImageReadVariables/Quant_SAS/'
txt_number = ["00", "01", "03", "05", "06"]
months = ['00m', '12m', '24m', '36m', '48m']
# Initialize an empty DataFrame to store combined data
all_dataframes = pd.DataFrame()

# Iterate through each month
for i in range(len(months)):
    month = months[i]
    txt = txt_number[i]
    file_path = os.path.join(directory_path, f'kmri_qcart_eckstein{txt}.csv')  # Construct the full path with the specific month
    each_dataframe = pd.read_csv(file_path)  # Skip the first row (header)
    each_dataframe['SIDE'] = ["RIGHT" if pd.notna(side) and int(side) == 1 else "LEFT" for side in each_dataframe["SIDE"]]
    each_dataframe['File_Name'] = ['_'.join([str(row['ID']), month, row['SIDE'], "KNEE.hdf5"]) for index, row in each_dataframe.iterrows()]
    each_dataframe.insert(0, 'File_Name', each_dataframe.pop('File_Name'))
    renamed_columns = [col[3:] if col.startswith(f'V{txt}') else col for col in each_dataframe.columns[1:]]  # Skip the first column ("File_Name")
    each_dataframe.columns = ['File_Name'] + renamed_columns  # Prepend "File_Name" to renamed columns
    reordered_cols = ['File_Name'] + sorted(col for col in renamed_columns if col != 'File_Name')
    each_dataframe = each_dataframe[reordered_cols] 

    # file_path_new = os.path.join(directory_path, f'reordered_kmri_qcart_eckstein{txt}.csv') 
    # each_dataframe.to_csv(file_path_new, index=False)

    all_dataframes = pd.concat([all_dataframes, each_dataframe], ignore_index=True)

# Specify the output file path
output_file = 'Datasets/MRI_ImageReadVariables/Quant_SAS/Combined_kmri_qcart_eckstein.csv'

# Save the combined DataFrame to a new CSV file with header
all_dataframes.to_csv(output_file, index=False)

print(f"Combined data saved to {output_file}")
##########################################################################################################################
##########################################################################################################################
## Here we combine Semi-quantitatie MOAKS MRI Readings for different months in a csv file for different months

directory_path = 'Datasets/MRI_ImageReadVariables/Semi-Quant Scoring_SAS/'
txt_number = ["00", "01", "03", "05", "06"]
months = ['00m', '12m', '24m', '36m', '48m']

# Initialize an empty DataFrame to store combined data
all_dataframes = pd.DataFrame()

# Iterate through each month
for i in range(len(months)):
    month = months[i]
    txt = txt_number[i]
    file_path = os.path.join(directory_path, f'kmri_sq_moaks_bicl{txt}.csv')  # Construct the full path with the specific month
    each_dataframe = pd.read_csv(file_path)  # Skip the first row (header)
    each_dataframe['SIDE'] = ["RIGHT" if pd.notna(side) and int(side) == 1 else "LEFT" for side in each_dataframe["SIDE"]]
    each_dataframe['File_Name'] = ['_'.join([str(row['ID']), month, row['SIDE'], "KNEE.hdf5"]) for index, row in each_dataframe.iterrows()]
    each_dataframe.insert(0, 'File_Name', each_dataframe.pop('File_Name'))
    renamed_columns = [col[3:] if col.startswith(f'V{txt}') else col for col in each_dataframe.columns[1:]]  # Skip the first column ("File_Name")
    each_dataframe.columns = ['File_Name'] + renamed_columns  # Prepend "File_Name" to renamed columns
    reordered_cols = ['File_Name'] + sorted(col for col in renamed_columns if col != 'File_Name')
    each_dataframe = each_dataframe[reordered_cols] 

    # file_path_new = os.path.join(directory_path, f'reordered_kmri_sq_moaks_bicl{txt}.csv') 
    # each_dataframe.to_csv(file_path_new, index=False)

    all_dataframes = pd.concat([all_dataframes, each_dataframe], ignore_index=True)

# Specify the output file path
output_file = 'Datasets/MRI_ImageReadVariables/Semi-Quant Scoring_SAS/Combined_kmri_sq_moaks_bicl.csv'

# Save the combined DataFrame to a new CSV file with header
all_dataframes.to_csv(output_file, index=False)

print(f"Combined data saved to {output_file}")

######################################################

##########################################################################################################################
## Here we combine Xray Readings for different months in a csv file for different months
directory_path = 'Datasets/Radiograph_ImageReadVariables/'
months = ['00m', '12m', '24m', '36m', '48m', '72m', '96m']
txt_number = ['00','01','03','05','06','08','10']
all_xray_list = ["kxr_sq_bu", "kxr_fta_duryea", "kxr_qjsw_duryea"]
for reading_type in all_xray_list:
    all_dataframes = pd.DataFrame()    # Initialize an empty DataFrame to store combined data
    for i in range(len(months)):
        month = months[i]
        txt = txt_number[i]
        file_path = os.path.join(directory_path, f'{reading_type}{txt}.csv')  # Construct the full path with the specific month
        each_dataframe = pd.read_csv(file_path)  # Skip the first row (header)
        each_dataframe = each_dataframe.rename(columns={'side': 'SIDE', 'readprj':'READPRJ'} if 'side' or 'readprj' in each_dataframe.columns else {})
        each_dataframe['SIDE'] = ["RIGHT" if pd.notna(side) and int(side) == 1 else "LEFT" for side in each_dataframe["SIDE"]]
        each_dataframe['File_Name'] = ['_'.join([str(row['ID']), month, row['SIDE'], "KNEE.hdf5"]) for index, row in each_dataframe.iterrows()]
        each_dataframe.insert(0, 'File_Name', each_dataframe.pop('File_Name'))
        renamed_columns = [col[3:] if col.startswith(f'V{txt}') else col for col in each_dataframe.columns[1:]]  # Skip the first column ("File_Name")
        each_dataframe.columns = ['File_Name'] + renamed_columns  # Prepend "File_Name" to renamed columns
        reordered_cols = ['File_Name'] + sorted(col for col in renamed_columns if col != 'File_Name')
        each_dataframe = each_dataframe[reordered_cols] 
        # file_path_new = os.path.join(directory_path, f'reordered_kmri_sq_moaks_bicl{txt}.csv') 
        # each_dataframe.to_csv(file_path_new, index=False)
        all_dataframes = pd.concat([all_dataframes, each_dataframe], ignore_index=True)
    output_file = f'Datasets/Radiograph_ImageReadVariables/Combined_{reading_type}.csv'
    all_dataframes.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
###########################################################################################################
###########################################################################################################
    
###########################################################################################################
###########################################################################################################
## Get the intersection of MRI and XRAY readings
###########################################################################################################
###########################################################################################################
# Get the reference data list from image data (intersection of all available TSE, DESS, Xray images)
dataDESS = pd.read_csv('/models/TWIST_pretraining/extracted_features/SAG_3D_DESS_Features.csv').iloc[:,:4]
dataTSE= pd.read_csv("/models/TWIST_pretraining/extracted_features/SAG_IW_TSE_OAI_Features.csv").iloc[:,1:2]
dataXray = pd.read_csv('/models/ResNet18/Pretrained2DResnet18_XRay.csv').iloc[:,1:2]
dataXray["File_Name"] = dataXray["File_Name"].apply(lambda x: '_'.join([*x.split('_')[:3], 'KNEE.hdf5']))
dataTSE["File_Name"] = dataTSE["File_Name"].apply(lambda x: '_'.join([*x.split('_')[:3], 'KNEE.hdf5']))
dataDESS["File_Name"] = dataDESS["File_Name"].apply(lambda x: '_'.join([*x.split('_')[:3], 'KNEE.hdf5']))
dataMRI = pd.merge(dataTSE, dataDESS, on ='File_Name', how='inner')
Ref_File_all = pd.merge(dataMRI, dataXray, on ='File_Name', how = 'inner')
column_order_ref = ['TrainValTest', 'File_Name', 'status','timeVar'] 
Ref_File = Ref_File_all[column_order_ref]
###########################################################################################################
dir_XrayQuant = 'Datasets/Radiograph_ImageReadVariables/Combined_kxr_qjsw_duryea.csv'        
dir_XrayAngle = 'Datasets/Radiograph_ImageReadVariables/Combined_kxr_fta_duryea.csv'  
dir_kxr_bu =    'Datasets/Radiograph_ImageReadVariables/Combined_kxr_sq_bu.csv'  

XrayQuant = pd.read_csv(dir_XrayQuant)
XrayAngle = pd.read_csv(dir_XrayAngle)
XrayCateg = pd.read_csv(dir_kxr_bu)
###########################################################################################################
Xray_Quant_Angle = pd.merge(XrayAngle, XrayQuant, on='File_Name', how='inner')  # They have the same sizes (all quantitatives)
Xray_Quant_Angle_2 = pd.merge(Ref_File, Xray_Quant_Angle, on='File_Name', how='left')
###########################################################################################################
XrayCateg_2 = pd.merge(Ref_File, XrayCateg, on='File_Name', how='left')
###########################################################################################################
# Group by ID, SIDE, and readprj, find the readprj values with the most occurrences. Get either 15 or 37 Project No.
most_frequent_readprjXrayCateg_2 = XrayCateg_2.groupby(['ID', 'SIDE', 'READPRJ']).size().groupby(['ID', 'SIDE']).idxmax()
new_long_dataXrayCateg = XrayCateg_2[XrayCateg_2[['ID', 'SIDE', 'READPRJ']].apply(tuple, axis=1).isin(most_frequent_readprjXrayCateg_2.values)]
new_long_dataXrayCateg2 = new_long_dataXrayCateg.drop_duplicates(subset=['File_Name'], keep='first')
###########################################################################################################
columns_drop_Xrays = ["SIDE", "SIDE_x", "SIDE_y", "READPRJ", "READPRJ_x","READPRJ_y", "readprj","readprj_x","readprj_y", 
                            "ID","ID_x","ID_y", "VERSION","VERSION_x","VERSION_y",
                            "BRCDJD","BARCDJD","V01BARCDBU", "BARCDBU", "BARCDFE","SION",
                            "BRCDJD","BARCDJD", "TKRpositives","FTAFLAG", "INCPLL", "INCPLM",
                            "INCSTPS", "LTPMEBE", "MJSWBB", "NOLJSWX", "NOLMIN", "NOMJSWX", "NOMMJSW"]
new_long_dataXrayCateg2_clean = new_long_dataXrayCateg2.drop(columns=[col for col in new_long_dataXrayCateg2.columns if col in columns_drop_Xrays])
Xray_Quant_Angle_clean =  Xray_Quant_Angle_2.drop(columns=[col for col in Xray_Quant_Angle_2.columns if col in columns_drop_Xrays])
Xray_Quant_Angle_clean = Xray_Quant_Angle_clean.drop(columns=['TrainValTest', 'status', 'timeVar'])
###########################################################################################################
## Merge the Quantitative+Angle with semi-quantitative Xray readings
merged_Xray2 = pd.merge(Xray_Quant_Angle_clean, new_long_dataXrayCateg2_clean, on='File_Name', how='left')
all_dataframes_Xray = merged_Xray2.drop_duplicates(subset=['File_Name'])  # These are the XRAY readings for all TKR- and TKR+
###########################################################################################################
# There are 547 TKR+ subjects ( and right knees of the same subject are assumed as seperate subjects). 
# In this 547 subjects, 9(right) and 12(left) subjects have self-reported TKR operation time. We excluded 
# these subjects' images in our Time2TKR_Xray_Pos_All.csv file. Also 9466244_12m_LEFT_KNEE.hdf knee 
# had TKR operation, so we excluded it as well. In total, 547-22=525 subjects have knee images.
# XrayCateg has 534 TKR+ subjects, 2502 longitidunal knees with different project number, and 2378 knees with same vendor.
# all_dataframes_Xray (quant, categ, angle) has 525 TKR+ subjects and 2323 knees with same vendor.
###########################################################################################################
###########################################################################################################
## Get the MRI readings for all TKR positive knees
###########################################################################################################
MRI_readings_MOAKS = pd.read_csv("Datasets/MRI_ImageReadVariables/Semi-Quant Scoring_SAS/Combined_kmri_sq_moaks_bicl.csv") 
MRI_readings_QUAN = pd.read_csv("Datasets/MRI_ImageReadVariables/Quant_SAS/Combined_kmri_qcart_eckstein.csv") 
###########################################################################################################

if 'readprj' in MRI_readings_QUAN.columns:
    MRI_readings_QUAN.rename(columns={'readprj': 'READPRJ'}, inplace=True)
readprj_mapping_QUAN = {'09A': '09', '09B': '09', '22b': '22'}
MRI_readings_QUAN["READPRJ"].replace(readprj_mapping_QUAN, inplace=True)
MRI_readings_QUAN_1 = pd.merge(Ref_File, MRI_readings_QUAN, on='File_Name', how='left') 
most_frequent_readprj_MRI_QUAN = MRI_readings_QUAN_1.groupby(['ID', 'SIDE', 'READPRJ']).size().groupby(['ID', 'SIDE']).idxmax() # Group by ID, SIDE, and readprj, find the readprj values with the most occurrences
MRI_readings_QUAN_2 = MRI_readings_QUAN_1[MRI_readings_QUAN_1[['ID', 'SIDE', 'READPRJ']].apply(tuple, axis=1).isin(most_frequent_readprj_MRI_QUAN.values)]
TKR_MRI_QUANT = MRI_readings_QUAN_2.drop_duplicates(subset=['File_Name'], keep='first') # There are subjects with 9A and 9B (above we replaced them to be 9). We get the first one
TKR_MRI_QUANT.dropna(subset=['File_Name'], inplace=True)
###########################################################################################################

if 'readprj' in MRI_readings_MOAKS.columns:
    MRI_readings_MOAKS.rename(columns={'readprj': 'READPRJ'}, inplace=True)
readprj_mapping_MOAKS = {'63A': '63', '63B': '63', '63C': '63', '63D': '63', '63E': '63', '63F': '63'}
MRI_readings_MOAKS["READPRJ"].replace(readprj_mapping_MOAKS, inplace=True)

new_long_data2_MOAKS = pd.merge(Ref_File, MRI_readings_MOAKS, on='File_Name', how='left')  # 1054 TKR+ knees have MRI readings
most_frequent_readprj_MOAKS = new_long_data2_MOAKS.groupby(['ID', 'SIDE', 'READPRJ']).size().groupby(['ID', 'SIDE']).idxmax() # Group by ID, SIDE, and READPRJ, find the READPRJ values with the most occurrences
new_long_data_MOAKS = new_long_data2_MOAKS[new_long_data2_MOAKS[['ID', 'SIDE', 'READPRJ']].apply(tuple, axis=1).isin(most_frequent_readprj_MOAKS.values)]
TKR_MRI_MOAKS = new_long_data_MOAKS.drop_duplicates(subset=['File_Name'], keep='first')
TKR_MRI_MOAKS.dropna(subset=['File_Name'], inplace=True)

###########################################################################################################
columns_drop_MRI = ["SIDE", "SIDE_x", "SIDE_y", "READPRJ", "READPRJ_x","READPRJ_y", "readprj","readprj_x","readprj_y", 
                            "ID","ID_x","ID_y", "VERSION","VERSION_x","VERSION_y",
                            "BRCDJD","BARCDJD","V01BARCDBU", "BARCDBU", "BARCDFE","SION",
                            "BRCDJD","BARCDJD", "TKRpositives"]
TKR_MRI_QUANT_clean = TKR_MRI_QUANT.drop(columns=[col for col in TKR_MRI_QUANT.columns if col in columns_drop_MRI])
TKR_MRI_MOAKS_clean = TKR_MRI_MOAKS.drop(columns=[col for col in TKR_MRI_MOAKS.columns if col in columns_drop_MRI])
TKR_MRI_MOAKS_clean = TKR_MRI_MOAKS_clean.drop(columns=['TrainValTest', 'status', 'timeVar'])
ref_MOAKS_list = pd.read_csv("Y:\cigdeo01\TKR_days/runModel\Datasets\InputCsv_ModelResultsCsv\Final_Datasets_Journal\OldFiles_Ignore/AllMRIreadingMOAKSImputed.csv")
TKR_MRI_MOAKS_clean = TKR_MRI_MOAKS_clean[[col[3:] if col.startswith('V00') else col for col in ref_MOAKS_list.columns[1:] if (col[3:] if col.startswith('V00') else col) not in ['status', 'timeVar']]]
###########################################################################################################
TKR_MRI_All = pd.merge(TKR_MRI_QUANT_clean, TKR_MRI_MOAKS_clean, on='File_Name', how='inner') 
# most_frequent_readprj_TKR_MRI_All = TKR_MRI_All.groupby(['ID_x', 'SIDE_x', 'READPRJ_x']).size().groupby(['ID_x', 'SIDE_x']).idxmax() 
TKR_MRI_All.drop_duplicates(subset=['File_Name'], inplace=True) 
###########################################################################################################
TKR_MRI_All.drop(columns=['TrainValTest', 'status', 'timeVar'], inplace= True)
all_dataframes_Xray.drop(columns=['TrainValTest', 'status', 'timeVar'], inplace= True)

All_Image_Readings = pd.merge(TKR_MRI_All, all_dataframes_Xray, on ='File_Name', how = 'inner')
###########################################################################################################
# There are 547 TKR+ subjects (left and right knees of the same subject are assumed as seperate subjects). 
# 394 subjects have Quantitative MRI readings and 358 subjects have MOAKS readings.  323 subjects have both 
# Quantitative and MOAKS MRI readings. These subjects include self-reported TKR operations times. We will 
# exclude them when we merge MRI and Xray readings below. After merging them, 311 subjects have both MRI and
# Xray image readings (including all quantitative and categorical readings). 
###########################################################################################################
# Below, we get the DESS, TSE, XRAY images and merge them to have knees with available DESS, TSE, XRAY images

dataDESS = pd.read_csv('/models/TWIST_pretraining/extracted_features/SAG_3D_DESS_Features.csv')
dataTSE= pd.read_csv("/models/TWIST_pretraining/extracted_features/SAG_IW_TSE_OAI_Features.csv")
dataXray = pd.read_csv('/models/ResNet18/Pretrained2DResnet18_XRay.csv')

dataXray["File_Name"] = dataXray["File_Name"].apply(lambda x: '_'.join([*x.split('_')[:3], 'KNEE.hdf5']))
dataTSE["File_Name"] = dataTSE["File_Name"].apply(lambda x: '_'.join([*x.split('_')[:3], 'KNEE.hdf5']))
dataDESS["File_Name"] = dataDESS["File_Name"].apply(lambda x: '_'.join([*x.split('_')[:3], 'KNEE.hdf5']))
dataDESS = dataDESS.drop(columns=['TrainValTest', 'status', 'timeVar'])
dataXray = dataXray.drop(columns=['TrainValTest'])

dataMRI = pd.merge(dataTSE, dataDESS, on ='File_Name', how='inner')
All_Images = pd.merge(dataMRI, dataXray, on ='File_Name', how = 'inner')

data_All_Images_Readings = pd.merge(All_Images, All_Image_Readings, on ='File_Name', how = 'inner')
data_All_Images_Readings_last = data_All_Images_Readings.drop_duplicates(subset=['File_Name'])
column_order_last = ['TrainValTest', 'File_Name', 'status','timeVar'] + [col for col in data_All_Images_Readings_last.columns if col not in ['TrainValTest', 'File_Name', 'status','timeVar']]
data_All_Images_Readings_last_last = data_All_Images_Readings_last[column_order_last]
###########################################################################################################
## Drop the variables available less than 10% of all subjects and merge them with imaging data. Count missing values in each column (in clinical data)
missing_counts_readings = (data_All_Images_Readings_last_last.eq("") | data_All_Images_Readings_last_last.isnull()).sum(axis=0)
# print(missing_counts_clinical)
columns_to_drop_readings = missing_counts_readings[missing_counts_readings > 0.2 * len(data_All_Images_Readings_last_last)].index.tolist()
print(f'{len(columns_to_drop_readings)} features are removed')
Alldata_readings =  data_All_Images_Readings_last_last.drop(columns=columns_to_drop_readings) # Remove the columns which have more than 90% of missing values
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
# Below, we get the clinical variables for knees obtained above. These knees have different months 
# (00m,12m,24m,36m,48m). For each month, we get the clinical variables for these knees. Eg: Clinical: 00m --> AllClinical00.txt
# 00m, 298. It means in all knees, there are 298 knees at month 00m. 

selected_clincial_features = []
############################################################################################################################################################
txt_files_path = "Datasets/AllClinical/" 
months = ['00m', '12m', '24m', '36m', '48m']
txt_number = ['00', '01', '03', '05', '06']
############################################################################################################################################################
# Extract the patient IDs from the file names (eg extract 12345 from 12345_00m_LEFT_SAG.hdf5)
patient_ids = [file_name.split('_')[0] for file_name in Alldata_readings['File_Name']]
patient_file_name = [file_name for file_name in Alldata_readings['File_Name']]
patient_ids_by_month = {} # Create a dictionary to store the patient IDs for each month
############################################################################################################################################################
for month_patient in months:
    # Filter the patient IDs and file names for the current month
    filtered_data = [(patient_id, file_name) for patient_id, file_name in zip(patient_ids, patient_file_name) if file_name.split('_')[1].startswith(month_patient)]
    filtered_ids, filtered_file_names = zip(*filtered_data)    # Separate the patient IDs and file names into separate lists
    patient_ids_by_month[month_patient] = {'patient_ids': filtered_ids,'file_names': filtered_file_names}    # Assign the patient IDs and file names to the current month
############################################################################################################################################################
# # Iterate over each element in months and retrieve the matching element from txt_number
for i in range(len(months)):  
    month = months[i]    # Get each month
    txt = txt_number[i]  # Get the index of each txt file 
    txt_file_path = txt_files_path + f"AllClinical{txt}.txt"   # Get the related txt file for each month
    print(f'Clinical: {month} --> {f"AllClinical{txt}.txt"}')
    patient_id_all =  patient_ids_by_month[month]['patient_ids']
    file_names_all = patient_ids_by_month[month]['file_names']
    print(f'{month}, {len(patient_id_all)}')
    save_path = txt_files_path + f"Datasets/AllClinical/{month}_AllClinical.csv"  # Replace with the desired save path
    file_exists = os.path.isfile(save_path)    # Check if the output file already exists
    a = '1'
    with open(save_path, "a", newline='') as output_file:
        csv_writer = csv.writer(output_file)

        for patient_ID, file_name_patient in zip(patient_id_all, file_names_all): 
            with open(txt_file_path, "r") as txt_file:
                txt_reader = csv.reader(txt_file, delimiter="|") 
                header = next(txt_reader)  # Read the header  
                if a =='1' : # Write the header only if the file doesn't exist
                    csv_writer.writerow(['File_Name'] + header)  # Write the header
                    a = '2'
                patient_id_index = header.index("ID")  # Find the index of the column with patient ID in the text file
                for txt_row in txt_reader:
                    txt_patient_id = txt_row[patient_id_index]   
                    if txt_patient_id == patient_ID:                     # Check if the patient ID is in the list of patient IDs for the current month
                        txt_columns = txt_row                            # Get all columns from the text file
                        csv_data = [file_name_patient] + txt_columns     # Combine the file_name_patient and txt_columns into a single list (eg: train, File_Name, Clinical Variables) data_group:train/val/test
                        csv_writer.writerow(csv_data)
            txt_file.close() 
###########################################################################################################
# Below  we found all clinical variables in all months and append all features (union of all months) in reference_features_stripped.
directory_path = 'Datasets/AllClinical/'
reference_features_stripped = []
for i in range(len(months)):  
    file_path = directory_path + f"{months[i]}_AllClinical.csv"  
    df_forall = pd.read_csv(file_path)  
    df_forall.columns = df_forall.columns.str.replace('ID', 'XXXID').str.replace('File_Name', 'XXXFile_Name')
    df_forall.columns = [column[3:] for column in df_forall.columns]
    if not reference_features_stripped:
        reference_features_stripped = list(df_forall.columns)         # For the first DataFrame, set reference_features_stripped
    for feature in df_forall.columns:     # Check and add missing columns
        if feature not in reference_features_stripped:
            reference_features_stripped.append(feature)
    print(len(reference_features_stripped))
reference_features_stripped = list(set(reference_features_stripped)) #remove duplicates from reference_features_stripped
###########################################################################################################
# Below, I concatenated all (union) clinical variables available in all months for all data
dataframes = []
for i in range(len(months)):  
    file_path = directory_path + f"{months[i]}_AllClinical.csv"  
    df_forall = pd.read_csv(file_path)  
    df_forall.columns = df_forall.columns.str.replace('File_Name', 'XXXFile_Name')
    df_forall.columns = [column[3:] for column in df_forall.columns]
    for feature in reference_features_stripped:    # Check and add missing columns
        if feature not in df_forall.columns:
            df_forall[feature] = ''  # Add an empty column
    dataset = df_forall[reference_features_stripped]
    dataset = dataset.loc[:, ~dataset.columns.duplicated(keep='first')]
    print(dataset.shape)
    dataframes.append(dataset)
output_file = f'{directory_path}AllClinical.csv'
selected_clincial_features = pd.concat(dataframes, ignore_index=True)
selected_clincial_features.to_csv(output_file, index=False)
# selected_clincial_features = pd.read_csv(output_file)
###########################################################################################################
# Below, we first remove the variables which were set as 0 in 
# /gpfs/data/denizlab/Datasets/denizlab/OAI/Clinical/AllClinical/VariableInformation.xls. 
# Then we use 10% of threshold. We only get the variables(features) which are available for more than 90% of
# all knees. 
columns_drop_unnecessary_clinical = ["SIDE", "SIDE_x", "SIDE_y", "READPRJ", "READPRJ_x","READPRJ_y", "readprj","readprj_x","readprj_y", 
                            "ID","ID_x","ID_y", "VERSION","VERSION_x","VERSION_y",
                            "BRCDJD","BARCDJD","V01BARCDBU", "BARCDBU", "BARCDFE","SION",
                            "BRCDJD","BARCDJD","MCMNTS", "MTCMNTS","W2STFID", "W4STFID","K1STFID","P01SVXRRID", 
                            "PDATE1","PDATE2", "UCDATE1", "MRSEQNR", "MRSEQNL","P01STFID2", 
                            "STFID1", "HESTFID", "BPSTFID", "RPSTFID","ACSTFID", "SCSTFID", 
                            "RCSTFID", "isstfid", "isexmdt",  
                            "VERSION", "SITE","MJSWBB", "KIDTRAN", "PTH", "CHELCV", "FOLKCV", 
                            "RXFLUOR", "RXMSM", "RXTPRTD", "RXSAME", "BPTERM", 
                            "UCDATE1","UCDATE2","DATE",'SVDATE', "EVDATE", "PSDATE", "SSDATE", "FVDATE","VOXXXID1","VOXXXID2",
                            "URINHR1", "URINHR2", "BLDHRS1", "BLDHRS2", "SEAQHR1", "SEAQHR2",
                            "PLAQHR1", "PLAQHR2", "URINOB1", "URINOB2", "VOID1", "VOID2",
                            "HOURSP1", "HOURSP2", "QOVP1", "QOVP2", "VCOLL1", "VCOLL2",
                            "HEMAT1", "HEMAT2", "VEIN1", "VEIN2", "MULTST1", "MULTST2",
                            "EXCESS1", "EXCESS2", "LEAKAG1", "LEAKAG2", "OTHVP1", "OTHVP2",
                            "BLDRAW1", "BLDRAW2", "HRSUC1", "HRSUC2", "P01SVXRRID",
                            "P01SVXRELK", "P01XRKOA", "BLDCOLL", "UCDATE2", "URNCOLL",
                            "PDATE2", "PDATE1", "UCDATE1", "BLSURD2", "URSURD2", "BLSURD1",
                            "URSURD1", "MRSEQNR", "MRSEQNL", "SERUM", "EDTA", "CITRATE",
                            "P01STFID2", "P01STFID1", "P01HESTFID", "BPSTFID", "BPCFSZ",
                            "BPARM", "RPSTFID", "ACSTFID", "SCSTFID", "RCSTFID", "W2STFID",
                            "W4STFID", "K1STFID", "isstfid", "isexmdt", "P02DATE", "P02HR1",
                            "P02HR2", "P02HR3", "P02HR4", "P02HR5", "P02HR6", "P02HR7",
                            "P02HR8", "P02HR9", "P02HR10", "P02HR11", "P01SVDATE", "P01MRIB4",
                            "P01MRCMP", "P01CLAU", "EVDATE", "P02STMEDCV", "P01MRPRBCV", "P01SVXRRKR", "P01SVXRLKR", 
                            "P01SVRKMI", "P01SVLKMI", "XRBCODE", "XROSFLR", "XRSCFLR",
                            "XRCYFLR", "XRJSLR","XRCHLR","XROSTLR","XRSCTLR","XRCYTLR",
                            "XRATTLR","XROSFMR", "XRSCFMR", "XRCYFMR", "XRJSMR", "XRCHMR",
                            "XROSTMR", "XRSCTMR", "XRCYTMR", "XRATTMR", "XRKLR", "XROSFML",
                            "XRSCFML", "XRCYFML", "XRJSML", "XRCHML", "XROSTML", "XRSCTML",
                            "XRCYTML", "XRATTML", "XRKLL", "XROSFLL", "XRSCFLL", "XRCYFLL",
                            "XRJSLL", "XRCHLL", "XROSTLL", "XRSCTLL", "XRCYTLL", "XRATTLL"]
alldata_clinical_dropped= selected_clincial_features.drop(columns=[col for col in selected_clincial_features.columns if col in columns_drop_unnecessary_clinical])
###########################################################################################################
## Drop the variables available less than 10% of all subjects and merge them with imaging data. Count missing values in each column (in clinical data)
missing_counts_clinical = (alldata_clinical_dropped.eq("") | alldata_clinical_dropped.isnull()).sum(axis=0)
columns_to_drop_clinical = missing_counts_clinical[missing_counts_clinical > 0.1 * len(alldata_clinical_dropped)].index.tolist()
# columns_to_drop_clinical = missing_counts_clinical[missing_counts_clinical > 0.204 * len(alldata_clinical_dropped)].index.tolist()
print(f'{len(columns_to_drop_clinical)} features are removed')
alldata_clinical_dropped =  alldata_clinical_dropped.drop(columns=columns_to_drop_clinical) # Remove the columns which have more than 90% of missing values
###########################################################################################################
# Below, we merged the MRI and XRAY image readings with clinical variables
data_All_Images_Readings_Clinical = pd.merge(Alldata_readings, alldata_clinical_dropped, on ='File_Name', how = 'inner')
count_status_TKR_p = len(data_All_Images_Readings_Clinical[data_All_Images_Readings_Clinical['status'] == 1])
count_status_TKR_n = len(data_All_Images_Readings_Clinical[data_All_Images_Readings_Clinical['status'] == 0])
print(f'Number of TKR+:{count_status_TKR_p} \nNumber of TKR-:{count_status_TKR_n}')
###########################################################################################################
# Below, we get the Quantitative and Categorical variables by using threhold of 20.
unique_counts = data_All_Images_Readings_Clinical.nunique()  # Calculate the number of unique values for each variable
categorical_ones = ["BPDAYCV", "KPACDCV", "CESD", "BPTOT"]   # Define variables with more than 20 unique values but still categorical
unique_counts.drop(['TrainValTest', 'File_Name', 'status', 'timeVar'], inplace=True)
categorical_vars = []
quantitative_vars = []
threshold = 20                                                # Threshold for considering variables as categorical
for col, unique_count in unique_counts.items():               # Determine categorical and quantitative variables
    if col in categorical_ones or unique_count < threshold:
        categorical_vars.append(col)
    else:
        quantitative_vars.append(col)
print("Categorical Variables:", len(categorical_vars))
print("Quantitative Variables:", len(quantitative_vars))
selected_columns = ['TrainValTest', 'File_Name', 'status', 'timeVar'] + quantitative_vars + categorical_vars # Define the order of columns. Add quantitative and categorical variables
data_All_Images_Readings_Clinical_reordered = data_All_Images_Readings_Clinical[selected_columns]   # Reorder the dataframe
All_Quantitative = data_All_Images_Readings_Clinical[quantitative_vars]
All_Categorical = data_All_Images_Readings_Clinical[categorical_vars]
###########################################################################################################
# Some categorical variables are needed to be "floor". We do masking here
mask = All_Categorical.notnull() & (All_Categorical != '')    # Mask for non-empty cells
All_Categorical[mask] = np.floor(All_Categorical[mask])       # Apply floor function to non-empty cells
####################################################################
# Below, we do imputation for categorical (using mode) and numeric(quantitative) variables (using MLP)
####################################################################

# Define a custom dataset class
class CSVDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
        self.imputer = SimpleImputer(strategy='mean')

        # Preprocess data to convert non-numeric columns
        self.data = self.data.select_dtypes(include=[np.number])
        self.imputer.fit(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data.iloc[idx].values
        return torch.tensor(sample, dtype=torch.float)
dataset = CSVDataset(All_Quantitative)
# Create a data loader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_size = len(dataset[0]) # Define the model
output_size = input_size
print(output_size)
model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, output_size)
)
criterion = nn.MSELoss()                                    # Define the loss function and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 50
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
imputed_data = dataset.imputer.transform(dataset.data)  # Get the imputed values
print(dataset.data.isnull().sum())
print(imputed_data.shape)
# Convert imputed data to DataFrame
All_Quantitative_imputed = pd.DataFrame(imputed_data, columns=dataset.data.columns)
#####################################################################################################
## Impute categorical variables with its mode
modes = All_Categorical.mode().iloc[0]                        # Calculate the mode of each column
All_Categorical_filled = All_Categorical.fillna(modes)        # Fill missing values with mode for each column
# Concatenate the two DataFrames column-wise
All_reordered_imputed = pd.concat([data_All_Images_Readings_Clinical[['TrainValTest', 'File_Name', 'status', 'timeVar']], All_Quantitative_imputed, All_Categorical_filled], axis=1)
# All_reordered_imputed.to_csv('Datasets/Quantitative_Moaks_Radiographic_Clinical_Imputed.csv', index=False)
#####################################################################################################
#####################################################################################################
#####################################################################################################
# Merge all clinical and image readings with image data:DESS, TSE, XRAY.
img_DESS = pd.read_csv('/models/TWIST_pretraining/extracted_features/SAG_3D_DESS_Features.csv')
img_TSE= pd.read_csv("/models/TWIST_pretraining/extracted_features/SAG_IW_TSE_OAI_Features.csv")
img_XRAY = pd.read_csv('/models/ResNet18/Pretrained2DResnet18_XRay.csv')


img_DESS['File_Name'] = img_DESS['File_Name'].apply(lambda x: '_'.join([*x.split('_')[:3], 'KNEE.hdf5']))
img_TSE['File_Name'] = img_TSE['File_Name'].apply(lambda x: '_'.join([*x.split('_')[:3], 'KNEE.hdf5']))
img_XRAY['File_Name'] = img_XRAY['File_Name'].apply(lambda x: '_'.join([*x.split('_')[:3], 'KNEE.hdf5']))
img_DESS.drop(columns=['TrainValTest'], inplace=True)
img_TSE.drop(columns=['TrainValTest'], inplace=True)
img_XRAY.drop(columns=['TrainValTest'], inplace=True)

XRAY_All = pd.merge(All_reordered_imputed, img_XRAY, on ='File_Name', how = 'inner')
TSE_ALL = pd.merge(XRAY_All, img_TSE, on ='File_Name', how = 'inner') 
DESS_ALL= pd.merge(TSE_ALL, img_DESS, on ='File_Name', how = 'inner')
DESS_ALL_last = DESS_ALL.drop_duplicates(subset=['File_Name'])
DESS_ALL_last.to_csv('Datasets/DESS_TSE_XRAY_Quantitative_Moaks_Radiographic_Clinical.csv', index=False)

num_unique_columns = len(DESS_ALL_last.columns.drop_duplicates())
print(len(DESS_ALL_last.columns), num_unique_columns)
#####################################################################################################
