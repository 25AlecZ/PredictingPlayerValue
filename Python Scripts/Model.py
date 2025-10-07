#install packages
!pip install --upgrade pip
!pip install pandas --upgrade pip
!pip install xgboost --upgrade pip
!pip install sklearn --upgrade pip
!pip install shap
!pip install graphviz

from google.colab import drive
from google.colab import files
drive.mount('/content/drive')

#import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from xgboost import plot_tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import gc
import numpy as np

#read df for each year model
InjuryModelOneYeardf = pd.read_csv('/content/drive/MyDrive/PlayerValue/InjuryYearOneSeparateInjuries.csv')
InjuryModelOneYeardf.drop('Unnamed: 0', axis=1, inplace=True)
InjuryModelOneYeardf.drop(columns = ['injuryType'], inplace = True)
x1year = InjuryModelOneYeardf.dropna(subset=['duration_truth'])
y1year = x1year['duration_truth'].copy()

InjuryModelTwoYeardf = pd.read_csv('/content/drive/MyDrive/PlayerValue/InjuryYearTwoSeparateInjuries.csv')
InjuryModelTwoYeardf.drop('Unnamed: 0', axis=1, inplace=True)
InjuryModelTwoYeardf.drop(columns = ['injuryType'], inplace = True)
x2year = InjuryModelTwoYeardf.dropna(subset=['duration_truth'])
y2year = x2year['duration_truth'].copy()

InjuryModelThreeYeardf = pd.read_csv('/content/drive/MyDrive/PlayerValue/InjuryYearThreeSeparateInjuries.csv')
InjuryModelThreeYeardf.drop('Unnamed: 0', axis=1, inplace=True)
InjuryModelThreeYeardf.drop(columns = ['injuryType'], inplace = True)
x3year = InjuryModelThreeYeardf.dropna(subset=['duration_truth'])
y3year = x3year['duration_truth'].copy()

InjuryModelFourYeardf = pd.read_csv('/content/drive/MyDrive/PlayerValue/InjuryYearFourSeparateInjuries.csv')
InjuryModelFourYeardf.drop('Unnamed: 0', axis=1, inplace=True)
InjuryModelFourYeardf.drop(columns = ['injuryType'], inplace = True)
x4year = InjuryModelFourYeardf.dropna(subset=['duration_truth'])
y4year = x4year['duration_truth'].copy()

InjuryModelFiveYeardf = pd.read_csv('/content/drive/MyDrive/PlayerValue/InjuryYearFiveSeparateInjuries.csv')
InjuryModelFiveYeardf.drop('Unnamed: 0', axis=1, inplace=True)
InjuryModelFiveYeardf.drop(columns = ['injuryType'], inplace = True)
x5year = InjuryModelFiveYeardf.dropna(subset=['duration_truth'])
y5year = x5year['duration_truth'].copy()

# @title
playerdf2018 = pd.read_csv('/content/drive/MyDrive/PlayerValue/playerdf2018.csv')
playerdf2018.drop('Unnamed: 0', axis=1, inplace=True)

playerdf2019 = pd.read_csv('/content/drive/MyDrive/PlayerValue/playerdf2019.csv')
playerdf2019.drop('Unnamed: 0', axis=1, inplace=True)

playerdf2020 = pd.read_csv('/content/drive/MyDrive/PlayerValue/playerdf2020.csv')
playerdf2020.drop('Unnamed: 0', axis=1, inplace=True)

playerdf2021 = pd.read_csv('/content/drive/MyDrive/PlayerValue/playerdf2021.csv')
playerdf2021.drop('Unnamed: 0', axis=1, inplace=True)

playerdf2022 = pd.read_csv('/content/drive/MyDrive/PlayerValue/playerdf2022.csv')
playerdf2022.drop('Unnamed: 0', axis=1, inplace=True)

playerdf2023 = pd.read_csv('/content/drive/MyDrive/PlayerValue/playerdf2023.csv')
playerdf2023.drop('Unnamed: 0', axis=1, inplace=True)

InjuryYearDurationUpdatedYearsdf = pd.read_csv('/content/drive/MyDrive/PlayerValue/InjuryYearDurationUpdatedYearsdf.csv')
InjuryYearDurationUpdatedYearsdf.drop('Unnamed: 0', axis=1, inplace=True)

# @title
playerFrebdf = pd.concat([playerdf2018,playerdf2019, playerdf2020, playerdf2021, playerdf2022, playerdf2023], ignore_index = True)

# @title
#concat all season dataframes from freb datasets
playerFrebdf['Age'] = pd.to_numeric(playerFrebdf['Age'])
playerFrebdf.rename(columns = {'player_url':'Player_url'}, inplace = True)
playerFrebdf = playerFrebdf.merge(InjuryYearDurationUpdatedYearsdf, on = 'Player_url', how = 'left')
playerFrebdf.count()
#no blank spaces in column names
#playerFrebdf.columns = playerFrebdf.columns.str.replace(' ','_')
#playerFrebdf['player_height_mtrs'] = pd.to_numeric(playerFrebdf['player_height_mtrs'])
#playerFrebdf.to_csv('/content/drive/MyDrive/PlayerValue/Frebdf.csv')

# @title
#Injury with regular column names
Injury2018df = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2018df.csv')
Injury2018df.drop('Unnamed: 0', axis=1, inplace=True)

Injury2019df = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2019df.csv')
Injury2019df.drop('Unnamed: 0', axis=1, inplace=True)

Injury2020df = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2020df.csv')
Injury2020df.drop('Unnamed: 0', axis=1, inplace=True)

Injury2021df = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2021df.csv')
Injury2021df.drop('Unnamed: 0', axis=1, inplace=True)

Injury2022df = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2022df.csv')
Injury2022df.drop('Unnamed: 0', axis=1, inplace=True)

Injury2023df = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2023df.csv')
Injury2023df.drop('Unnamed: 0', axis=1, inplace=True)

#Injury with columns names including year in duration and type of injury
Injury2018YearDurationdf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2018YearDurationdf.csv')
Injury2018YearDurationdf.drop('Unnamed: 0', axis=1, inplace=True)

Injury2019YearDurationdf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2019YearDurationdf.csv')
Injury2019YearDurationdf.drop('Unnamed: 0', axis=1, inplace=True)

Injury2020YearDurationdf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2020YearDurationdf.csv')
Injury2020YearDurationdf.drop('Unnamed: 0', axis=1, inplace=True)

Injury2021YearDurationdf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2021YearDurationdf.csv')
Injury2021YearDurationdf.drop('Unnamed: 0', axis=1, inplace=True)

Injury2022YearDurationdf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2022YearDurationdf.csv')
Injury2022YearDurationdf.drop('Unnamed: 0', axis=1, inplace=True)

Injury2023YearDurationdf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Injury2023YearDurationdf.csv')
Injury2023YearDurationdf.drop('Unnamed: 0', axis=1, inplace=True)

#Rename player_url column
Injury2018df.rename(columns = {'Player_url':'player_url'}, inplace = True)
Injury2019df.rename(columns = {'Player_url':'player_url'}, inplace = True)
Injury2020df.rename(columns = {'Player_url':'player_url'}, inplace = True)
Injury2021df.rename(columns = {'Player_url':'player_url'}, inplace = True)
Injury2022df.rename(columns = {'Player_url':'player_url'}, inplace = True)
Injury2023df.rename(columns = {'Player_url':'player_url'}, inplace = True)

#Injury Model
#Concatenate all data into overall injury df, then for loop search for player name, check player year duration and type duration, then replace NaN values with 0
InjuryYearDurationdf = pd.concat([Injury2018YearDurationdf, Injury2019YearDurationdf], ignore_index = True)
InjuryYearDurationdf = pd.concat([InjuryYearDurationdf, Injury2020YearDurationdf], ignore_index = True)
InjuryYearDurationdf = pd.concat([InjuryYearDurationdf, Injury2021YearDurationdf], ignore_index = True)
InjuryYearDurationdf = pd.concat([InjuryYearDurationdf, Injury2022YearDurationdf], ignore_index = True)
InjuryYearDurationdf = pd.concat([InjuryYearDurationdf, Injury2023YearDurationdf], ignore_index = True)

#Correcting columns for different years model by setting duration and type for each player
for i in range(0, len(InjuryYearDurationdf)):
  playerUrl = InjuryYearDurationdf.loc[i, "Player_url"]
  for j in range(i, len(InjuryYearDurationdf)):
    if (InjuryYearDurationdf.loc[j, "Player_url"] == playerUrl):
        if (not InjuryYearDurationdf.isnull().loc[j, "Duration_2019"]):
          InjuryYearDurationdf.loc[i, "Duration_2019"] = InjuryYearDurationdf.loc[j, "Duration_2019"]
          InjuryYearDurationdf.loc[i, "Type_2019"] = InjuryYearDurationdf.loc[j, "Type_2019"]
        elif (not InjuryYearDurationdf.isnull().loc[j, "Duration_2020"]):
          InjuryYearDurationdf.loc[i, "Duration_2020"] = InjuryYearDurationdf.loc[j, "Duration_2020"]
          InjuryYearDurationdf.loc[i, "Type_2020"] = InjuryYearDurationdf.loc[j, "Type_2020"]
        elif (not InjuryYearDurationdf.isnull().loc[j, "Duration_2021"]):
          InjuryYearDurationdf.loc[i, "Duration_2021"] = InjuryYearDurationdf.loc[j, "Duration_2021"]
          InjuryYearDurationdf.loc[i, "Type_2021"] = InjuryYearDurationdf.loc[j, "Type_2021"]
        elif (not InjuryYearDurationdf.isnull().loc[j, "Duration_2022"]):
          InjuryYearDurationdf.loc[i, "Duration_2022"] = InjuryYearDurationdf.loc[j, "Duration_2022"]
          InjuryYearDurationdf.loc[i, "Type_2022"] = InjuryYearDurationdf.loc[j, "Type_2022"]
        elif (not InjuryYearDurationdf.isnull().loc[j, "Duration_2023"]):
          InjuryYearDurationdf.loc[i, "Duration_2023"] = InjuryYearDurationdf.loc[j, "Duration_2023"]
          InjuryYearDurationdf.loc[i, "Type_2023"] = InjuryYearDurationdf.loc[j, "Type_2023"]
  print(i)

# Concat on player_url
playerdf2018 = playerdf2018.merge(Injury2018df, on = 'player_url', how = 'left')
playerdf2019 = playerdf2019.merge(Injury2019df, on = 'player_url', how = 'left')
playerdf2020 = playerdf2020.merge(Injury2020df, on = 'player_url', how = 'left')
playerdf2021 = playerdf2021.merge(Injury2021df, on = 'player_url', how = 'left')
playerdf2022 = playerdf2022.merge(Injury2022df, on = 'player_url', how = 'left')
playerdf2023 = playerdf2023.merge(Injury2023df, on = 'player_url', how = 'left')

#concat all season dataframes from freb datasets
playerFrebdf = pd.concat([playerdf2018,playerdf2019, playerdf2020, playerdf2021, playerdf2022, playerdf2023], ignore_index = True)
playerFrebdf['Age'] = pd.to_numeric(playerFrebdf['Age'])
playerFrebdf['player_height_mtrs'] = pd.to_numeric(playerFrebdf['player_height_mtrs'])
playerFrebdf.sample(2)

# injurytypes new list
injurytypes = []

#Creates list with all injuries
for i in range (0, len(x1year)):
  found = False
  for j in range (0, len(injurytypes)):
    if (injurytypes[j] == x1year.iloc[i, 222]):
      found = True
      break
  if (found == False):
    injurytypes.append(x1year.iloc[i, 222])

x1year['injurytype_nan'] = 0 #nan

x1year['injurytype_abdomen'] = 0 #Abdominal Strain

x1year['injurytype_abductor'] = 0 #Tear in the abductor muscle,

x1year['injurytype_achilles'] = 0 #Achilles tendon rupture, Achilles tendon problems,

x1year['injurytype_adductor'] = 0 #Adductor problems,

x1year['injurytype_ankle'] = 0 #Ankle Injury, Sprained ankle, Ankle problems, Torn ankle ligament, Ankle Surgery

x1year['injurytype_arm'] = 0 #Arm Injury, Wirst Injury, Shoulder Injury

x1year['injurytype_back'] = 0 #Back Injury, Back trouble, Back bruise

x1year['injurytype_bruise'] = 0 #Bruise

x1year['injurytype_calf'] = 0 #Calf Problems, Calf Injury, Calf Strain

x1year['injurytype_chest'] = 0 #Angina,

x1year['injurytype_collarbone'] = 0 #Collarbone fracture

x1year['injurytype_concussion'] = 0 #Concussion

x1year['injurytype_cruciateligament'] = 0 #Cruciate Ligament Rupture, Cruciate Ligament Surgery, Cruciate Ligament Strain

x1year['injurytype_face'] = 0 #Facial Injury

x1year['injurytype_finger'] = 0 #Finger Injury

x1year['injurytype_fitness'] = 0 #Fitness

x1year['injurytype_foot'] = 0 #Foot Injury, Bruised Foot, Fractured Foot,

x1year['injurytype_groin'] = 0 #groin strain, Groin Strain (w/Capital), Groin Injury,

x1year['injurytype_hamstring'] = 0 #Hamstring Injury

x1year['injurytype_hand'] = 0 #Hand Injury, Hand Problems,

x1year['injurytype_head'] = 0 #Head Injury

x1year['injurytype_heart'] = 0 # Heart Condition

x1year['injurytype_hip'] = 0 #Hip problems, Hip Injury,

x1year['injurytype_ill'] = 0 #Cold, Influenza, Flu, Abdominal Influenza, Ill, Corona virus, Infection, Virus Infection, Quaratine, Testicular Cancer
#include abdominal influenza, cold, ill, covid, influenza, fever, quarantine, infection

x1year['injurytype_inflammation'] = 0 #Inflammation

x1year['injurytype_knee'] = 0 #Knee Injury, Knee Problems, Knee inflammation, Bruised Knee, Ruptured intraarticular ligament initiation in knee,

x1year['injurytype_knock'] = 0 #Knock, Minor Knock,

x1year['injurytype_leg'] = 0 #Leg Injury

x1year['injurytype_muscular'] = 0 #Muscular problems, Muscle Injury, Biceps femoris muscle injury, Torn Muscle Fibre, Muscle Fatigue

x1year['injurytype_pelvis'] = 0 #Pubitis, bruised pelvis, Pelvis Injury, Pubis bone irritation

x1year['injurytype_rest'] = 0 #Rest

x1year['injurytype_rib'] = 0 #Fractured Rib

x1year['injurytype_spine'] = 0 #Lumbar Vertebra Fracture, Lumbar vertebrae problems, Blockade in the spinal,

x1year['injurytype_thigh'] = 0 #Thigh Muscle Strain, Thigh Problems,

x1year['injurytype_toe'] = 0 #Toe Injury

x1year['injurytype_unknown'] = 0 #Unknown Injury,

x1year['injurytype_other'] = 0 #Surgery, Contracture, Sprain, Laceration

x2year['injurytype_nan'] = 0 #nan

x2year['injurytype_abdomen'] = 0 #Abdominal Strain

x2year['injurytype_abductor'] = 0 #Tear in the abductor muscle,

x2year['injurytype_achilles'] = 0 #Achilles tendon rupture, Achilles tendon problems,

x2year['injurytype_adductor'] = 0 #Adductor problems,

x2year['injurytype_ankle'] = 0 #Ankle Injury, Sprained ankle, Ankle problems, Torn ankle ligament, Ankle Surgery

x2year['injurytype_arm'] = 0 #Arm Injury, Wirst Injury, Shoulder Injury

x2year['injurytype_back'] = 0 #Back Injury, Back trouble, Back bruise

x2year['injurytype_bruise'] = 0 #Bruise

x2year['injurytype_calf'] = 0 #Calf Problems, Calf Injury, Calf Strain

x2year['injurytype_chest'] = 0 #Angina,

x2year['injurytype_collarbone'] = 0 #Collarbone fracture

x2year['injurytype_concussion'] = 0 #Concussion

x2year['injurytype_cruciateligament'] = 0 #Cruciate Ligament Rupture, Cruciate Ligament Surgery, Cruciate Ligament Strain

x2year['injurytype_face'] = 0 #Facial Injury

x2year['injurytype_finger'] = 0 #Finger Injury

x2year['injurytype_fitness'] = 0 #Fitness

x2year['injurytype_foot'] = 0 #Foot Injury, Bruised Foot, Fractured Foot,

x2year['injurytype_groin'] = 0 #groin strain, Groin Strain (w/Capital), Groin Injury,

x2year['injurytype_hamstring'] = 0 #Hamstring Injury

x2year['injurytype_hand'] = 0 #Hand Injury, Hand Problems,

x2year['injurytype_head'] = 0 #Head Injury

x2year['injurytype_heart'] = 0 # Heart Condition

x2year['injurytype_hip'] = 0 #Hip problems, Hip Injury,

x2year['injurytype_ill'] = 0 #Cold, Influenza, Flu, Abdominal Influenza, Ill, Corona virus, Infection, Virus Infection, Quaratine, Testicular Cancer
#include abdominal influenza, cold, ill, covid, influenza, fever, quarantine, infection

x2year['injurytype_inflammation'] = 0 #Inflammation

x2year['injurytype_knee'] = 0 #Knee Injury, Knee Problems, Knee inflammation, Bruised Knee, Ruptured intraarticular ligament initiation in knee,

x2year['injurytype_knock'] = 0 #Knock, Minor Knock,

x2year['injurytype_leg'] = 0 #Leg Injury

x2year['injurytype_muscular'] = 0 #Muscular problems, Muscle Injury, Biceps femoris muscle injury, Torn Muscle Fibre, Muscle Fatigue

x2year['injurytype_pelvis'] = 0 #Pubitis, bruised pelvis, Pelvis Injury, Pubis bone irritation

x2year['injurytype_rest'] = 0 #Rest

x2year['injurytype_rib'] = 0 #Fractured Rib

x2year['injurytype_spine'] = 0 #Lumbar Vertebra Fracture, Lumbar vertebrae problems, Blockade in the spinal, Cervical spine injury

x2year['injurytype_thigh'] = 0 #Thigh Muscle Strain, Thigh Problems,

x2year['injurytype_toe'] = 0 #Toe Injury

x2year['injurytype_unknown'] = 0 #Unknown Injury,

x2year['injurytype_other'] = 0 #Surgery, Contracture, Sprain, Laceration

#nan, lower body, upper body, ill, head, other
x3year['injurytype_nan'] = 0 #nan

x3year['injurytype_abdomen'] = 0 #Abdominal Strain

x3year['injurytype_abductor'] = 0 #Tear in the abductor muscle,

x3year['injurytype_achilles'] = 0 #Achilles tendon rupture, Achilles tendon problems,

x3year['injurytype_adductor'] = 0 #Adductor problems,

x3year['injurytype_ankle'] = 0 #Ankle Injury, Sprained ankle, Ankle problems, Torn ankle ligament, Ankle Surgery

x3year['injurytype_arm'] = 0 #Arm Injury, Wirst Injury, Shoulder Injury

x3year['injurytype_back'] = 0 #Back Injury, Back trouble, Back bruise

x3year['injurytype_bruise'] = 0 #Bruise

x3year['injurytype_calf'] = 0 #Calf Problems, Calf Injury, Calf Strain

x3year['injurytype_chest'] = 0 #Angina,

x3year['injurytype_collarbone'] = 0 #Collarbone fracture

x3year['injurytype_concussion'] = 0 #Concussion

x3year['injurytype_cruciateligament'] = 0 #Cruciate Ligament Rupture, Cruciate Ligament Surgery, Cruciate Ligament Strain

x3year['injurytype_face'] = 0 #Facial Injury

x3year['injurytype_finger'] = 0 #Finger Injury

x3year['injurytype_fitness'] = 0 #Fitness

x3year['injurytype_foot'] = 0 #Foot Injury, Bruised Foot, Fractured Foot,

x3year['injurytype_groin'] = 0 #groin strain, Groin Strain (w/Capital), Groin Injury,

x3year['injurytype_hamstring'] = 0 #Hamstring Injury

x3year['injurytype_hand'] = 0 #Hand Injury, Hand Problems,

x3year['injurytype_head'] = 0 #Head Injury

x3year['injurytype_heart'] = 0 # Heart Condition

x3year['injurytype_hip'] = 0 #Hip problems, Hip Injury,

x3year['injurytype_ill'] = 0 #Cold, Influenza, Flu, Abdominal Influenza, Ill, Corona virus, Infection, Virus Infection, Quaratine, Testicular Cancer
#include abdominal influenza, cold, ill, covid, influenza, fever, quarantine, infection

x3year['injurytype_inflammation'] = 0 #Inflammation

x3year['injurytype_knee'] = 0 #Knee Injury, Knee Problems, Knee inflammation, Bruised Knee, Ruptured intraarticular ligament initiation in knee,

x3year['injurytype_knock'] = 0 #Knock, Minor Knock,

x3year['injurytype_leg'] = 0 #Leg Injury

x3year['injurytype_muscular'] = 0 #Muscular problems, Muscle Injury, Biceps femoris muscle injury, Torn Muscle Fibre, Muscle Fatigue

x3year['injurytype_pelvis'] = 0 #Pubitis, bruised pelvis, Pelvis Injury, Pubis bone irritation

x3year['injurytype_rest'] = 0 #Rest

x3year['injurytype_rib'] = 0 #Fractured Rib

x3year['injurytype_spine'] = 0 #Lumbar Vertebra Fracture, Lumbar vertebrae problems, Blockade in the spinal, Cervical spine injury

x3year['injurytype_thigh'] = 0 #Thigh Muscle Strain, Thigh Problems,

x3year['injurytype_toe'] = 0 #Toe Injury

x3year['injurytype_unknown'] = 0 #Unknown Injury,

x3year['injurytype_other'] = 0 #Surgery, Contracture, Sprain, Laceration

#nan, lower body, upper body, ill, head, other
x4year['injurytype_nan'] = 0 #nan

x4year['injurytype_abdomen'] = 0 #Abdominal Strain

x4year['injurytype_abductor'] = 0 #Tear in the abductor muscle,

x4year['injurytype_achilles'] = 0 #Achilles tendon rupture, Achilles tendon problems,

x4year['injurytype_adductor'] = 0 #Adductor problems,

x4year['injurytype_ankle'] = 0 #Ankle Injury, Sprained ankle, Ankle problems, Torn ankle ligament, Ankle Surgery

x4year['injurytype_arm'] = 0 #Arm Injury, Wirst Injury, Shoulder Injury

x4year['injurytype_back'] = 0 #Back Injury, Back trouble, Back bruise

x4year['injurytype_bruise'] = 0 #Bruise

x4year['injurytype_calf'] = 0 #Calf Problems, Calf Injury, Calf Strain

x4year['injurytype_chest'] = 0 #Angina,

x4year['injurytype_collarbone'] = 0 #Collarbone fracture

x4year['injurytype_concussion'] = 0 #Concussion

x4year['injurytype_cruciateligament'] = 0 #Cruciate Ligament Rupture, Cruciate Ligament Surgery, Cruciate Ligament Strain

x4year['injurytype_face'] = 0 #Facial Injury

x4year['injurytype_finger'] = 0 #Finger Injury

x4year['injurytype_fitness'] = 0 #Fitness

x4year['injurytype_foot'] = 0 #Foot Injury, Bruised Foot, Fractured Foot,

x4year['injurytype_groin'] = 0 #groin strain, Groin Strain (w/Capital), Groin Injury,

x4year['injurytype_hamstring'] = 0 #Hamstring Injury

x4year['injurytype_hand'] = 0 #Hand Injury, Hand Problems,

x4year['injurytype_head'] = 0 #Head Injury

x4year['injurytype_heart'] = 0 # Heart Condition

x4year['injurytype_hip'] = 0 #Hip problems, Hip Injury,

x4year['injurytype_ill'] = 0 #Cold, Influenza, Flu, Abdominal Influenza, Ill, Corona virus, Infection, Virus Infection, Quaratine, Testicular Cancer
#include abdominal influenza, cold, ill, covid, influenza, fever, quarantine, infection

x4year['injurytype_inflammation'] = 0 #Inflammation

x4year['injurytype_knee'] = 0 #Knee Injury, Knee Problems, Knee inflammation, Bruised Knee, Ruptured intraarticular ligament initiation in knee,

x4year['injurytype_knock'] = 0 #Knock, Minor Knock,

x4year['injurytype_leg'] = 0 #Leg Injury

x4year['injurytype_muscular'] = 0 #Muscular problems, Muscle Injury, Biceps femoris muscle injury, Torn Muscle Fibre, Muscle Fatigue

x4year['injurytype_pelvis'] = 0 #Pubitis, bruised pelvis, Pelvis Injury, Pubis bone irritation

x4year['injurytype_rest'] = 0 #Rest

x4year['injurytype_rib'] = 0 #Fractured Rib

x4year['injurytype_spine'] = 0 #Lumbar Vertebra Fracture, Lumbar vertebrae problems, Blockade in the spinal, Cervical spine injury

x4year['injurytype_thigh'] = 0 #Thigh Muscle Strain, Thigh Problems,

x4year['injurytype_toe'] = 0 #Toe Injury

x4year['injurytype_unknown'] = 0 #Unknown Injury,

x4year['injurytype_other'] = 0 #Surgery, Contracture, Sprain, Laceration

#nan, lower body, upper body, ill, head, other
x5year['injurytype_nan'] = 0 #nan

x5year['injurytype_abdomen'] = 0 #Abdominal Strain

x5year['injurytype_abductor'] = 0 #Tear in the abductor muscle,

x5year['injurytype_achilles'] = 0 #Achilles tendon rupture, Achilles tendon problems,

x5year['injurytype_adductor'] = 0 #Adductor problems,

x5year['injurytype_ankle'] = 0 #Ankle Injury, Sprained ankle, Ankle problems, Torn ankle ligament, Ankle Surgery

x5year['injurytype_arm'] = 0 #Arm Injury, Wirst Injury, Shoulder Injury

x5year['injurytype_back'] = 0 #Back Injury, Back trouble, Back bruise

x5year['injurytype_bruise'] = 0 #Bruise

x5year['injurytype_calf'] = 0 #Calf Problems, Calf Injury, Calf Strain

x5year['injurytype_chest'] = 0 #Angina,

x5year['injurytype_collarbone'] = 0 #Collarbone fracture

x5year['injurytype_concussion'] = 0 #Concussion

x5year['injurytype_cruciateligament'] = 0 #Cruciate Ligament Rupture, Cruciate Ligament Surgery, Cruciate Ligament Strain

x5year['injurytype_face'] = 0 #Facial Injury

x5year['injurytype_finger'] = 0 #Finger Injury

x5year['injurytype_fitness'] = 0 #Fitness

x5year['injurytype_foot'] = 0 #Foot Injury, Bruised Foot, Fractured Foot,

x5year['injurytype_groin'] = 0 #groin strain, Groin Strain (w/Capital), Groin Injury,

x5year['injurytype_hamstring'] = 0 #Hamstring Injury

x5year['injurytype_hand'] = 0 #Hand Injury, Hand Problems,

x5year['injurytype_head'] = 0 #Head Injury

x5year['injurytype_heart'] = 0 # Heart Condition

x5year['injurytype_hip'] = 0 #Hip problems, Hip Injury,

x5year['injurytype_ill'] = 0 #Cold, Influenza, Flu, Abdominal Influenza, Ill, Corona virus, Infection, Virus Infection, Quaratine, Testicular Cancer
#include abdominal influenza, cold, ill, covid, influenza, fever, quarantine, infection

x5year['injurytype_inflammation'] = 0 #Inflammation

x5year['injurytype_knee'] = 0 #Knee Injury, Knee Problems, Knee inflammation, Bruised Knee, Ruptured intraarticular ligament initiation in knee,

x5year['injurytype_knock'] = 0 #Knock, Minor Knock,

x5year['injurytype_leg'] = 0 #Leg Injury

x5year['injurytype_muscular'] = 0 #Muscular problems, Muscle Injury, Biceps femoris muscle injury, Torn Muscle Fibre, Muscle Fatigue

x5year['injurytype_pelvis'] = 0 #Pubitis, bruised pelvis, Pelvis Injury, Pubis bone irritation

x5year['injurytype_rest'] = 0 #Rest

x5year['injurytype_rib'] = 0 #Fractured Rib

x5year['injurytype_spine'] = 0 #Lumbar Vertebra Fracture, Lumbar vertebrae problems, Blockade in the spinal, Cervical spine injury

x5year['injurytype_thigh'] = 0 #Thigh Muscle Strain, Thigh Problems,

x5year['injurytype_toe'] = 0 #Toe Injury

x5year['injurytype_unknown'] = 0 #Unknown Injury,

x5year['injurytype_other'] = 0 #Surgery, Contracture, Sprain, Laceration

for i, row in x1year.iterrows(): #len(x1year)
  s = row['injuryType']
  if (s == 'nan'):
    x1year.at[i, 'injurytype_nan'] += 1
  elif type(s) == str:
    injury_list = s.split(" ,")
    injury_list = injury_list[:-1]
    for injury in injury_list:
      injury = injury.strip()
      if (injury == 'Abdominal Strain' or injury == 'Abdominal muscles injury' or injury == 'Appendectomy' or injury == 'Umbilical hernia'):
        x1year.at[i, 'injurytype_abdomen'] += 1
      elif (injury == 'Tear in the abductor muscle'):
        x1year.at[i, 'injurytype_abductor'] += 1
      elif (injury == 'Achilles tendon rupture' or injury == 'Achilles tendon problems' or injury == 'Achilles Irritation' or injury == 'Achilles tendon surgery'):
        x1year.at[i, 'injurytype_achilles'] += 1
      elif (injury == 'Adductor problems'):
        x1year.at[i, 'injurytype_adductor'] += 1
      elif (injury == 'Ankle Injury' or injury == 'Sprained ankle' or injury == 'Ankle problems' or injury == 'Torn ankle ligament' or injury == 'Ankle Surgery' or injury == 'Bruised Ankle' or injury == 'Distortion of the ankle' or injury == 'Ruptured syndesmotic ligament' or injury == 'Torn Ankle Ligament' or injury == 'Fracture-dislocation of the ankle' or injury == 'Sprained Ankle' or injury == 'Injury to the ankle' or injury == 'Ankle fracture' or injury == 'Ankle Inflammation' or injury == 'Syndesmotic ligament tear' or injury == 'Ankle Fracture' or injury == 'Capsular rupture in the ankle' or injury == 'Ruptured ankle ligament' or injury == 'Ruptured ankle aigament' or injury == 'Bruise on ankle' or injury == 'Ruptured intraarticular ligament initiation in the ankle' or injury == 'Peroneus tendon injury'):
        x1year.at[i, 'injurytype_ankle'] += 1
      elif (injury == 'Arm Injury' or injury == 'Shoulder Injury' or injury == 'Shoulder fracture' or injury == 'Elbow Injury' or injury == 'Bruised Acromioclavicular' or injury == 'Fractured Arm' or injury == 'Acromioclavicular Separation'):
        x1year.at[i, 'injurytype_arm'] += 1
      elif (injury == 'Back Injury' or injury == 'Back trouble' or injury == 'Back bruise' or injury == 'Lumbago'):
        x1year.at[i, 'injurytype_back'] += 1
      elif (injury == 'Bruise' or injury == 'Bruised Rib' or injury == 'Metatarsal bone bruise' or injury == 'Bone Bruise' or injury == 'Muscle bruise'):
        x1year.at[i, 'injurytype_bruise'] += 1
      elif (injury == 'Calf Problems' or injury == 'Calf Injury' or injury == 'Calf Strain' or injury == 'Disrupted Calf Muscle' or injury == 'Hairline crack in calfbone' or injury == 'Calf muscle strain'):
        x1year.at[i, 'injurytype_calf'] += 1
      elif (injury == 'Angina' or injury == 'Chest injury' or injury == 'Pneumothorax' or injury == 'Lung contusion'):
        x1year.at[i, 'injurytype_chest'] += 1
      elif (injury == 'Collarbone fracture'):
        x1year.at[i, 'injurytype_collarbone'] += 1
      elif (injury == 'Concussion'):
        x1year.at[i, 'injurytype_concussion'] += 1
      elif (injury == 'Cruciate Ligament Rupture' or injury == 'Cruciate Ligament Surgery' or injury == 'Cruciate Ligament Strain' or injury == 'Cruciate Ligament Injury' or injury == 'Ruptured cruciate ligament' or injury == 'Partial damage to the cruciate ligament' or injury == 'Cruciate ligament stretch'):
        x1year.at[i, 'injurytype_cruciateligament'] += 1
      elif (injury == 'Facial Injury' or injury == 'Nose surgery' or injury == 'Dental Surgery' or injury == 'Nasal Bone Fracture' or injury == 'Facial Fracture' or injury == 'Cheekbone Fracture' or injury == 'Eye Injury' or injury == 'Toothache' or injury == 'Frontal bone fracture' or injury == 'Fractured Jaw' or injury == 'Fractured Skull' or injury == 'Fracture of the orbit' or injury == 'Tooth Inflammation' or injury == 'Nose Injury'):
        x1year.at[i, 'injurytype_face'] += 1
      elif (injury == 'Finger Injury'):
        x1year.at[i, 'injurytype_finger'] += 1
      elif (injury == 'Fitness'):
        x1year.at[i, 'injurytype_fitness'] += 1
      elif (injury == 'Foot Injury' or injury == 'Bruised Foot' or injury == 'Fractured Foot' or injury == 'Arch pain' or injury == 'Heel Injury' or injury == 'Hell pain' or injury == 'Heelspur' or injury == 'Heel Bone Injury' or injury == 'Metatarsal Fracture' or injury == 'Heel pain' or injury == 'Plantar fascia' or injury == 'Foot surgery' or injury == 'Hairline crack in the foot' or injury == 'Partial demolition of the plantar fascia'):
        x1year.at[i, 'injurytype_foot'] += 1
      elif (injury == 'groin strain' or injury == 'Groin Strain' or injury == 'Groin Injury' or injury == 'Groin Surgery' or injury == 'Inguinal Hernia' or injury == 'Pubalgia' or injury == 'Testicular disruption'):
        x1year.at[i, 'injurytype_groin'] += 1
      elif (injury == 'Hamstring Injury' or injury == 'Pulled hamstring at the adductors' or injury == 'Hamstring contusion'):
        x1year.at[i, 'injurytype_hamstring'] += 1
      elif (injury == 'Hand Injury' or injury == 'Fractured Hand' or injury == 'Broken wrist' or injury == 'Wirst Injury' or injury == 'scaphoid operation' or injury == 'Hand fracture' or injury == 'Fractured Finger' or injury == 'Thumb Injury'):
        x1year.at[i, 'injurytype_hand'] += 1
      elif (injury == 'Head Injury' or injury == 'Neck Injury' or injury == 'Neck bruise'):
        x1year.at[i, 'injurytype_head'] += 1
      elif (injury == 'Heart Condition'):
        x1year.at[i, 'injurytype_heart'] += 1
      elif (injury == 'Hip problems' or injury == 'Hip Injury' or injury == 'Problems with the hip flexor' or injury == 'Bruised Hip' or injury == 'Pubis bone contusion' or injury == 'Problems with the right hip flexor'):
        x1year.at[i, 'injurytype_hip'] += 1
      elif (injury == 'Cold' or injury == 'Influenza' or injury == 'Flu' or injury == 'Abdominal Influenza' or injury == 'Ill' or injury == 'Corona virus' or injury == 'Infection' or injury == 'Virus Infection' or injury == 'Quarantine' or injury == 'Fever' or injury == 'Tonsillitis' or injury == 'Testicular Cancer' or injury == 'Pneumonia' or injury == 'Gastric problems' or injury == 'Mononucleosis' or injury == 'Chickenpox' or injury == 'Bronchitis' or injury == 'Stomach complaints' or injury == 'Food Poisoning' or injury == 'intestial virus' or injury == 'Infected wound' or injury == 'Malaria' or injury == 'cancer' or injury == 'Lymphoma' or injury == 'Depression' or injury == 'Cals Sclerosis' or injury == 'Kidney problems'):
        x1year.at[i, 'injurytype_ill'] += 1
      elif (injury == 'Inflammation'):
        x1year.at[i, 'injurytype_inflammation'] += 1
      elif (injury == 'Knee Injury' or injury == 'Knee Problems' or injury == 'Knee inflammation' or injury == 'Bruised Knee' or injury == 'Ruptured intraarticular ligament initiation in knee' or injury == 'Knee Surgery' or injury == 'Medial Collateral Ligament Tear' or injury == 'Medial Collateral Ligament Injury' or injury == 'Meniscal Injury' or injury == 'Sideband strain in the knee' or injury == 'Patella tendon irritation' or injury == 'Mensical Laceration' or injury == 'Twisted knee' or injury == 'Torn Knee Ligament' or injury == 'Medial Collateral Ligament Knee Injury' or injury == 'Patella problems' or injury == 'Ruptured lateral collateral ligament' or injury == 'Edema in the knee' or injury == 'Fractured Kneecap' or injury == 'Torn lateral collateral ligament' or injury == 'Torn Meniscus' or injury == 'Torn Collateral Ligament' or injury == 'Patella rupture' or injury == 'Torn knee ligament' or injury == 'Patella tendon luxation' or injury == 'Ruptured knee ligament' or injury == 'Rupture of Outer Meniscus' or injury == 'Double Torn Ligament' or injury == 'Meniscus Damage' or injury == 'Medial Collateral Ligament avulsion' or injury == 'Meniscus irritation' or injury == 'Inflamed ligaments of the knee' or injury == 'Rupture of the pattella'):
        x1year.at[i, 'injurytype_knee'] += 1
      elif (injury == 'Knock' or injury == 'Minor Knock' or injury == 'Dead Leg' or injury == 'Stiffness'):
        x1year.at[i, 'injurytype_knock'] += 1
      elif (injury == 'Leg Injury' or injury == 'Strain in the thigh and gluteal muscles' or injury == 'Shinbone injury' or injury == 'Shin bone bruise' or injury == 'Biceps femoris muscle injury' or injury == 'Fibula Fracture' or injury == 'Fractured Leg' or injury == 'Fissure of the fibula' or injury == 'Sciatic Problem' or injury == 'Tibia and Fibula Fracture' or injury == 'Fracture of the lower leg' or injury == 'Tibia Fracture' or injury == 'Inflamed head of fibula'):
        x1year.at[i, 'injurytype_leg'] += 1
      elif (injury == 'Muscular problems' or injury == 'Muscle Injury' or injury == 'Torn Muscle Fibre' or injury == 'Muscle Fatigue' or injury == 'Torn muscle bundle' or injury == 'Torn Muscle' or injury == 'Muscle fiber tear'):
        x1year.at[i, 'injurytype_muscular'] += 1
      elif (injury == 'Pubitis' or injury == 'Bruised pelvis' or injury == 'Pelvis Injury' or injury == 'Pubis bone irritation' or injury == 'bruised pelvis' or injury == 'bruised pelvis' or injury == 'Pelvic obliquity'):
        x1year.at[i, 'injurytype_pelvis'] += 1
      elif (injury == 'Rest'):
        x1year.at[i, 'injurytype_rest'] += 1
      elif (injury == 'Fractured Rib'):
        x1year.at[i, 'injurytype_rib'] += 1
      elif (injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar vertebrae problems' or injury == 'Blockade in the spinal' or injury == 'Cervical spine injury' or injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar Vertebra Fracture' or injury == 'Vertebra injury'):
        x1year.at[i, 'injurytype_spine'] += 1
      elif (injury == 'Thigh Muscle Strain' or injury == 'Thigh Problems' or injury == 'Thigh muscle rupture'):
        x1year.at[i, 'injurytype_thigh'] += 1
      elif (injury == 'Toe Injury' or injury == 'Fractured Toe'):
        x1year.at[i, 'injurytype_toe'] += 1
      elif (injury == 'Unknown Injury'):
        x1year.at[i, 'injurytype_unknown'] += 1
      elif (injury == 'Surgery' or injury == 'Contracture' or injury == 'Sprain' or injury == 'Laceration' or injury == 'Fracture' or injury == 'Ligament Injury' or injury == 'Stress response of the bone' or injury == 'Strain' or injury == 'Contused laceration' or injury == 'Muscle partial avulsion' or injury == 'torn tendon' or injury == 'Tear in a joint capsule' or injury == 'tendon irritation' or injury == 'Ligament Problems' or injury == 'Sideband injury' or injury == 'Bone buckling' or injury == 'Stretched Ligament' or injury == 'Pinched nerve' or injury == 'Overstretching' or injury == 'Tendonitis' or injury == 'laceration' or injury == 'Arthroscopie' or injury == 'Herniated Disc' or injury == 'Fatigue fracture' or injury == 'Torn Ligament' or injury == 'Cartilage Damage' or injury == 'Bursitis' or injury == 'Circulation Problems' or injury == 'Tendon crack' or injury == 'Cut' or injury == 'sprain' or injury == 'Burns' or injury == 'open wound' or injury == 'Sideband tear' or injury == 'Marrow bulge' or injury == 'Compartment syndrome' or injury == 'Vestibular disorder' or injury == 'Intraarticular ligament fissure' or injury == 'Ruptured ligaments' or injury == 'Muscular hairline crack' or injury == 'capsular tear' or injury == 'Traffic Accident' or injury == 'Flesh Wound'):
        x1year.at[i, 'injurytype_other'] += 1
      else:
        print(injury)

for i, row in x2year.iterrows(): #len(x1year)
  s = row['injuryType']
  if (s == 'nan'):
    x2year.at[i, 'injurytype_nan'] += 1
  elif type(s) == str:
    injury_list = s.split(" ,")
    injury_list = injury_list[:-1]
    for injury in injury_list:
      injury = injury.strip()
      if (injury == 'Abdominal Strain' or injury == 'Abdominal muscles injury' or injury == 'Appendectomy' or injury == 'Umbilical hernia'):
        x2year.at[i, 'injurytype_abdomen'] += 1
      elif (injury == 'Tear in the abductor muscle'):
        x2year.at[i, 'injurytype_abductor'] += 1
      elif (injury == 'Achilles tendon rupture' or injury == 'Achilles tendon problems' or injury == 'Achilles Irritation' or injury == 'Achilles tendon surgery'):
        x2year.at[i, 'injurytype_achilles'] += 1
      elif (injury == 'Adductor problems'):
        x2year.at[i, 'injurytype_adductor'] += 1
      elif (injury == 'Ankle Injury' or injury == 'Sprained ankle' or injury == 'Ankle problems' or injury == 'Torn ankle ligament' or injury == 'Ankle Surgery' or injury == 'Bruised Ankle' or injury == 'Distortion of the ankle' or injury == 'Ruptured syndesmotic ligament' or injury == 'Torn Ankle Ligament' or injury == 'Fracture-dislocation of the ankle' or injury == 'Sprained Ankle' or injury == 'Injury to the ankle' or injury == 'Ankle fracture' or injury == 'Ankle Inflammation' or injury == 'Syndesmotic ligament tear' or injury == 'Ankle Fracture' or injury == 'Capsular rupture in the ankle' or injury == 'Ruptured ankle ligament' or injury == 'Ruptured ankle aigament' or injury == 'Bruise on ankle' or injury == 'Ruptured intraarticular ligament initiation in the ankle' or injury == 'Peroneus tendon injury'):
        x2year.at[i, 'injurytype_ankle'] += 1
      elif (injury == 'Arm Injury' or injury == 'Shoulder Injury' or injury == 'Shoulder fracture' or injury == 'Elbow Injury' or injury == 'Bruised Acromioclavicular' or injury == 'Fractured Arm' or injury == 'Acromioclavicular Separation'):
        x2year.at[i, 'injurytype_arm'] += 1
      elif (injury == 'Back Injury' or injury == 'Back trouble' or injury == 'Back bruise' or injury == 'Lumbago'):
        x2year.at[i, 'injurytype_back'] += 1
      elif (injury == 'Bruise' or injury == 'Bruised Rib' or injury == 'Metatarsal bone bruise' or injury == 'Bone Bruise' or injury == 'Muscle bruise'):
        x2year.at[i, 'injurytype_bruise'] += 1
      elif (injury == 'Calf Problems' or injury == 'Calf Injury' or injury == 'Calf Strain' or injury == 'Disrupted Calf Muscle' or injury == 'Hairline crack in calfbone' or injury == 'Calf muscle strain'):
        x2year.at[i, 'injurytype_calf'] += 1
      elif (injury == 'Angina' or injury == 'Chest injury' or injury == 'Pneumothorax' or injury == 'Lung contusion'):
        x2year.at[i, 'injurytype_chest'] += 1
      elif (injury == 'Collarbone fracture'):
        x2year.at[i, 'injurytype_collarbone'] += 1
      elif (injury == 'Concussion'):
        x2year.at[i, 'injurytype_concussion'] += 1
      elif (injury == 'Cruciate Ligament Rupture' or injury == 'Cruciate Ligament Surgery' or injury == 'Cruciate Ligament Strain' or injury == 'Cruciate Ligament Injury' or injury == 'Ruptured cruciate ligament' or injury == 'Partial damage to the cruciate ligament' or injury == 'Cruciate ligament stretch'):
        x2year.at[i, 'injurytype_cruciateligament'] += 1
      elif (injury == 'Facial Injury' or injury == 'Nose surgery' or injury == 'Dental Surgery' or injury == 'Nasal Bone Fracture' or injury == 'Facial Fracture' or injury == 'Cheekbone Fracture' or injury == 'Eye Injury' or injury == 'Toothache' or injury == 'Frontal bone fracture' or injury == 'Fractured Jaw' or injury == 'Fractured Skull' or injury == 'Fracture of the orbit' or injury == 'Tooth Inflammation' or injury == 'Nose Injury'):
        x2year.at[i, 'injurytype_face'] += 1
      elif (injury == 'Finger Injury'):
        x2year.at[i, 'injurytype_finger'] += 1
      elif (injury == 'Fitness'):
        x2year.at[i, 'injurytype_fitness'] += 1
      elif (injury == 'Foot Injury' or injury == 'Bruised Foot' or injury == 'Fractured Foot' or injury == 'Arch pain' or injury == 'Heel Injury' or injury == 'Hell pain' or injury == 'Heelspur' or injury == 'Heel Bone Injury' or injury == 'Metatarsal Fracture' or injury == 'Heel pain' or injury == 'Plantar fascia' or injury == 'Foot surgery' or injury == 'Hairline crack in the foot' or injury == 'Partial demolition of the plantar fascia'):
        x2year.at[i, 'injurytype_foot'] += 1
      elif (injury == 'groin strain' or injury == 'Groin Strain' or injury == 'Groin Injury' or injury == 'Groin Surgery' or injury == 'Inguinal Hernia' or injury == 'Pubalgia' or injury == 'Testicular disruption'):
        x2year.at[i, 'injurytype_groin'] += 1
      elif (injury == 'Hamstring Injury' or injury == 'Pulled hamstring at the adductors' or injury == 'Hamstring contusion'):
        x2year.at[i, 'injurytype_hamstring'] += 1
      elif (injury == 'Hand Injury' or injury == 'Fractured Hand' or injury == 'Broken wrist' or injury == 'Wirst Injury' or injury == 'scaphoid operation' or injury == 'Hand fracture' or injury == 'Fractured Finger' or injury == 'Thumb Injury'):
        x2year.at[i, 'injurytype_hand'] += 1
      elif (injury == 'Head Injury' or injury == 'Neck Injury' or injury == 'Neck bruise'):
        x2year.at[i, 'injurytype_head'] += 1
      elif (injury == 'Heart Condition'):
        x2year.at[i, 'injurytype_heart'] += 1
      elif (injury == 'Hip problems' or injury == 'Hip Injury' or injury == 'Problems with the hip flexor' or injury == 'Bruised Hip' or injury == 'Pubis bone contusion' or injury == 'Problems with the right hip flexor'):
        x2year.at[i, 'injurytype_hip'] += 1
      elif (injury == 'Cold' or injury == 'Influenza' or injury == 'Flu' or injury == 'Abdominal Influenza' or injury == 'Ill' or injury == 'Corona virus' or injury == 'Infection' or injury == 'Virus Infection' or injury == 'Quarantine' or injury == 'Fever' or injury == 'Tonsillitis' or injury == 'Testicular Cancer' or injury == 'Pneumonia' or injury == 'Gastric problems' or injury == 'Mononucleosis' or injury == 'Chickenpox' or injury == 'Bronchitis' or injury == 'Stomach complaints' or injury == 'Food Poisoning' or injury == 'intestial virus' or injury == 'Infected wound' or injury == 'Malaria' or injury == 'cancer' or injury == 'Lymphoma' or injury == 'Depression' or injury == 'Cals Sclerosis' or injury == 'Kidney problems'):
        x2year.at[i, 'injurytype_ill'] += 1
      elif (injury == 'Inflammation'):
        x2year.at[i, 'injurytype_inflammation'] += 1
      elif (injury == 'Knee Injury' or injury == 'Knee Problems' or injury == 'Knee inflammation' or injury == 'Bruised Knee' or injury == 'Ruptured intraarticular ligament initiation in knee' or injury == 'Knee Surgery' or injury == 'Medial Collateral Ligament Tear' or injury == 'Medial Collateral Ligament Injury' or injury == 'Meniscal Injury' or injury == 'Sideband strain in the knee' or injury == 'Patella tendon irritation' or injury == 'Mensical Laceration' or injury == 'Twisted knee' or injury == 'Torn Knee Ligament' or injury == 'Medial Collateral Ligament Knee Injury' or injury == 'Patella problems' or injury == 'Ruptured lateral collateral ligament' or injury == 'Edema in the knee' or injury == 'Fractured Kneecap' or injury == 'Torn lateral collateral ligament' or injury == 'Torn Meniscus' or injury == 'Torn Collateral Ligament' or injury == 'Patella rupture' or injury == 'Torn knee ligament' or injury == 'Patella tendon luxation' or injury == 'Ruptured knee ligament' or injury == 'Rupture of Outer Meniscus' or injury == 'Double Torn Ligament' or injury == 'Meniscus Damage' or injury == 'Medial Collateral Ligament avulsion' or injury == 'Meniscus irritation' or injury == 'Inflamed ligaments of the knee' or injury == 'Rupture of the pattella'):
        x2year.at[i, 'injurytype_knee'] += 1
      elif (injury == 'Knock' or injury == 'Minor Knock' or injury == 'Dead Leg' or injury == 'Stiffness'):
        x2year.at[i, 'injurytype_knock'] += 1
      elif (injury == 'Leg Injury' or injury == 'Strain in the thigh and gluteal muscles' or injury == 'Shinbone injury' or injury == 'Shin bone bruise' or injury == 'Biceps femoris muscle injury' or injury == 'Fibula Fracture' or injury == 'Fractured Leg' or injury == 'Fissure of the fibula' or injury == 'Sciatic Problem' or injury == 'Tibia and Fibula Fracture' or injury == 'Fracture of the lower leg' or injury == 'Tibia Fracture' or injury == 'Inflamed head of fibula'):
        x2year.at[i, 'injurytype_leg'] += 1
      elif (injury == 'Muscular problems' or injury == 'Muscle Injury' or injury == 'Torn Muscle Fibre' or injury == 'Muscle Fatigue' or injury == 'Torn muscle bundle' or injury == 'Torn Muscle' or injury == 'Muscle fiber tear'):
        x2year.at[i, 'injurytype_muscular'] += 1
      elif (injury == 'Pubitis' or injury == 'Bruised pelvis' or injury == 'Pelvis Injury' or injury == 'Pubis bone irritation' or injury == 'bruised pelvis' or injury == 'bruised pelvis' or injury == 'Pelvic obliquity'):
        x2year.at[i, 'injurytype_pelvis'] += 1
      elif (injury == 'Rest'):
        x2year.at[i, 'injurytype_rest'] += 1
      elif (injury == 'Fractured Rib'):
        x2year.at[i, 'injurytype_rib'] += 1
      elif (injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar vertebrae problems' or injury == 'Blockade in the spinal' or injury == 'Cervical spine injury' or injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar Vertebra Fracture' or injury == 'Vertebra injury'):
        x2year.at[i, 'injurytype_spine'] += 1
      elif (injury == 'Thigh Muscle Strain' or injury == 'Thigh Problems' or injury == 'Thigh muscle rupture'):
        x2year.at[i, 'injurytype_thigh'] += 1
      elif (injury == 'Toe Injury' or injury == 'Fractured Toe'):
        x2year.at[i, 'injurytype_toe'] += 1
      elif (injury == 'Unknown Injury'):
        x2year.at[i, 'injurytype_unknown'] += 1
      elif (injury == 'Surgery' or injury == 'Contracture' or injury == 'Sprain' or injury == 'Laceration' or injury == 'Fracture' or injury == 'Ligament Injury' or injury == 'Stress response of the bone' or injury == 'Strain' or injury == 'Contused laceration' or injury == 'Muscle partial avulsion' or injury == 'torn tendon' or injury == 'Tear in a joint capsule' or injury == 'tendon irritation' or injury == 'Ligament Problems' or injury == 'Sideband injury' or injury == 'Bone buckling' or injury == 'Stretched Ligament' or injury == 'Pinched nerve' or injury == 'Overstretching' or injury == 'Tendonitis' or injury == 'laceration' or injury == 'Arthroscopie' or injury == 'Herniated Disc' or injury == 'Fatigue fracture' or injury == 'Torn Ligament' or injury == 'Cartilage Damage' or injury == 'Bursitis' or injury == 'Circulation Problems' or injury == 'Tendon crack' or injury == 'Cut' or injury == 'sprain' or injury == 'Burns' or injury == 'open wound' or injury == 'Sideband tear' or injury == 'Marrow bulge' or injury == 'Compartment syndrome' or injury == 'Vestibular disorder' or injury == 'Intraarticular ligament fissure' or injury == 'Ruptured ligaments' or injury == 'Muscular hairline crack' or injury == 'capsular tear' or injury == 'Traffic Accident' or injury == 'Flesh Wound'):
        x2year.at[i, 'injurytype_other'] += 1
      else:
        print(injury)

for i, row in x3year.iterrows(): #len(x1year)
  s = row['injuryType']
  if (s == 'nan'):
    x3year.at[i, 'injurytype_nan'] += 1
  elif type(s) == str:
    injury_list = s.split(" ,")
    injury_list = injury_list[:-1]
    for injury in injury_list:
      injury = injury.strip()
      if (injury == 'Abdominal Strain' or injury == 'Abdominal muscles injury' or injury == 'Appendectomy' or injury == 'Umbilical hernia'):
        x3year.at[i, 'injurytype_abdomen'] += 1
      elif (injury == 'Tear in the abductor muscle'):
        x3year.at[i, 'injurytype_abductor'] += 1
      elif (injury == 'Achilles tendon rupture' or injury == 'Achilles tendon problems' or injury == 'Achilles Irritation' or injury == 'Achilles tendon surgery'):
        x3year.at[i, 'injurytype_achilles'] += 1
      elif (injury == 'Adductor problems'):
        x3year.at[i, 'injurytype_adductor'] += 1
      elif (injury == 'Ankle Injury' or injury == 'Sprained ankle' or injury == 'Ankle problems' or injury == 'Torn ankle ligament' or injury == 'Ankle Surgery' or injury == 'Bruised Ankle' or injury == 'Distortion of the ankle' or injury == 'Ruptured syndesmotic ligament' or injury == 'Torn Ankle Ligament' or injury == 'Fracture-dislocation of the ankle' or injury == 'Sprained Ankle' or injury == 'Injury to the ankle' or injury == 'Ankle fracture' or injury == 'Ankle Inflammation' or injury == 'Syndesmotic ligament tear' or injury == 'Ankle Fracture' or injury == 'Capsular rupture in the ankle' or injury == 'Ruptured ankle ligament' or injury == 'Ruptured ankle aigament' or injury == 'Bruise on ankle' or injury == 'Ruptured intraarticular ligament initiation in the ankle' or injury == 'Peroneus tendon injury'):
        x3year.at[i, 'injurytype_ankle'] += 1
      elif (injury == 'Arm Injury' or injury == 'Shoulder Injury' or injury == 'Shoulder fracture' or injury == 'Elbow Injury' or injury == 'Bruised Acromioclavicular' or injury == 'Fractured Arm' or injury == 'Acromioclavicular Separation'):
        x3year.at[i, 'injurytype_arm'] += 1
      elif (injury == 'Back Injury' or injury == 'Back trouble' or injury == 'Back bruise' or injury == 'Lumbago'):
        x3year.at[i, 'injurytype_back'] += 1
      elif (injury == 'Bruise' or injury == 'Bruised Rib' or injury == 'Metatarsal bone bruise' or injury == 'Bone Bruise' or injury == 'Muscle bruise'):
        x3year.at[i, 'injurytype_bruise'] += 1
      elif (injury == 'Calf Problems' or injury == 'Calf Injury' or injury == 'Calf Strain' or injury == 'Disrupted Calf Muscle' or injury == 'Hairline crack in calfbone' or injury == 'Calf muscle strain'):
        x3year.at[i, 'injurytype_calf'] += 1
      elif (injury == 'Angina' or injury == 'Chest injury' or injury == 'Pneumothorax' or injury == 'Lung contusion'):
        x3year.at[i, 'injurytype_chest'] += 1
      elif (injury == 'Collarbone fracture'):
        x3year.at[i, 'injurytype_collarbone'] += 1
      elif (injury == 'Concussion'):
        x3year.at[i, 'injurytype_concussion'] += 1
      elif (injury == 'Cruciate Ligament Rupture' or injury == 'Cruciate Ligament Surgery' or injury == 'Cruciate Ligament Strain' or injury == 'Cruciate Ligament Injury' or injury == 'Ruptured cruciate ligament' or injury == 'Partial damage to the cruciate ligament' or injury == 'Cruciate ligament stretch'):
        x3year.at[i, 'injurytype_cruciateligament'] += 1
      elif (injury == 'Facial Injury' or injury == 'Nose surgery' or injury == 'Dental Surgery' or injury == 'Nasal Bone Fracture' or injury == 'Facial Fracture' or injury == 'Cheekbone Fracture' or injury == 'Eye Injury' or injury == 'Toothache' or injury == 'Frontal bone fracture' or injury == 'Fractured Jaw' or injury == 'Fractured Skull' or injury == 'Fracture of the orbit' or injury == 'Tooth Inflammation' or injury == 'Nose Injury'):
        x3year.at[i, 'injurytype_face'] += 1
      elif (injury == 'Finger Injury'):
        x3year.at[i, 'injurytype_finger'] += 1
      elif (injury == 'Fitness'):
        x3year.at[i, 'injurytype_fitness'] += 1
      elif (injury == 'Foot Injury' or injury == 'Bruised Foot' or injury == 'Fractured Foot' or injury == 'Arch pain' or injury == 'Heel Injury' or injury == 'Hell pain' or injury == 'Heelspur' or injury == 'Heel Bone Injury' or injury == 'Metatarsal Fracture' or injury == 'Heel pain' or injury == 'Plantar fascia' or injury == 'Foot surgery' or injury == 'Hairline crack in the foot' or injury == 'Partial demolition of the plantar fascia'):
        x3year.at[i, 'injurytype_foot'] += 1
      elif (injury == 'groin strain' or injury == 'Groin Strain' or injury == 'Groin Injury' or injury == 'Groin Surgery' or injury == 'Inguinal Hernia' or injury == 'Pubalgia' or injury == 'Testicular disruption'):
        x3year.at[i, 'injurytype_groin'] += 1
      elif (injury == 'Hamstring Injury' or injury == 'Pulled hamstring at the adductors' or injury == 'Hamstring contusion'):
        x3year.at[i, 'injurytype_hamstring'] += 1
      elif (injury == 'Hand Injury' or injury == 'Fractured Hand' or injury == 'Broken wrist' or injury == 'Wirst Injury' or injury == 'scaphoid operation' or injury == 'Hand fracture' or injury == 'Fractured Finger' or injury == 'Thumb Injury'):
        x3year.at[i, 'injurytype_hand'] += 1
      elif (injury == 'Head Injury' or injury == 'Neck Injury' or injury == 'Neck bruise'):
        x3year.at[i, 'injurytype_head'] += 1
      elif (injury == 'Heart Condition'):
        x3year.at[i, 'injurytype_heart'] += 1
      elif (injury == 'Hip problems' or injury == 'Hip Injury' or injury == 'Problems with the hip flexor' or injury == 'Bruised Hip' or injury == 'Pubis bone contusion' or injury == 'Problems with the right hip flexor'):
        x3year.at[i, 'injurytype_hip'] += 1
      elif (injury == 'Cold' or injury == 'Influenza' or injury == 'Flu' or injury == 'Abdominal Influenza' or injury == 'Ill' or injury == 'Corona virus' or injury == 'Infection' or injury == 'Virus Infection' or injury == 'Quarantine' or injury == 'Fever' or injury == 'Tonsillitis' or injury == 'Testicular Cancer' or injury == 'Pneumonia' or injury == 'Gastric problems' or injury == 'Mononucleosis' or injury == 'Chickenpox' or injury == 'Bronchitis' or injury == 'Stomach complaints' or injury == 'Food Poisoning' or injury == 'intestial virus' or injury == 'Infected wound' or injury == 'Malaria' or injury == 'cancer' or injury == 'Lymphoma' or injury == 'Depression' or injury == 'Cals Sclerosis' or injury == 'Kidney problems'):
        x3year.at[i, 'injurytype_ill'] += 1
      elif (injury == 'Inflammation'):
        x3year.at[i, 'injurytype_inflammation'] += 1
      elif (injury == 'Knee Injury' or injury == 'Knee Problems' or injury == 'Knee inflammation' or injury == 'Bruised Knee' or injury == 'Ruptured intraarticular ligament initiation in knee' or injury == 'Knee Surgery' or injury == 'Medial Collateral Ligament Tear' or injury == 'Medial Collateral Ligament Injury' or injury == 'Meniscal Injury' or injury == 'Sideband strain in the knee' or injury == 'Patella tendon irritation' or injury == 'Mensical Laceration' or injury == 'Twisted knee' or injury == 'Torn Knee Ligament' or injury == 'Medial Collateral Ligament Knee Injury' or injury == 'Patella problems' or injury == 'Ruptured lateral collateral ligament' or injury == 'Edema in the knee' or injury == 'Fractured Kneecap' or injury == 'Torn lateral collateral ligament' or injury == 'Torn Meniscus' or injury == 'Torn Collateral Ligament' or injury == 'Patella rupture' or injury == 'Torn knee ligament' or injury == 'Patella tendon luxation' or injury == 'Ruptured knee ligament' or injury == 'Rupture of Outer Meniscus' or injury == 'Double Torn Ligament' or injury == 'Meniscus Damage' or injury == 'Medial Collateral Ligament avulsion' or injury == 'Meniscus irritation' or injury == 'Inflamed ligaments of the knee' or injury == 'Rupture of the pattella'):
        x3year.at[i, 'injurytype_knee'] += 1
      elif (injury == 'Knock' or injury == 'Minor Knock' or injury == 'Dead Leg' or injury == 'Stiffness'):
        x3year.at[i, 'injurytype_knock'] += 1
      elif (injury == 'Leg Injury' or injury == 'Strain in the thigh and gluteal muscles' or injury == 'Shinbone injury' or injury == 'Shin bone bruise' or injury == 'Biceps femoris muscle injury' or injury == 'Fibula Fracture' or injury == 'Fractured Leg' or injury == 'Fissure of the fibula' or injury == 'Sciatic Problem' or injury == 'Tibia and Fibula Fracture' or injury == 'Fracture of the lower leg' or injury == 'Tibia Fracture' or injury == 'Inflamed head of fibula'):
        x3year.at[i, 'injurytype_leg'] += 1
      elif (injury == 'Muscular problems' or injury == 'Muscle Injury' or injury == 'Torn Muscle Fibre' or injury == 'Muscle Fatigue' or injury == 'Torn muscle bundle' or injury == 'Torn Muscle' or injury == 'Muscle fiber tear'):
        x3year.at[i, 'injurytype_muscular'] += 1
      elif (injury == 'Pubitis' or injury == 'Bruised pelvis' or injury == 'Pelvis Injury' or injury == 'Pubis bone irritation' or injury == 'bruised pelvis' or injury == 'bruised pelvis' or injury == 'Pelvic obliquity'):
        x3year.at[i, 'injurytype_pelvis'] += 1
      elif (injury == 'Rest'):
        x3year.at[i, 'injurytype_rest'] += 1
      elif (injury == 'Fractured Rib'):
        x3year.at[i, 'injurytype_rib'] += 1
      elif (injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar vertebrae problems' or injury == 'Blockade in the spinal' or injury == 'Cervical spine injury' or injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar Vertebra Fracture' or injury == 'Vertebra injury'):
        x3year.at[i, 'injurytype_spine'] += 1
      elif (injury == 'Thigh Muscle Strain' or injury == 'Thigh Problems' or injury == 'Thigh muscle rupture'):
        x3year.at[i, 'injurytype_thigh'] += 1
      elif (injury == 'Toe Injury' or injury == 'Fractured Toe'):
        x3year.at[i, 'injurytype_toe'] += 1
      elif (injury == 'Unknown Injury'):
        x3year.at[i, 'injurytype_unknown'] += 1
      elif (injury == 'Surgery' or injury == 'Contracture' or injury == 'Sprain' or injury == 'Laceration' or injury == 'Fracture' or injury == 'Ligament Injury' or injury == 'Stress response of the bone' or injury == 'Strain' or injury == 'Contused laceration' or injury == 'Muscle partial avulsion' or injury == 'torn tendon' or injury == 'Tear in a joint capsule' or injury == 'tendon irritation' or injury == 'Ligament Problems' or injury == 'Sideband injury' or injury == 'Bone buckling' or injury == 'Stretched Ligament' or injury == 'Pinched nerve' or injury == 'Overstretching' or injury == 'Tendonitis' or injury == 'laceration' or injury == 'Arthroscopie' or injury == 'Herniated Disc' or injury == 'Fatigue fracture' or injury == 'Torn Ligament' or injury == 'Cartilage Damage' or injury == 'Bursitis' or injury == 'Circulation Problems' or injury == 'Tendon crack' or injury == 'Cut' or injury == 'sprain' or injury == 'Burns' or injury == 'open wound' or injury == 'Sideband tear' or injury == 'Marrow bulge' or injury == 'Compartment syndrome' or injury == 'Vestibular disorder' or injury == 'Intraarticular ligament fissure' or injury == 'Ruptured ligaments' or injury == 'Muscular hairline crack' or injury == 'capsular tear' or injury == 'Traffic Accident' or injury == 'Flesh Wound'):
        x3year.at[i, 'injurytype_other'] += 1
      else:
        print(injury)

for i, row in x4year.iterrows(): #len(x1year)
  s = row['injuryType']
  if (s == 'nan'):
    x4year.at[i, 'injurytype_nan'] += 1
  elif type(s) == str:
    injury_list = s.split(" ,")
    injury_list = injury_list[:-1]
    for injury in injury_list:
      injury = injury.strip()
      if (injury == 'Abdominal Strain' or injury == 'Abdominal muscles injury' or injury == 'Appendectomy' or injury == 'Umbilical hernia'):
        x4year.at[i, 'injurytype_abdomen'] += 1
      elif (injury == 'Tear in the abductor muscle'):
        x4year.at[i, 'injurytype_abductor'] += 1
      elif (injury == 'Achilles tendon rupture' or injury == 'Achilles tendon problems' or injury == 'Achilles Irritation' or injury == 'Achilles tendon surgery'):
        x4year.at[i, 'injurytype_achilles'] += 1
      elif (injury == 'Adductor problems'):
        x4year.at[i, 'injurytype_adductor'] += 1
      elif (injury == 'Ankle Injury' or injury == 'Sprained ankle' or injury == 'Ankle problems' or injury == 'Torn ankle ligament' or injury == 'Ankle Surgery' or injury == 'Bruised Ankle' or injury == 'Distortion of the ankle' or injury == 'Ruptured syndesmotic ligament' or injury == 'Torn Ankle Ligament' or injury == 'Fracture-dislocation of the ankle' or injury == 'Sprained Ankle' or injury == 'Injury to the ankle' or injury == 'Ankle fracture' or injury == 'Ankle Inflammation' or injury == 'Syndesmotic ligament tear' or injury == 'Ankle Fracture' or injury == 'Capsular rupture in the ankle' or injury == 'Ruptured ankle ligament' or injury == 'Ruptured ankle aigament' or injury == 'Bruise on ankle' or injury == 'Ruptured intraarticular ligament initiation in the ankle' or injury == 'Peroneus tendon injury'):
        x4year.at[i, 'injurytype_ankle'] += 1
      elif (injury == 'Arm Injury' or injury == 'Shoulder Injury' or injury == 'Shoulder fracture' or injury == 'Elbow Injury' or injury == 'Bruised Acromioclavicular' or injury == 'Fractured Arm' or injury == 'Acromioclavicular Separation'):
        x4year.at[i, 'injurytype_arm'] += 1
      elif (injury == 'Back Injury' or injury == 'Back trouble' or injury == 'Back bruise' or injury == 'Lumbago'):
        x4year.at[i, 'injurytype_back'] += 1
      elif (injury == 'Bruise' or injury == 'Bruised Rib' or injury == 'Metatarsal bone bruise' or injury == 'Bone Bruise' or injury == 'Muscle bruise'):
        x4year.at[i, 'injurytype_bruise'] += 1
      elif (injury == 'Calf Problems' or injury == 'Calf Injury' or injury == 'Calf Strain' or injury == 'Disrupted Calf Muscle' or injury == 'Hairline crack in calfbone' or injury == 'Calf muscle strain'):
        x4year.at[i, 'injurytype_calf'] += 1
      elif (injury == 'Angina' or injury == 'Chest injury' or injury == 'Pneumothorax' or injury == 'Lung contusion'):
        x4year.at[i, 'injurytype_chest'] += 1
      elif (injury == 'Collarbone fracture'):
        x4year.at[i, 'injurytype_collarbone'] += 1
      elif (injury == 'Concussion'):
        x4year.at[i, 'injurytype_concussion'] += 1
      elif (injury == 'Cruciate Ligament Rupture' or injury == 'Cruciate Ligament Surgery' or injury == 'Cruciate Ligament Strain' or injury == 'Cruciate Ligament Injury' or injury == 'Ruptured cruciate ligament' or injury == 'Partial damage to the cruciate ligament' or injury == 'Cruciate ligament stretch'):
        x4year.at[i, 'injurytype_cruciateligament'] += 1
      elif (injury == 'Facial Injury' or injury == 'Nose surgery' or injury == 'Dental Surgery' or injury == 'Nasal Bone Fracture' or injury == 'Facial Fracture' or injury == 'Cheekbone Fracture' or injury == 'Eye Injury' or injury == 'Toothache' or injury == 'Frontal bone fracture' or injury == 'Fractured Jaw' or injury == 'Fractured Skull' or injury == 'Fracture of the orbit' or injury == 'Tooth Inflammation' or injury == 'Nose Injury'):
        x4year.at[i, 'injurytype_face'] += 1
      elif (injury == 'Finger Injury'):
        x4year.at[i, 'injurytype_finger'] += 1
      elif (injury == 'Fitness'):
        x4year.at[i, 'injurytype_fitness'] += 1
      elif (injury == 'Foot Injury' or injury == 'Bruised Foot' or injury == 'Fractured Foot' or injury == 'Arch pain' or injury == 'Heel Injury' or injury == 'Hell pain' or injury == 'Heelspur' or injury == 'Heel Bone Injury' or injury == 'Metatarsal Fracture' or injury == 'Heel pain' or injury == 'Plantar fascia' or injury == 'Foot surgery' or injury == 'Hairline crack in the foot' or injury == 'Partial demolition of the plantar fascia'):
        x4year.at[i, 'injurytype_foot'] += 1
      elif (injury == 'groin strain' or injury == 'Groin Strain' or injury == 'Groin Injury' or injury == 'Groin Surgery' or injury == 'Inguinal Hernia' or injury == 'Pubalgia' or injury == 'Testicular disruption'):
        x4year.at[i, 'injurytype_groin'] += 1
      elif (injury == 'Hamstring Injury' or injury == 'Pulled hamstring at the adductors' or injury == 'Hamstring contusion'):
        x4year.at[i, 'injurytype_hamstring'] += 1
      elif (injury == 'Hand Injury' or injury == 'Fractured Hand' or injury == 'Broken wrist' or injury == 'Wirst Injury' or injury == 'scaphoid operation' or injury == 'Hand fracture' or injury == 'Fractured Finger' or injury == 'Thumb Injury'):
        x4year.at[i, 'injurytype_hand'] += 1
      elif (injury == 'Head Injury' or injury == 'Neck Injury' or injury == 'Neck bruise'):
        x4year.at[i, 'injurytype_head'] += 1
      elif (injury == 'Heart Condition'):
        x4year.at[i, 'injurytype_heart'] += 1
      elif (injury == 'Hip problems' or injury == 'Hip Injury' or injury == 'Problems with the hip flexor' or injury == 'Bruised Hip' or injury == 'Pubis bone contusion' or injury == 'Problems with the right hip flexor'):
        x4year.at[i, 'injurytype_hip'] += 1
      elif (injury == 'Cold' or injury == 'Influenza' or injury == 'Flu' or injury == 'Abdominal Influenza' or injury == 'Ill' or injury == 'Corona virus' or injury == 'Infection' or injury == 'Virus Infection' or injury == 'Quarantine' or injury == 'Fever' or injury == 'Tonsillitis' or injury == 'Testicular Cancer' or injury == 'Pneumonia' or injury == 'Gastric problems' or injury == 'Mononucleosis' or injury == 'Chickenpox' or injury == 'Bronchitis' or injury == 'Stomach complaints' or injury == 'Food Poisoning' or injury == 'intestial virus' or injury == 'Infected wound' or injury == 'Malaria' or injury == 'cancer' or injury == 'Lymphoma' or injury == 'Depression' or injury == 'Cals Sclerosis' or injury == 'Kidney problems'):
        x4year.at[i, 'injurytype_ill'] += 1
      elif (injury == 'Inflammation'):
        x4year.at[i, 'injurytype_inflammation'] += 1
      elif (injury == 'Knee Injury' or injury == 'Knee Problems' or injury == 'Knee inflammation' or injury == 'Bruised Knee' or injury == 'Ruptured intraarticular ligament initiation in knee' or injury == 'Knee Surgery' or injury == 'Medial Collateral Ligament Tear' or injury == 'Medial Collateral Ligament Injury' or injury == 'Meniscal Injury' or injury == 'Sideband strain in the knee' or injury == 'Patella tendon irritation' or injury == 'Mensical Laceration' or injury == 'Twisted knee' or injury == 'Torn Knee Ligament' or injury == 'Medial Collateral Ligament Knee Injury' or injury == 'Patella problems' or injury == 'Ruptured lateral collateral ligament' or injury == 'Edema in the knee' or injury == 'Fractured Kneecap' or injury == 'Torn lateral collateral ligament' or injury == 'Torn Meniscus' or injury == 'Torn Collateral Ligament' or injury == 'Patella rupture' or injury == 'Torn knee ligament' or injury == 'Patella tendon luxation' or injury == 'Ruptured knee ligament' or injury == 'Rupture of Outer Meniscus' or injury == 'Double Torn Ligament' or injury == 'Meniscus Damage' or injury == 'Medial Collateral Ligament avulsion' or injury == 'Meniscus irritation' or injury == 'Inflamed ligaments of the knee' or injury == 'Rupture of the pattella'):
        x4year.at[i, 'injurytype_knee'] += 1
      elif (injury == 'Knock' or injury == 'Minor Knock' or injury == 'Dead Leg' or injury == 'Stiffness'):
        x4year.at[i, 'injurytype_knock'] += 1
      elif (injury == 'Leg Injury' or injury == 'Strain in the thigh and gluteal muscles' or injury == 'Shinbone injury' or injury == 'Shin bone bruise' or injury == 'Biceps femoris muscle injury' or injury == 'Fibula Fracture' or injury == 'Fractured Leg' or injury == 'Fissure of the fibula' or injury == 'Sciatic Problem' or injury == 'Tibia and Fibula Fracture' or injury == 'Fracture of the lower leg' or injury == 'Tibia Fracture' or injury == 'Inflamed head of fibula'):
        x4year.at[i, 'injurytype_leg'] += 1
      elif (injury == 'Muscular problems' or injury == 'Muscle Injury' or injury == 'Torn Muscle Fibre' or injury == 'Muscle Fatigue' or injury == 'Torn muscle bundle' or injury == 'Torn Muscle' or injury == 'Muscle fiber tear'):
        x4year.at[i, 'injurytype_muscular'] += 1
      elif (injury == 'Pubitis' or injury == 'Bruised pelvis' or injury == 'Pelvis Injury' or injury == 'Pubis bone irritation' or injury == 'bruised pelvis' or injury == 'bruised pelvis' or injury == 'Pelvic obliquity'):
        x4year.at[i, 'injurytype_pelvis'] += 1
      elif (injury == 'Rest'):
        x4year.at[i, 'injurytype_rest'] += 1
      elif (injury == 'Fractured Rib'):
        x4year.at[i, 'injurytype_rib'] += 1
      elif (injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar vertebrae problems' or injury == 'Blockade in the spinal' or injury == 'Cervical spine injury' or injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar Vertebra Fracture' or injury == 'Vertebra injury'):
        x4year.at[i, 'injurytype_spine'] += 1
      elif (injury == 'Thigh Muscle Strain' or injury == 'Thigh Problems' or injury == 'Thigh muscle rupture'):
        x4year.at[i, 'injurytype_thigh'] += 1
      elif (injury == 'Toe Injury' or injury == 'Fractured Toe'):
        x4year.at[i, 'injurytype_toe'] += 1
      elif (injury == 'Unknown Injury'):
        x4year.at[i, 'injurytype_unknown'] += 1
      elif (injury == 'Surgery' or injury == 'Contracture' or injury == 'Sprain' or injury == 'Laceration' or injury == 'Fracture' or injury == 'Ligament Injury' or injury == 'Stress response of the bone' or injury == 'Strain' or injury == 'Contused laceration' or injury == 'Muscle partial avulsion' or injury == 'torn tendon' or injury == 'Tear in a joint capsule' or injury == 'tendon irritation' or injury == 'Ligament Problems' or injury == 'Sideband injury' or injury == 'Bone buckling' or injury == 'Stretched Ligament' or injury == 'Pinched nerve' or injury == 'Overstretching' or injury == 'Tendonitis' or injury == 'laceration' or injury == 'Arthroscopie' or injury == 'Herniated Disc' or injury == 'Fatigue fracture' or injury == 'Torn Ligament' or injury == 'Cartilage Damage' or injury == 'Bursitis' or injury == 'Circulation Problems' or injury == 'Tendon crack' or injury == 'Cut' or injury == 'sprain' or injury == 'Burns' or injury == 'open wound' or injury == 'Sideband tear' or injury == 'Marrow bulge' or injury == 'Compartment syndrome' or injury == 'Vestibular disorder' or injury == 'Intraarticular ligament fissure' or injury == 'Ruptured ligaments' or injury == 'Muscular hairline crack' or injury == 'capsular tear' or injury == 'Traffic Accident' or injury == 'Flesh Wound'):
        x4year.at[i, 'injurytype_other'] += 1
      else:
        print(injury)

for i, row in x5year.iterrows(): #len(x1year)
  s = row['injuryType']
  if (s == 'nan'):
    x5year.at[i, 'injurytype_nan'] += 1
  elif type(s) == str:
    injury_list = s.split(" ,")
    injury_list = injury_list[:-1]
    for injury in injury_list:
      injury = injury.strip()
      if (injury == 'Abdominal Strain' or injury == 'Abdominal muscles injury' or injury == 'Appendectomy' or injury == 'Umbilical hernia'):
        x5year.at[i, 'injurytype_abdomen'] += 1
      elif (injury == 'Tear in the abductor muscle'):
        x5year.at[i, 'injurytype_abductor'] += 1
      elif (injury == 'Achilles tendon rupture' or injury == 'Achilles tendon problems' or injury == 'Achilles Irritation' or injury == 'Achilles tendon surgery'):
        x5year.at[i, 'injurytype_achilles'] += 1
      elif (injury == 'Adductor problems'):
        x5year.at[i, 'injurytype_adductor'] += 1
      elif (injury == 'Ankle Injury' or injury == 'Sprained ankle' or injury == 'Ankle problems' or injury == 'Torn ankle ligament' or injury == 'Ankle Surgery' or injury == 'Bruised Ankle' or injury == 'Distortion of the ankle' or injury == 'Ruptured syndesmotic ligament' or injury == 'Torn Ankle Ligament' or injury == 'Fracture-dislocation of the ankle' or injury == 'Sprained Ankle' or injury == 'Injury to the ankle' or injury == 'Ankle fracture' or injury == 'Ankle Inflammation' or injury == 'Syndesmotic ligament tear' or injury == 'Ankle Fracture' or injury == 'Capsular rupture in the ankle' or injury == 'Ruptured ankle ligament' or injury == 'Ruptured ankle aigament' or injury == 'Bruise on ankle' or injury == 'Ruptured intraarticular ligament initiation in the ankle' or injury == 'Peroneus tendon injury'):
        x5year.at[i, 'injurytype_ankle'] += 1
      elif (injury == 'Arm Injury' or injury == 'Shoulder Injury' or injury == 'Shoulder fracture' or injury == 'Elbow Injury' or injury == 'Bruised Acromioclavicular' or injury == 'Fractured Arm' or injury == 'Acromioclavicular Separation'):
        x5year.at[i, 'injurytype_arm'] += 1
      elif (injury == 'Back Injury' or injury == 'Back trouble' or injury == 'Back bruise' or injury == 'Lumbago'):
        x5year.at[i, 'injurytype_back'] += 1
      elif (injury == 'Bruise' or injury == 'Bruised Rib' or injury == 'Metatarsal bone bruise' or injury == 'Bone Bruise' or injury == 'Muscle bruise'):
        x5year.at[i, 'injurytype_bruise'] += 1
      elif (injury == 'Calf Problems' or injury == 'Calf Injury' or injury == 'Calf Strain' or injury == 'Disrupted Calf Muscle' or injury == 'Hairline crack in calfbone' or injury == 'Calf muscle strain'):
        x5year.at[i, 'injurytype_calf'] += 1
      elif (injury == 'Angina' or injury == 'Chest injury' or injury == 'Pneumothorax' or injury == 'Lung contusion'):
        x5year.at[i, 'injurytype_chest'] += 1
      elif (injury == 'Collarbone fracture'):
        x5year.at[i, 'injurytype_collarbone'] += 1
      elif (injury == 'Concussion'):
        x5year.at[i, 'injurytype_concussion'] += 1
      elif (injury == 'Cruciate Ligament Rupture' or injury == 'Cruciate Ligament Surgery' or injury == 'Cruciate Ligament Strain' or injury == 'Cruciate Ligament Injury' or injury == 'Ruptured cruciate ligament' or injury == 'Partial damage to the cruciate ligament' or injury == 'Cruciate ligament stretch'):
        x5year.at[i, 'injurytype_cruciateligament'] += 1
      elif (injury == 'Facial Injury' or injury == 'Nose surgery' or injury == 'Dental Surgery' or injury == 'Nasal Bone Fracture' or injury == 'Facial Fracture' or injury == 'Cheekbone Fracture' or injury == 'Eye Injury' or injury == 'Toothache' or injury == 'Frontal bone fracture' or injury == 'Fractured Jaw' or injury == 'Fractured Skull' or injury == 'Fracture of the orbit' or injury == 'Tooth Inflammation' or injury == 'Nose Injury'):
        x5year.at[i, 'injurytype_face'] += 1
      elif (injury == 'Finger Injury'):
        x5year.at[i, 'injurytype_finger'] += 1
      elif (injury == 'Fitness'):
        x5year.at[i, 'injurytype_fitness'] += 1
      elif (injury == 'Foot Injury' or injury == 'Bruised Foot' or injury == 'Fractured Foot' or injury == 'Arch pain' or injury == 'Heel Injury' or injury == 'Hell pain' or injury == 'Heelspur' or injury == 'Heel Bone Injury' or injury == 'Metatarsal Fracture' or injury == 'Heel pain' or injury == 'Plantar fascia' or injury == 'Foot surgery' or injury == 'Hairline crack in the foot' or injury == 'Partial demolition of the plantar fascia'):
        x5year.at[i, 'injurytype_foot'] += 1
      elif (injury == 'groin strain' or injury == 'Groin Strain' or injury == 'Groin Injury' or injury == 'Groin Surgery' or injury == 'Inguinal Hernia' or injury == 'Pubalgia' or injury == 'Testicular disruption'):
        x5year.at[i, 'injurytype_groin'] += 1
      elif (injury == 'Hamstring Injury' or injury == 'Pulled hamstring at the adductors' or injury == 'Hamstring contusion'):
        x5year.at[i, 'injurytype_hamstring'] += 1
      elif (injury == 'Hand Injury' or injury == 'Fractured Hand' or injury == 'Broken wrist' or injury == 'Wirst Injury' or injury == 'scaphoid operation' or injury == 'Hand fracture' or injury == 'Fractured Finger' or injury == 'Thumb Injury'):
        x5year.at[i, 'injurytype_hand'] += 1
      elif (injury == 'Head Injury' or injury == 'Neck Injury' or injury == 'Neck bruise'):
        x5year.at[i, 'injurytype_head'] += 1
      elif (injury == 'Heart Condition'):
        x5year.at[i, 'injurytype_heart'] += 1
      elif (injury == 'Hip problems' or injury == 'Hip Injury' or injury == 'Problems with the hip flexor' or injury == 'Bruised Hip' or injury == 'Pubis bone contusion' or injury == 'Problems with the right hip flexor'):
        x5year.at[i, 'injurytype_hip'] += 1
      elif (injury == 'Cold' or injury == 'Influenza' or injury == 'Flu' or injury == 'Abdominal Influenza' or injury == 'Ill' or injury == 'Corona virus' or injury == 'Infection' or injury == 'Virus Infection' or injury == 'Quarantine' or injury == 'Fever' or injury == 'Tonsillitis' or injury == 'Testicular Cancer' or injury == 'Pneumonia' or injury == 'Gastric problems' or injury == 'Mononucleosis' or injury == 'Chickenpox' or injury == 'Bronchitis' or injury == 'Stomach complaints' or injury == 'Food Poisoning' or injury == 'intestial virus' or injury == 'Infected wound' or injury == 'Malaria' or injury == 'cancer' or injury == 'Lymphoma' or injury == 'Depression' or injury == 'Cals Sclerosis' or injury == 'Kidney problems'):
        x5year.at[i, 'injurytype_ill'] += 1
      elif (injury == 'Inflammation'):
        x5year.at[i, 'injurytype_inflammation'] += 1
      elif (injury == 'Knee Injury' or injury == 'Knee Problems' or injury == 'Knee inflammation' or injury == 'Bruised Knee' or injury == 'Ruptured intraarticular ligament initiation in knee' or injury == 'Knee Surgery' or injury == 'Medial Collateral Ligament Tear' or injury == 'Medial Collateral Ligament Injury' or injury == 'Meniscal Injury' or injury == 'Sideband strain in the knee' or injury == 'Patella tendon irritation' or injury == 'Mensical Laceration' or injury == 'Twisted knee' or injury == 'Torn Knee Ligament' or injury == 'Medial Collateral Ligament Knee Injury' or injury == 'Patella problems' or injury == 'Ruptured lateral collateral ligament' or injury == 'Edema in the knee' or injury == 'Fractured Kneecap' or injury == 'Torn lateral collateral ligament' or injury == 'Torn Meniscus' or injury == 'Torn Collateral Ligament' or injury == 'Patella rupture' or injury == 'Torn knee ligament' or injury == 'Patella tendon luxation' or injury == 'Ruptured knee ligament' or injury == 'Rupture of Outer Meniscus' or injury == 'Double Torn Ligament' or injury == 'Meniscus Damage' or injury == 'Medial Collateral Ligament avulsion' or injury == 'Meniscus irritation' or injury == 'Inflamed ligaments of the knee' or injury == 'Rupture of the pattella'):
        x5year.at[i, 'injurytype_knee'] += 1
      elif (injury == 'Knock' or injury == 'Minor Knock' or injury == 'Dead Leg' or injury == 'Stiffness'):
        x5year.at[i, 'injurytype_knock'] += 1
      elif (injury == 'Leg Injury' or injury == 'Strain in the thigh and gluteal muscles' or injury == 'Shinbone injury' or injury == 'Shin bone bruise' or injury == 'Biceps femoris muscle injury' or injury == 'Fibula Fracture' or injury == 'Fractured Leg' or injury == 'Fissure of the fibula' or injury == 'Sciatic Problem' or injury == 'Tibia and Fibula Fracture' or injury == 'Fracture of the lower leg' or injury == 'Tibia Fracture' or injury == 'Inflamed head of fibula'):
        x5year.at[i, 'injurytype_leg'] += 1
      elif (injury == 'Muscular problems' or injury == 'Muscle Injury' or injury == 'Torn Muscle Fibre' or injury == 'Muscle Fatigue' or injury == 'Torn muscle bundle' or injury == 'Torn Muscle' or injury == 'Muscle fiber tear'):
        x5year.at[i, 'injurytype_muscular'] += 1
      elif (injury == 'Pubitis' or injury == 'Bruised pelvis' or injury == 'Pelvis Injury' or injury == 'Pubis bone irritation' or injury == 'bruised pelvis' or injury == 'bruised pelvis' or injury == 'Pelvic obliquity'):
        x5year.at[i, 'injurytype_pelvis'] += 1
      elif (injury == 'Rest'):
        x5year.at[i, 'injurytype_rest'] += 1
      elif (injury == 'Fractured Rib'):
        x5year.at[i, 'injurytype_rib'] += 1
      elif (injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar vertebrae problems' or injury == 'Blockade in the spinal' or injury == 'Cervical spine injury' or injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar Vertebra Fracture' or injury == 'Vertebra injury'):
        x5year.at[i, 'injurytype_spine'] += 1
      elif (injury == 'Thigh Muscle Strain' or injury == 'Thigh Problems' or injury == 'Thigh muscle rupture'):
        x5year.at[i, 'injurytype_thigh'] += 1
      elif (injury == 'Toe Injury' or injury == 'Fractured Toe'):
        x5year.at[i, 'injurytype_toe'] += 1
      elif (injury == 'Unknown Injury'):
        x5year.at[i, 'injurytype_unknown'] += 1
      elif (injury == 'Surgery' or injury == 'Contracture' or injury == 'Sprain' or injury == 'Laceration' or injury == 'Fracture' or injury == 'Ligament Injury' or injury == 'Stress response of the bone' or injury == 'Strain' or injury == 'Contused laceration' or injury == 'Muscle partial avulsion' or injury == 'torn tendon' or injury == 'Tear in a joint capsule' or injury == 'tendon irritation' or injury == 'Ligament Problems' or injury == 'Sideband injury' or injury == 'Bone buckling' or injury == 'Stretched Ligament' or injury == 'Pinched nerve' or injury == 'Overstretching' or injury == 'Tendonitis' or injury == 'laceration' or injury == 'Arthroscopie' or injury == 'Herniated Disc' or injury == 'Fatigue fracture' or injury == 'Torn Ligament' or injury == 'Cartilage Damage' or injury == 'Bursitis' or injury == 'Circulation Problems' or injury == 'Tendon crack' or injury == 'Cut' or injury == 'sprain' or injury == 'Burns' or injury == 'open wound' or injury == 'Sideband tear' or injury == 'Marrow bulge' or injury == 'Compartment syndrome' or injury == 'Vestibular disorder' or injury == 'Intraarticular ligament fissure' or injury == 'Ruptured ligaments' or injury == 'Muscular hairline crack' or injury == 'capsular tear' or injury == 'Traffic Accident' or injury == 'Flesh Wound'):
        x5year.at[i, 'injurytype_other'] += 1
      else:
        print(injury)

x2year.to_csv('/content/drive/MyDrive/PlayerValue/InjuryYearTwoSeparateInjuries.csv')
x3year.to_csv('/content/drive/MyDrive/PlayerValue/InjuryYearThreeSeparateInjuries.csv')
x4year.to_csv('/content/drive/MyDrive/PlayerValue/InjuryYearFourSeparateInjuries.csv')
x5year.to_csv('/content/drive/MyDrive/PlayerValue/InjuryYearFiveSeparateInjuries.csv')

x1year.to_csv('/content/drive/MyDrive/PlayerValue/InjuryYearOneSeparateInjuries.csv')

# @title
InjuryYearDurationUpdatedYearsdf = InjuryYearDurationUpdatedYearsdf.drop_duplicates(subset=['Player_url'], keep = 'first')
InjuryYearDurationUpdatedYearsdf.count()
InjuryYearDurationUpdatedYearsdf.to_csv('/content/drive/MyDrive/PlayerValue/InjuryYearDurationUpdatedYearsdf.csv')

# @title
#Clean dataframe so that each row only has one injury from that season year
for i in range (0, len(InjuryModeldf)):
  Year = InjuryModeldf.loc[i, 'Season_End_Year']
  if (Year == 2018):
    InjuryModeldf.at[i, 'Duration_2019'] = np.nan
    InjuryModeldf.at[i, 'Type_2019'] = np.nan
    InjuryModeldf.at[i, 'Duration_2020'] = np.nan
    InjuryModeldf.at[i, 'Type_2020'] = np.nan
    InjuryModeldf.at[i, 'Duration_2021'] = np.nan
    InjuryModeldf.at[i, 'Type_2021'] = np.nan
    InjuryModeldf.at[i, 'Duration_2022'] = np.nan
    InjuryModeldf.at[i, 'Type_2022'] = np.nan
    InjuryModeldf.at[i, 'Duration_2023'] = np.nan
    InjuryModeldf.at[i, 'Type_2023'] = np.nan
  elif (Year == 2019):
    InjuryModeldf.at[i, 'Duration_2018'] = np.nan
    InjuryModeldf.at[i, 'Type_2018'] = np.nan
    InjuryModeldf.at[i, 'Duration_2020'] = np.nan
    InjuryModeldf.at[i, 'Type_2020'] = np.nan
    InjuryModeldf.at[i, 'Duration_2021'] = np.nan
    InjuryModeldf.at[i, 'Type_2021'] = np.nan
    InjuryModeldf.at[i, 'Duration_2022'] = np.nan
    InjuryModeldf.at[i, 'Type_2022'] = np.nan
    InjuryModeldf.at[i, 'Duration_2023'] = np.nan
    InjuryModeldf.at[i, 'Type_2023'] = np.nan
  elif (Year == 2020):
    InjuryModeldf.at[i, 'Duration_2018'] = np.nan
    InjuryModeldf.at[i, 'Type_2018'] = np.nan
    InjuryModeldf.at[i, 'Duration_2019'] = np.nan
    InjuryModeldf.at[i, 'Type_2019'] = np.nan
    InjuryModeldf.at[i, 'Duration_2021'] = np.nan
    InjuryModeldf.at[i, 'Type_2021'] = np.nan
    InjuryModeldf.at[i, 'Duration_2022'] = np.nan
    InjuryModeldf.at[i, 'Type_2022'] = np.nan
    InjuryModeldf.at[i, 'Duration_2023'] = np.nan
    InjuryModeldf.at[i, 'Type_2023'] = np.nan
  elif (Year == 2021):
    InjuryModeldf.at[i, 'Duration_2018'] = np.nan
    InjuryModeldf.at[i, 'Type_2018'] = np.nan
    InjuryModeldf.at[i, 'Duration_2019'] = np.nan
    InjuryModeldf.at[i, 'Type_2019'] = np.nan
    InjuryModeldf.at[i, 'Duration_2020'] = np.nan
    InjuryModeldf.at[i, 'Type_2020'] = np.nan
    InjuryModeldf.at[i, 'Duration_2022'] = np.nan
    InjuryModeldf.at[i, 'Type_2022'] = np.nan
    InjuryModeldf.at[i, 'Duration_2023'] = np.nan
    InjuryModeldf.at[i, 'Type_2023'] = np.nan
  elif (Year == 2022):
    InjuryModeldf.at[i, 'Duration_2018'] = np.nan
    InjuryModeldf.at[i, 'Type_2018'] = np.nan
    InjuryModeldf.at[i, 'Duration_2019'] = np.nan
    InjuryModeldf.at[i, 'Type_2019'] = np.nan
    InjuryModeldf.at[i, 'Duration_2020'] = np.nan
    InjuryModeldf.at[i, 'Type_2020'] = np.nan
    InjuryModeldf.at[i, 'Duration_2021'] = np.nan
    InjuryModeldf.at[i, 'Type_2021'] = np.nan
    InjuryModeldf.at[i, 'Duration_2023'] = np.nan
    InjuryModeldf.at[i, 'Type_2023'] = np.nan
  elif (Year == 2023):
    InjuryModeldf.at[i, 'Duration_2018'] = np.nan
    InjuryModeldf.at[i, 'Type_2018'] = np.nan
    InjuryModeldf.at[i, 'Duration_2019'] = np.nan
    InjuryModeldf.at[i, 'Type_2019'] = np.nan
    InjuryModeldf.at[i, 'Duration_2020'] = np.nan
    InjuryModeldf.at[i, 'Type_2020'] = np.nan
    InjuryModeldf.at[i, 'Duration_2021'] = np.nan
    InjuryModeldf.at[i, 'Type_2021'] = np.nan
    InjuryModeldf.at[i, 'Duration_2022'] = np.nan
    InjuryModeldf.at[i, 'Type_2022'] = np.nan

#InjuryModeldf.to_csv('/content/drive/MyDrive/PlayerValue/FrebInjuryPerYear.csv')

InjuryModeldf = pd.read_csv('/content/drive/MyDrive/PlayerValue/FrebInjuryPerYear.csv')
InjuryModeldf.drop('Unnamed: 0', axis=1, inplace=True)

Testdf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Testdf.csv')
Testdf.drop('Unnamed: 0', axis=1, inplace=True)

Testdf.to_csv('/content/drive/MyDrive/PlayerValue/Testdfmodified.csv')

Testdf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Testdfmodified.csv')
Testdf.drop('Unnamed: 0', axis=1, inplace=True)

# Debug
Testdf = InjuryModeldf.copy()
for i in range (0, len(InjuryModeldf)):
  print(i)
  if (not (InjuryModeldf.loc[i, 'Player_x'] == 'John Stones' or InjuryModeldf.loc[i, 'Player_x'] == 'Danny Welbeck')):
    Testdf.drop(i, inplace = True)

#make sure to include type
#parameter years needed, concatenate dataframe and add injury duration + ground truth
def slidingWindowstest(years):
  InjuryDurationdf = pd.DataFrame()
  #truth dataframe for only duration of injuries across years
  InjuryDurationdf['duration_truth'] = np.nan
  #dataframe for stats across all years
  Statsdf = Testdf.copy()

  for year in range(1, years):
    for col in Testdf.columns:
      #duplicates columns for years-1
      Statsdf[str(col) + 'year' + str(year+1)] = np.nan

  Statsdf['duration'] = 0
  Statsdf['injuryType'] = ''



#concatenating dataframe
  for originalRow in range (0, len(Statsdf)):
    print('OriginalRow: ' + str(originalRow))
    url = Statsdf.loc[originalRow, 'Url']
    originalSeason = Statsdf.loc[originalRow, 'Season_End_Year'] #2018
    truthSeason = originalSeason + years #2019
    foundtruth = False
    #duration not working properly for year one, truth duration also doesn't work
    if (not str(Statsdf.loc[originalRow, 'Type_' + str(originalSeason)]) == 'nan'):
      Statsdf.loc[originalRow, 'injuryType'] += str(Statsdf.loc[originalRow, 'Type_' + str(originalSeason)])
    if (not str(Statsdf.loc[originalRow, 'Duration_' + str(originalSeason)]) == 'nan'):
      Statsdf.loc[originalRow, 'duration'] += Statsdf.loc[originalRow, 'Duration_' + str(originalSeason)]

    if (truthSeason > 2023):
      break
    for changedRow in range (originalRow, len(Statsdf)):
      currentSeason = Statsdf.loc[changedRow, 'Season_End_Year'] #2019
      if (currentSeason > truthSeason):
        break
      #if url matches
      if (Statsdf.loc[changedRow, 'Url'] == url and currentSeason > originalSeason):
        #year is not truth
        if (currentSeason < truthSeason):
          #iterate through all columns to concatenate to original row
          for col in range(0, len(Testdf.columns)):
            Statsdf.iloc[originalRow, (col + (currentSeason - originalSeason) * len(Testdf.columns))] = Statsdf.iloc[changedRow, col]
          if ((not str(Statsdf.loc[changedRow, 'Type_' + str(currentSeason)]) == 'nan') and years != 1):
            Statsdf.loc[originalRow, 'injuryType'] += str(Statsdf.loc[changedRow, 'Type_' + str(currentSeason)]) #should it be currentSeason?
          if ((not str(Statsdf.loc[changedRow, 'Duration_' + str(currentSeason)]) == 'nan') and years != 1):
            Statsdf.loc[originalRow, 'duration'] += Statsdf.loc[changedRow, 'Duration_' + str(currentSeason)]

        #if year is truth, add duration of injury
        elif (currentSeason == truthSeason):
          foundtruth = True
          if (str(Statsdf.loc[changedRow, 'Duration_' + str(truthSeason)]) == 'nan'):
            InjuryDurationdf.loc[len(InjuryDurationdf), 'duration_truth'] = 0
          else:
            InjuryDurationdf.loc[len(InjuryDurationdf), 'duration_truth'] = Statsdf.loc[changedRow, 'Duration_' + str(truthSeason)]
          break;

    if (foundtruth == False):
      InjuryDurationdf.loc[len(InjuryDurationdf), 'duration_truth'] = np.nan

#error may be how I am joining df

  Statsdf = Statsdf.join(InjuryDurationdf)

  #return modified df
  return Statsdf

#make sure to include type
#parameter years needed, concatenate dataframe and add injury duration + ground truth
def slidingWindows(years):
  InjuryDurationdf = pd.DataFrame()
  #truth dataframe for only duration of injuries across years
  InjuryDurationdf['duration_truth'] = np.nan
  #dataframe for stats across all years
  Statsdf = InjuryModeldf.copy()

  for year in range(1, years):
    for col in InjuryModeldf.columns:
      #duplicates columns for years-1
      Statsdf[str(col) + 'year' + str(year+1)] = np.nan

  Statsdf['duration'] = 0
  Statsdf['injuryType'] = ''



#concatenating dataframe
  for originalRow in range (0, len(Statsdf)):
    print('OriginalRow: ' + str(originalRow))
    url = Statsdf.loc[originalRow, 'Url']
    originalSeason = Statsdf.loc[originalRow, 'Season_End_Year'] #2018
    truthSeason = originalSeason + years #2019
    foundtruth = False
    #duration not working properly for year one, truth duration also doesn't work
    if (not str(Statsdf.loc[originalRow, 'Type_' + str(originalSeason)]) == 'nan'):
      Statsdf.loc[originalRow, 'injuryType'] += str(Statsdf.loc[originalRow, 'Type_' + str(originalSeason)])
    if (not str(Statsdf.loc[originalRow, 'Duration_' + str(originalSeason)]) == 'nan'):
      Statsdf.loc[originalRow, 'duration'] += Statsdf.loc[originalRow, 'Duration_' + str(originalSeason)]

    if (truthSeason > 2023):
      break
    for changedRow in range (originalRow + 2800, len(Statsdf)):
      currentSeason = Statsdf.loc[changedRow, 'Season_End_Year'] #2019
      if (currentSeason > truthSeason):
        break
      #if url matches
      if (Statsdf.loc[changedRow, 'Url'] == url and currentSeason > originalSeason):
        #year is not truth
        if (currentSeason < truthSeason):
          #iterate through all columns to concatenate to original row
          for col in range(0, len(Testdf.columns)):
            Statsdf.iloc[originalRow, (col + (currentSeason - originalSeason) * len(Testdf.columns))] = Statsdf.iloc[changedRow, col]
          if ((not str(Statsdf.loc[changedRow, 'Type_' + str(currentSeason)]) == 'nan') and years != 1):
            Statsdf.loc[originalRow, 'injuryType'] += str(Statsdf.loc[changedRow, 'Type_' + str(currentSeason)]) #should it be currentSeason?
          if ((not str(Statsdf.loc[changedRow, 'Duration_' + str(currentSeason)]) == 'nan') and years != 1):
            Statsdf.loc[originalRow, 'duration'] += Statsdf.loc[changedRow, 'Duration_' + str(currentSeason)]

        #if year is truth, add duration of injury
        elif (currentSeason == truthSeason):
          foundtruth = True
          if (str(Statsdf.loc[changedRow, 'Duration_' + str(truthSeason)]) == 'nan'):
            InjuryDurationdf.loc[len(InjuryDurationdf), 'duration_truth'] = 0
          else:
            InjuryDurationdf.loc[len(InjuryDurationdf), 'duration_truth'] = Statsdf.loc[changedRow, 'Duration_' + str(truthSeason)]
          break;

    if (foundtruth == False):
      InjuryDurationdf.loc[len(InjuryDurationdf), 'duration_truth'] = np.nan

#error may be how I am joining df

  Statsdf = Statsdf.join(InjuryDurationdf)

  #return modified df
  return Statsdf

InjuryModel5year = slidingWindows(5)

InjuryModel5year.to_csv('/content/drive/MyDrive/PlayerValue/InjuryModel5year.csv')

# @title
types = x1year.dtypes

# @title
for i in range(0, len(types)):
  print(types[i])

# @title
counter = 0
for col in x1year.columns:
  column_name = str(col)
  if (x1year.column_name.dtype != "object"):
    x1year.drop(counter, axis = 1)
    counter = counter + 1

#One year model
x1yearEncoding = pd.DataFrame()
x1year["Squad"] = x1year["Squad"].astype("category")
extracted = x1year["Squad"]
x1year.drop("Squad", axis = 1)
x1yearEncoding = pd.concat([x1yearEncoding, extracted], axis=1)
x1year["Comp"] = x1year["Comp"].astype("category")
extracted = x1year["Comp"]
x1year.drop("Comp", axis = 1)
x1yearEncoding = pd.concat([x1yearEncoding, extracted], axis=1)
x1year["Pos"] = x1year["Pos"].astype("category")
extracted = x1year["Pos"]
x1year.drop("Pos", axis = 1)
x1yearEncoding = pd.concat([x1yearEncoding, extracted], axis=1)
x1year["Nation"] = x1year["Nation"].astype("category")
extracted = x1year["Nation"]
x1year.drop("Nation", axis = 1)
x1yearEncoding = pd.concat([x1yearEncoding, extracted], axis=1)
x1year["player_foot"] = x1year["player_foot"].astype("category")
extracted = x1year["player_foot"]
x1year.drop("player_foot", axis = 1)
x1yearEncoding = pd.concat([x1yearEncoding, extracted], axis=1)
x1year["country"] = x1year["country"].astype("category")
extracted = x1year["country"]
x1year.drop("country", axis = 1)
x1yearEncoding = pd.concat([x1yearEncoding, extracted], axis=1)
x1year.drop(['duration_truth'], axis = 1, inplace = True)

#One hot encoder also converting floats+duration to booleans

#check all hot encoded columns/features

# Check how one hot encoding is changing column names and values

# Manually clean injury types

#delete spaces/commas in column names

#born vs age

enc = OneHotEncoder()
enc.fit(x1yearEncoding)

#transform categorical features
X1year_encoded = enc.transform(x1yearEncoding).toarray()
feature_names = x1yearEncoding.columns
new_feature_names = enc.get_feature_names_out(feature_names)
Xencoded = pd.DataFrame(X1year_encoded, columns= new_feature_names)
X = x1year.join(Xencoded)

X = X[["Age", "Mins_Per_90_Playing", "injurytype_calf", "injurytype_hamstring", "injurytype_knee", "injurytype_ankle", "injurytype_ill", "duration"]]

# @title
for i in list(Xencoded.columns):
    print(i)

x1year_train, x1year_test, y1year_train, y1year_test = train_test_split(X, y1year, random_state=1, train_size=0.75)
InjuryModelXGB1year = xgb.XGBRegressor(tree_method="approx", max_depth = 3, n_estimators = 200, seed = 2, enable_categorical=True)
InjuryModelXGB1year.fit(x1year_train, y1year_train, verbose = True, eval_set = [(x1year_train, y1year_train), (x1year_test, y1year_test)])
# print("feature importances:", InjuryModelXGB1year.feature_importances_)

feat_imp_list = list(zip(list(InjuryModelXGB1year.feature_importances_) , x1year_train.columns.to_list()) )
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])
#feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[1], reverse=True) , columns = ['feature_value','feature_name'])
#print(feature_imp_df.loc[1, "Pos_DF"])
#print(feature_imp_df[feature_imp_df['feature_name'] == "injuryType_nan"]['feature_value'])

x1year_train["injurytype_knee"].hist(bins=10)
x1year_test["injurytype_knee"].hist(bins=10)

#unSub_Subs, err - search up
pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(100))

y1year_pred = InjuryModelXGB1year.predict(x1year_test)
print(r2_score(y1year_test, y1year_pred))
print(mean_absolute_error(y1year_test, y1year_pred))
print(mean_squared_error(y1year_test, y1year_pred))

# @title
#plotting tree
format = 'svg' #You should try the 'svg'
image = xgb.to_graphviz(InjuryModelXGB1year)

#Set a different dpi (work only if format == 'png')
image.graph_attr = {'dpi':'400'}

image.render('filename', format = format)

# @title
# TODO remove cell
InjuryModelXGB1year = xgb.train(params = params, dtrain = d_matrix_train)
explainer = shap.TreeExplainer(InjuryModelXGB1year)

# @title
x1year_train, x1year_test, y1year_train, y1year_test = train_test_split(X, y1year, random_state=1, train_size = 0.8)
d_matrix_train = xgb.DMatrix(x1year_train, y1year_train, enable_categorical=True)
d_matrix_test = xgb.DMatrix(x1year_test, y1year_test, enable_categorical=True)
params = {"objective": "reg:squarederror", "tree_method": "approx"}
InjuryModelXGB1year = xgb.train(params = params, dtrain = d_matrix_train)
explainer = shap.TreeExplainer(InjuryModelXGB1year)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, plot_type='bar')

# @title
#shap_values = InjuryModelXGB1year.predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
explainer = shap.TreeExplainer(InjuryModelXGB1year)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, plot_type='bar')

# @title
shap_values = InjuryModelXGB1year.predict(XGBRegressor(x1year_test, enable_categorical=True), pred_contribs=True)
explainer = shap.TreeExplainer(InjuryModelXGB1year)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, plot_type='bar')

# @title
x1year_train, x1year_test, y1year_train, y1year_test = train_test_split(X, y1year, random_state=1, train_size = 0.8)
InjuryModelXGB1year = XGBRegressor(tree_method="approx", seed=1, enable_categorical = True)
InjuryModelXGB1year.fit(x1year_train, y1year_train, verbose = True, early_stopping_rounds = 10, eval_set = [(x1year_test, y1year_test)])

InjuryModelXGB1year.save_model('InjuryModelXGB1year.json')
#files.download('InjuryModelXGB1year.json')

InjuryModelXGB1year.load_model('InjuryModelXGB1year.json')

#y1year_pred = InjuryModelXGB1year.predict(x1year_test)
shap_values = InjuryModelXGB1year.predict(XGBRegressor(x1year_test, enable_categorical=True), pred_contribs=True)
#print(r2_score(y1year_test, y1year_pred))
#print(mean_absolute_error(y1year_test, y1year_pred))
#print(mean_squared_error(y1year_test, y1year_pred))

explainer = shap.TreeExplainer(InjuryModelXGB1year)
shap_values = explainer(X)

shap.summary_plot(shap_values, X, plot_type='bar')

# waterfall plot for first observation
shap.plots.beeswarm(shap_values, max_display = 20)

print(shap_values[0])

print(shap_values.base_values.shape)

#One year model
x2yearEncoding = pd.DataFrame()
x2year["Squad"] = x2year["Squad"].astype("category")
extracted2year = x2year["Squad"]
x2year.drop("Squad", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["Comp"] = x2year["Comp"].astype("category")
extracted2year = x2year["Comp"]
x2year.drop("Comp", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["Pos"] = x2year["Pos"].astype("category")
extracted2year = x2year["Pos"]
x2year.drop("Pos", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["Nation"] = x2year["Nation"].astype("category")
extracted2year = x2year["Nation"]
x2year.drop("Nation", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["player_foot"] = x2year["player_foot"].astype("category")
extracted2year = x2year["player_foot"]
x2year.drop("player_foot", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["country"] = x2year["country"].astype("category")
extracted2year = x2year["country"]
x2year.drop("country", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["Squadyear2"] = x2year["Squadyear2"].astype("category")
extracted2year = x2year["Squadyear2"]
x2year.drop("Squadyear2", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["Compyear2"] = x2year["Compyear2"].astype("category")
extracted2year = x2year["Compyear2"]
x2year.drop("Compyear2", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["Posyear2"] = x2year["Posyear2"].astype("category")
extracted2year = x2year["Posyear2"]
x2year.drop("Posyear2", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["Nationyear2"] = x2year["Nationyear2"].astype("category")
extracted2year = x2year["Nationyear2"]
x2year.drop("Nationyear2", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["player_footyear2"] = x2year["player_footyear2"].astype("category")
extracted2year = x2year["player_footyear2"]
x2year.drop("player_footyear2", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year["countryyear2"] = x2year["countryyear2"].astype("category")
extracted2year = x2year["countryyear2"]
x2year.drop("countryyear2", axis = 1)
x2yearEncoding = pd.concat([x2yearEncoding, extracted2year], axis=1)
x2year.drop(['duration_truth'], axis = 1, inplace = True)

#One hot encoder also converting floats+duration to booleans

#check all hot encoded columns/features

# Check how one hot encoding is changing column names and values

# Manually clean injury types

#delete spaces/commas in column names

#born vs age

enc2year = OneHotEncoder()
enc2year.fit(x2yearEncoding)

#transform categorical features
X2year_encoded = enc2year.transform(x2yearEncoding).toarray()
feature_names2year = x2yearEncoding.columns
new_feature_names2year = enc2year.get_feature_names_out(feature_names2year)
X2encoded = pd.DataFrame(X2year_encoded, columns= new_feature_names2year)
x2yearFinal = x2year.join(X2encoded)

x2year_train, x2year_test, y2year_train, y2year_test = train_test_split(x2yearFinal, y2year, random_state=1, train_size=0.8)
InjuryModelXGB2year = xgb.XGBRegressor(tree_method="approx", seed = 1, enable_categorical=True)
InjuryModelXGB2year.fit(x2year_train, y2year_train, verbose = True, eval_set = [(x2year_test, y2year_test)])
# print("feature importances:", InjuryModelXGB1year.feature_importances_)

feat_imp_list = list(zip ( list(InjuryModelXGB2year.feature_importances_) , x2year_train.columns.to_list()) )
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])
#feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[1], reverse=True) , columns = ['feature_value','feature_name'])
#print(feature_imp_df.loc[1, "Pos_DF"])
#print(feature_imp_df[feature_imp_df['feature_name'] == "injuryType_nan"]['feature_value'])

#unSub_Subs, err - search up
pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(100))

y2year_pred = InjuryModelXGB2year.predict(x2year_test)
print(r2_score(y2year_test, y2year_pred))
print(mean_absolute_error(y2year_test, y2year_pred))
print(mean_squared_error(y2year_test, y2year_pred))

print(y2year_train.iloc[0])
print()

xTest_train = pd.DataFrame()
xTest_train = x1year.iloc[0:6000]
yTest_train = y1year.iloc[0:6000]
xTest_test = x1year.iloc[6000:7000]
yTest_test = y1year.iloc[6000:7000]

## Fewer features (8-12), lower height of tree - find hyperparameters that prevent overfitting that fit well (testing and training have similar rmse)

#xTest_train, yTest_train = train_test_split(x2yearFinal, y2year, random_state=1)
InjuryModelXGB2yearTest = xgb.XGBRegressor(tree_method="approx", seed = 1, enable_categorical=True)
InjuryModelXGB2yearTest.fit(xTest_train, yTest_train, verbose = True, eval_set = [(xTest_train, yTest_train), (xTest_test, yTest_test)])
# print("feature importances:", InjuryModelXGB1year.feature_importances_)

feat_imp_list = list(zip(list(InjuryModelXGB2yearTest.feature_importances_), xTest_train.columns.to_list()))
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])
#feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[1], reverse=True) , columns = ['feature_value','feature_name'])
#print(feature_imp_df.loc[1, "Pos_DF"])
#print(feature_imp_df[feature_imp_df['feature_name'] == "injuryType_nan"]['feature_value'])

#unSub_Subs, err - search up
pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(100))

print(yTest_train.iloc[0:2])

y1year_pred = InjuryModelXGB2yearTest.predict(xTest_train)
print(r2_score(yTest_train, y1year_pred))
print(mean_absolute_error(yTest_train, y1year_pred))
print(mean_squared_error(yTest_train, y1year_pred))

#3 year model
x3yearEncoding = pd.DataFrame()
x3year["Squad"] = x3year["Squad"].astype("category")
extracted3year = x3year["Squad"]
x3year.drop("Squad", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["Comp"] = x3year["Comp"].astype("category")
extracted3year = x3year["Comp"]
x3year.drop("Comp", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["Pos"] = x3year["Pos"].astype("category")
extracted3year = x3year["Pos"]
x3year.drop("Pos", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["Nation"] = x3year["Nation"].astype("category")
extracted3year = x3year["Nation"]
x3year.drop("Nation", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["player_foot"] = x3year["player_foot"].astype("category")
extracted3year = x3year["player_foot"]
x3year.drop("player_foot", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["country"] = x3year["country"].astype("category")
extracted3year = x3year["country"]
x3year.drop("country", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)

x3year["Squadyear2"] = x3year["Squadyear2"].astype("category")
extracted3year = x3year["Squadyear2"]
x3year.drop("Squadyear2", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["Compyear2"] = x3year["Compyear2"].astype("category")
extracted3year = x3year["Compyear2"]
x3year.drop("Compyear2", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["Posyear2"] = x3year["Posyear2"].astype("category")
extracted3year = x3year["Posyear2"]
x3year.drop("Posyear2", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["Nationyear2"] = x3year["Nationyear2"].astype("category")
extracted3year = x3year["Nationyear2"]
x3year.drop("Nationyear2", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["player_footyear2"] = x3year["player_footyear2"].astype("category")
extracted3year = x3year["player_footyear2"]
x3year.drop("player_footyear2", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["countryyear2"] = x3year["countryyear2"].astype("category")
extracted3year = x3year["countryyear2"]
x3year.drop("countryyear2", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)

x3year["Squadyear3"] = x3year["Squadyear3"].astype("category")
extracted3year = x3year["Squadyear3"]
x3year.drop("Squadyear3", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["Compyear3"] = x3year["Compyear3"].astype("category")
extracted3year = x3year["Compyear3"]
x3year.drop("Compyear3", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["Posyear3"] = x3year["Posyear3"].astype("category")
extracted3year = x3year["Posyear3"]
x3year.drop("Posyear3", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["Nationyear3"] = x3year["Nationyear3"].astype("category")
extracted3year = x3year["Nationyear3"]
x3year.drop("Nationyear3", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["player_footyear3"] = x3year["player_footyear3"].astype("category")
extracted3year = x3year["player_footyear3"]
x3year.drop("player_footyear3", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year["countryyear3"] = x3year["countryyear3"].astype("category")
extracted3year = x3year["countryyear3"]
x3year.drop("countryyear3", axis = 1)
x3yearEncoding = pd.concat([x3yearEncoding, extracted3year], axis=1)
x3year.drop(['duration_truth'], axis = 1, inplace = True)

#One hot encoder also converting floats+duration to booleans

#check all hot encoded columns/features

# Check how one hot encoding is changing column names and values

# Manually clean injury types

#delete spaces/commas in column names

#born vs age

enc3year = OneHotEncoder()
enc3year.fit(x3yearEncoding)

#transform categorical features
X3year_encoded = enc3year.transform(x3yearEncoding).toarray()
feature_names3year = x3yearEncoding.columns
new_feature_names3year = enc3year.get_feature_names_out(feature_names3year)
X3encoded = pd.DataFrame(X3year_encoded, columns= new_feature_names3year)
x3yearFinal = x3year.join(X3encoded)

x3year_train, x3year_test, y3year_train, y3year_test = train_test_split(x3yearFinal, y3year, random_state=1, train_size=0.8)
InjuryModelXGB3year = xgb.XGBRegressor(tree_method="approx", seed = 1, enable_categorical=True)
InjuryModelXGB3year.fit(x3year_train, y3year_train, verbose = True, eval_set = [(x3year_test, y3year_test)])
# print("feature importances:", InjuryModelXGB1year.feature_importances_)

feat_imp_list = list(zip ( list(InjuryModelXGB1year.feature_importances_) , x3year_train.columns.to_list()) )
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])
#feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[1], reverse=True) , columns = ['feature_value','feature_name'])
#print(feature_imp_df.loc[1, "Pos_DF"])
#print(feature_imp_df[feature_imp_df['feature_name'] == "injuryType_nan"]['feature_value'])

#unSub_Subs, err - search up
pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(100))

y3year_pred = InjuryModelXGB3year.predict(x3year_test)
print(r2_score(y3year_test, y3year_pred))
print(mean_absolute_error(y3year_test, y3year_pred))
print(mean_squared_error(y3year_test, y3year_pred))

#4 year model
x4yearEncoding = pd.DataFrame()
x4year["Squad"] = x4year["Squad"].astype("category")
extracted4year = x4year["Squad"]
x4year.drop("Squad", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Comp"] = x4year["Comp"].astype("category")
extracted4year = x4year["Comp"]
x4year.drop("Comp", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Pos"] = x4year["Pos"].astype("category")
extracted4year = x4year["Pos"]
x4year.drop("Pos", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Nation"] = x4year["Nation"].astype("category")
extracted4year = x4year["Nation"]
x4year.drop("Nation", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["player_foot"] = x4year["player_foot"].astype("category")
extracted4year = x4year["player_foot"]
x4year.drop("player_foot", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["country"] = x4year["country"].astype("category")
extracted4year = x4year["country"]
x4year.drop("country", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)

x4year["Squadyear2"] = x4year["Squadyear2"].astype("category")
extracted4year = x4year["Squadyear2"]
x4year.drop("Squadyear2", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Compyear2"] = x4year["Compyear2"].astype("category")
extracted4year = x4year["Compyear2"]
x4year.drop("Compyear2", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Posyear2"] = x4year["Posyear2"].astype("category")
extracted4year = x4year["Posyear2"]
x4year.drop("Posyear2", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Nationyear2"] = x4year["Nationyear2"].astype("category")
extracted4year = x4year["Nationyear2"]
x4year.drop("Nationyear2", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["player_footyear2"] = x4year["player_footyear2"].astype("category")
extracted4year = x4year["player_footyear2"]
x4year.drop("player_footyear2", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["countryyear2"] = x4year["countryyear2"].astype("category")
extracted4year = x4year["countryyear2"]
x4year.drop("countryyear2", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)

x4year["Squadyear3"] = x4year["Squadyear3"].astype("category")
extracted4year = x3year["Squadyear3"]
x4year.drop("Squadyear3", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Compyear3"] = x4year["Compyear3"].astype("category")
extracted4year = x4year["Compyear3"]
x4year.drop("Compyear3", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Posyear3"] = x4year["Posyear3"].astype("category")
extracted4year = x4year["Posyear3"]
x4year.drop("Posyear3", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Nationyear3"] = x4year["Nationyear3"].astype("category")
extracted4year = x4year["Nationyear3"]
x4year.drop("Nationyear3", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["player_footyear3"] = x4year["player_footyear3"].astype("category")
extracted4year = x4year["player_footyear3"]
x4year.drop("player_footyear3", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["countryyear3"] = x4year["countryyear3"].astype("category")
extracted4year = x4year["countryyear3"]
x4year.drop("countryyear3", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)

x4year["Squadyear4"] = x4year["Squadyear4"].astype("category")
extracted4year = x4year["Squadyear4"]
x4year.drop("Squadyear4", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Compyear4"] = x4year["Compyear4"].astype("category")
extracted4year = x4year["Compyear4"]
x4year.drop("Compyear4", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Posyear4"] = x4year["Posyear4"].astype("category")
extracted4year = x4year["Posyear4"]
x4year.drop("Posyear4", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["Nationyear4"] = x4year["Nationyear4"].astype("category")
extracted4year = x4year["Nationyear4"]
x4year.drop("Nationyear4", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["player_footyear4"] = x4year["player_footyear4"].astype("category")
extracted4year = x4year["player_footyear4"]
x4year.drop("player_footyear4", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year["countryyear4"] = x4year["countryyear4"].astype("category")
extracted4year = x4year["countryyear4"]
x4year.drop("countryyear4", axis = 1)
x4yearEncoding = pd.concat([x4yearEncoding, extracted4year], axis=1)
x4year.drop(['duration_truth'], axis = 1, inplace = True)

#One hot encoder also converting floats+duration to booleans

#check all hot encoded columns/features

# Check how one hot encoding is changing column names and values

# Manually clean injury types

#delete spaces/commas in column names

#born vs age

enc4year = OneHotEncoder()
enc4year.fit(x4yearEncoding)

#transform categorical features
X4year_encoded = enc4year.transform(x4yearEncoding).toarray()
feature_names4year = x4yearEncoding.columns
new_feature_names4year = enc4year.get_feature_names_out(feature_names4year)
X4encoded = pd.DataFrame(X4year_encoded, columns= new_feature_names4year)
x4yearFinal = x4year.join(X4encoded)

x4year_train, x4year_test, y4year_train, y4year_test = train_test_split(x4yearFinal, y4year, random_state=1, train_size=0.8)
InjuryModelXGB4year = xgb.XGBRegressor(tree_method="approx", seed = 1, enable_categorical=True)
InjuryModelXGB4year.fit(x4year_train, y4year_train, verbose = True, eval_set = [(x4year_test, y4year_test)])
# print("feature importances:", InjuryModelXGB1year.feature_importances_)

feat_imp_list = list(zip ( list(InjuryModelXGB4year.feature_importances_) , x4year_train.columns.to_list()) )
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])
#feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[1], reverse=True) , columns = ['feature_value','feature_name'])
#print(feature_imp_df.loc[1, "Pos_DF"])
#print(feature_imp_df[feature_imp_df['feature_name'] == "injuryType_nan"]['feature_value'])

#unSub_Subs, err - search up
pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(100))

y4year_pred = InjuryModelXGB4year.predict(x4year_test)
print(r2_score(y4year_test, y4year_pred))
print(mean_absolute_error(y4year_test, y4year_pred))
print(mean_squared_error(y4year_test, y4year_pred))

#5 year model
x5yearEncoding = pd.DataFrame()
x5year["Squad"] = x5year["Squad"].astype("category")
extracted5year = x5year["Squad"]
x5year.drop("Squad", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Comp"] = x5year["Comp"].astype("category")
extracted5year = x5year["Comp"]
x5year.drop("Comp", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Pos"] = x5year["Pos"].astype("category")
extracted5year = x5year["Pos"]
x5year.drop("Pos", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Nation"] = x5year["Nation"].astype("category")
extracted5year = x5year["Nation"]
x5year.drop("Nation", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["player_foot"] = x5year["player_foot"].astype("category")
extracted5year = x5year["player_foot"]
x5year.drop("player_foot", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["country"] = x5year["country"].astype("category")
extracted5year = x5year["country"]
x5year.drop("country", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)

x5year["Squadyear2"] = x5year["Squadyear2"].astype("category")
extracted5year = x5year["Squadyear2"]
x5year.drop("Squadyear2", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Compyear2"] = x5year["Compyear2"].astype("category")
extracted5year = x5year["Compyear2"]
x5year.drop("Compyear2", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Posyear2"] = x5year["Posyear2"].astype("category")
extracted5year = x5year["Posyear2"]
x5year.drop("Posyear2", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Nationyear2"] = x5year["Nationyear2"].astype("category")
extracted5year = x5year["Nationyear2"]
x5year.drop("Nationyear2", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["player_footyear2"] = x5year["player_footyear2"].astype("category")
extracted5year = x5year["player_footyear2"]
x5year.drop("player_footyear2", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["countryyear2"] = x5year["countryyear2"].astype("category")
extracted5year = x5year["countryyear2"]
x5year.drop("countryyear2", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)

x5year["Squadyear3"] = x5year["Squadyear3"].astype("category")
extracted5year = x3year["Squadyear3"]
x5year.drop("Squadyear3", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Compyear3"] = x5year["Compyear3"].astype("category")
extracted5year = x5year["Compyear3"]
x5year.drop("Compyear3", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Posyear3"] = x5year["Posyear3"].astype("category")
extracted5year = x5year["Posyear3"]
x5year.drop("Posyear3", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Nationyear3"] = x5year["Nationyear3"].astype("category")
extracted5year = x5year["Nationyear3"]
x5year.drop("Nationyear3", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["player_footyear3"] = x5year["player_footyear3"].astype("category")
extracted5year = x5year["player_footyear3"]
x5year.drop("player_footyear3", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["countryyear3"] = x5year["countryyear3"].astype("category")
extracted5year = x5year["countryyear3"]
x5year.drop("countryyear3", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)

x5year["Squadyear4"] = x5year["Squadyear4"].astype("category")
extracted5year = x5year["Squadyear4"]
x5year.drop("Squadyear4", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Compyear4"] = x5year["Compyear4"].astype("category")
extracted5year = x5year["Compyear4"]
x5year.drop("Compyear4", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Posyear4"] = x5year["Posyear4"].astype("category")
extracted5year = x5year["Posyear4"]
x5year.drop("Posyear4", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Nationyear4"] = x5year["Nationyear4"].astype("category")
extracted5year = x5year["Nationyear4"]
x5year.drop("Nationyear4", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["player_footyear4"] = x5year["player_footyear4"].astype("category")
extracted5year = x5year["player_footyear4"]
x5year.drop("player_footyear4", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["countryyear4"] = x5year["countryyear4"].astype("category")
extracted5year = x5year["countryyear4"]
x5year.drop("countryyear4", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)

x5year["Squadyear5"] = x5year["Squadyear5"].astype("category")
extracted5year = x5year["Squadyear5"]
x5year.drop("Squadyear5", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Compyear5"] = x5year["Compyear5"].astype("category")
extracted5year = x5year["Compyear5"]
x5year.drop("Compyear5", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Posyear5"] = x5year["Posyear5"].astype("category")
extracted5year = x5year["Posyear5"]
x5year.drop("Posyear5", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["Nationyear5"] = x5year["Nationyear5"].astype("category")
extracted5year = x5year["Nationyear5"]
x5year.drop("Nationyear5", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["player_footyear5"] = x5year["player_footyear5"].astype("category")
extracted5year = x5year["player_footyear5"]
x5year.drop("player_footyear5", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year["countryyear5"] = x5year["countryyear5"].astype("category")
extracted5year = x5year["countryyear5"]
x5year.drop("countryyear5", axis = 1)
x5yearEncoding = pd.concat([x5yearEncoding, extracted5year], axis=1)
x5year.drop(['duration_truth'], axis = 1, inplace = True)

#One hot encoder also converting floats+duration to booleans

#check all hot encoded columns/features

# Check how one hot encoding is changing column names and values

# Manually clean injury types

#delete spaces/commas in column names

#born vs age

enc5year = OneHotEncoder()
enc5year.fit(x5yearEncoding)

#transform categorical features
X5year_encoded = enc5year.transform(x5yearEncoding).toarray()
feature_names5year = x5yearEncoding.columns
new_feature_names5year = enc5year.get_feature_names_out(feature_names5year)
X5encoded = pd.DataFrame(X5year_encoded, columns= new_feature_names5year)
x5yearFinal = x5year.join(X5encoded)

x5yearFinal = x5yearFinal[["Age", "Mins_Per_90_Playing", "injurytype_calf", "injurytype_hamstring", "injurytype_knee", "injurytype_ankle", "injurytype_ill", "duration"]]

x5year_train, x5year_test, y5year_train, y5year_test = train_test_split(x5yearFinal, y5year, random_state=1, train_size=0.8)
InjuryModelXGB5year = xgb.XGBRegressor(tree_method="approx", max_depth = 3, n_estimators = 200, seed = 1, enable_categorical=True)
InjuryModelXGB5year.fit(x5year_train, y5year_train, verbose = True, eval_set = [(x5year_train, y5year_train), (x5year_test, y5year_test)])
# print("feature importances:", InjuryModelXGB1year.feature_importances_)

feat_imp_list = list(zip(list(InjuryModelXGB5year.feature_importances_) , x5year_train.columns.to_list()) )
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])
#feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[1], reverse=True) , columns = ['feature_value','feature_name'])
#print(feature_imp_df.loc[1, "Pos_DF"])
#print(feature_imp_df[feature_imp_df['feature_name'] == "injuryType_nan"]['feature_value'])

print(x5year_train.shape, y5year_train.shape, x5year_test.shape, y5year_test.shape)

#unSub_Subs, err - search up
pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(100))

x5year_train["duration"].hist(bins=10)
x5year_test["duration"].hist(bins=10)

y5year_pred = InjuryModelXGB5year.predict(x5year_test)
print(r2_score(y5year_test, y5year_pred))
print(mean_absolute_error(y5year_test, y5year_pred))
print(mean_squared_error(y5year_test, y5year_pred))

#Iterating thorugh hyperparameters of model
x1year_train, x1year_test, y1year_train, y1year_test = train_test_split(x1year, y1year, random_state=1, train_size = 0.8)
for depth in [None, 5, 10, 15, 20]:
  for learning_rate in [0.1, 0.01, 0.001]:
    for min_child_weight in [1,2,3]:
      for gamma in [0.1, 0.2, 0.3]:
        InjuryModelXGB1year = xgb.XGBRegressor(tree_method="approx", seed=1, max_depth = depth, learning_rate = learning_rate, min_child_weight = min_child_weight, gamma = gamma, enable_categorical = True)
        InjuryModelXGB1year.fit(x1year_train, y1year_train, verbose = True, eval_set = [(x1year_test, y1year_test)])
        y1year_pred = InjuryModelXGB1year.predict(x1year_test)
        print(r2_score(y1year_test, y1year_pred))
        InjuryModelXGB1year.save_model('xgb_injury_1year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y1year_test, y1year_pred)) + '.json')
        files.download('xgb_injury_1year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y1year_test, y1year_pred)) + '.json')

#Iterating thorugh hyperparameters of model
x2year_train, x2year_test, y2year_train, y2year_test = train_test_split(x2year, y2year, random_state=1, train_size = 0.8)
for depth in [None, 5, 10, 15, 20]:
  for learning_rate in [0.1, 0.01, 0.001]:
    for min_child_weight in [1,2,3]:
      for gamma in [0.1, 0.2, 0.3]:
        InjuryModelXGB2year = xgb.XGBRegressor(tree_method="approx", seed=1, max_depth = depth, learning_rate = learning_rate, min_child_weight = min_child_weight, gamma = gamma, enable_categorical = True)
        InjuryModelXGB2year.fit(x2year_train, y2year_train, verbose = True, eval_set = [(x2year_test, y2year_test)])
        y2year_pred = InjuryModelXGB2year.predict(x2year_test)
        print(r2_score(y2year_test, y2year_pred))
        InjuryModelXGB2year.save_model('xgb_injury_2year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y2year_test, y2year_pred)) + '.json')
        files.download('xgb_injury_2year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y2year_test, y2year_pred)) + '.json')

#Iterating thorugh hyperparameters of model
x3year_train, x3year_test, y3year_train, y3year_test = train_test_split(x3year, y3year, random_state=1, train_size = 0.8)
for depth in [None, 5, 10, 15, 20]:
  for learning_rate in [0.1, 0.01, 0.001]:
    for min_child_weight in [1,2,3]:
      for gamma in [0.1, 0.2, 0.3]:
        InjuryModelXGB3year = xgb.XGBRegressor(tree_method="approx", seed=1, max_depth = depth, learning_rate = learning_rate, min_child_weight = min_child_weight, gamma = gamma, enable_categorical = True)
        InjuryModelXGB3year.fit(x3year_train, y3year_train, verbose = True, eval_set = [(x3year_test, y3year_test)])
        y3year_pred = InjuryModelXGB3year.predict(x3year_test)
        print(r2_score(y3year_test, y3year_pred))
        InjuryModelXGB3year.save_model('xgb_injury_3year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y3year_test, y3year_pred)) + '.json')
        files.download('xgb_injury_3year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y3year_test, y3year_pred)) + '.json')

#Iterating thorugh hyperparameters of model
x4year_train, x4year_test, y4year_train, y4year_test = train_test_split(x4year, y4year, random_state=1, train_size = 0.8)
for depth in [None, 5, 10, 15, 20]:
  for learning_rate in [0.1, 0.01, 0.001]:
    for min_child_weight in [1,2,3]:
      for gamma in [0.1, 0.2, 0.3]:
        InjuryModelXGB4year = xgb.XGBRegressor(tree_method="approx", seed=1, max_depth = depth, learning_rate = learning_rate, min_child_weight = min_child_weight, gamma = gamma, enable_categorical = True)
        InjuryModelXGB4year.fit(x4year_train, y4year_train, verbose = True, eval_set = [(x4year_test, y4year_test)])
        y4year_pred = InjuryModelXGB4year.predict(x4year_test)
        print(r2_score(y4year_test, y4year_pred))
        InjuryModelXGB4year.save_model('xgb_injury_4year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y4year_test, y4year_pred)) + '.json')
        files.download('xgb_injury_4year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y4year_test, y4year_pred)) + '.json')

#Iterating thorugh hyperparameters of model
x5year_train, x5year_test, y5year_train, y5year_test = train_test_split(x5year, y5year, random_state=1, train_size = 0.8)
for depth in [None, 5, 10, 15, 20]:
  for learning_rate in [0.1, 0.01, 0.001]:
    for min_child_weight in [1,2,3]:
      for gamma in [0.1, 0.2, 0.3]:
        InjuryModelXGB5year = xgb.XGBRegressor(tree_method="approx", seed=1, max_depth = depth, learning_rate = learning_rate, min_child_weight = min_child_weight, gamma = gamma, enable_categorical = True)
        InjuryModelXGB5year.fit(x5year_train, y5year_train, verbose = True, eval_set = [(x5year_test, y5year_test)])
        y5year_pred = InjuryModelXGB5year.predict(x5year_test)
        print(r2_score(y5year_test, y5year_pred))
        InjuryModelXGB5year.save_model('xgb_injury_5year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y5year_test, y5year_pred)) + '.json')
        files.download('xgb_injury_5year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y5year_test, y5year_pred)) + '.json')

#Check sample Ousmane Dembélé, row 5999, why over 365
InjuryModelOneYeardf.sample(10)

X5years_train, X5years_test, y5years_train, y5years_test = train_test_split(x5years, y5years, random_state=1, train_size = 0.8)
InjuryModelXGB5years = xgb.XGBRegressor(tree_method="approx", seed=1, enable_categorical = True)
InjuryModelXGB5years.fit(X5years_train, y5years_train, verbose = True, early_stopping_rounds = 10, eval_set = [(X5years_test, y5years_test)])

y6years_pred = InjuryModelXGB6years.predict(X6years_test)
print(r2_score(y6years_test, y6years_pred))
print(mean_absolute_error(y6years_test, y6years_pred))
print(mean_squared_error(y6years_test, y6years_pred))

plot_tree(model)

#Shap test
graphviz6years = xgb.to_graphviz(InjuryModelXGB6years)
explainer6years = shap.Explainer(graphviz6years)
shap_values6years = explainer6years.shap_values(X6years_test)
explainer6years.plot_importance(shap_values6years, X6years_test)
shap.summary_plot(shap_values6years, X6years_test)

## injury prediction for 1 year sliding window
## need to check date joined club, if change club, change values to -1, should be fixed since injury data on all history of player recorded
#drop player value column

#concatenate repeated players into same row
#add features into methods
#sliding window, rolling
#create numpy array to store r squared/mean squared of accuracy of each model with adjusted hyperparameters
#start with set seed
#5-10 seeds, find average of models
#cross validation

#read df for 1 year model
MarketModelOneYeardf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Market1year.csv')
MarketModelOneYeardf.drop(columns = ['Player', 'Unnamed: 0'], axis=1, inplace=True)
xMarket1year = MarketModelOneYeardf.dropna(subset=['player_market_value_euro'])
yMarket1year = xMarket1year['player_market_value_euro'].copy()
xMarket1year = xMarket1year.drop("player_market_value_euro", axis = 1)

#read df for 2 year model
MarketModelTwoYeardf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Market2year.csv')
MarketModelTwoYeardf.drop(columns = ['Player', 'Player_xyear2', 'Unnamed: 0'], axis=1, inplace=True)
xMarket2year = MarketModelTwoYeardf.dropna(subset=['player_market_value_euroyear2'])
yMarket2year = xMarket2year['player_market_value_euroyear2'].copy()
xMarket2year = xMarket2year.drop("player_market_value_euroyear2", axis = 1)

#read df for 3 year model
MarketModelThreeYeardf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Market3year.csv')
MarketModelThreeYeardf.drop(columns = ['Player', 'Player_xyear2', 'Player_xyear3', 'Unnamed: 0'], axis=1, inplace=True)
xMarket3year = MarketModelThreeYeardf.dropna(subset=['player_market_value_euroyear3'])
yMarket3year = xMarket3year['player_market_value_euroyear3'].copy()
xMarket3year = xMarket3year.drop("player_market_value_euroyear3", axis = 1)

#read df for 4 year model
MarketModelFourYeardf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Market4year.csv')
MarketModelFourYeardf.drop(columns = ['Player', 'Player_xyear2', 'Player_xyear3', 'Player_xyear4', 'Unnamed: 0'], axis=1, inplace=True)
xMarket4year = MarketModelFourYeardf.dropna(subset=['player_market_value_euroyear4'])
yMarket4year = xMarket4year['player_market_value_euroyear4'].copy()
xMarket4year = xMarket4year.drop("player_market_value_euroyear4", axis = 1)

#read df for 5 year model
MarketModelFiveYeardf = pd.read_csv('/content/drive/MyDrive/PlayerValue/Market5year.csv')
MarketModelFiveYeardf.drop(columns = ['Player', 'Player_xyear2', 'Player_xyear3', 'Player_xyear4', 'Player_xyear5', 'Unnamed: 0'], axis=1, inplace=True)
xMarket5year = MarketModelFiveYeardf.dropna(subset=['player_market_value_euroyear5'])
yMarket5year = xMarket5year['player_market_value_euroyear5'].copy()
xMarket5year = xMarket5year.drop("player_market_value_euroyear5", axis = 1)

# @title
xMarket1year['injurytype_nan'] = 0 #nan

xMarket1year['injurytype_abdomen'] = 0 #Abdominal Strain

xMarket1year['injurytype_abductor'] = 0 #Tear in the abductor muscle,

xMarket1year['inj1urytype_achilles'] = 0 #Achilles tendon rupture, Achilles tendon problems,

xMarket1year['injurytype_adductor'] = 0 #Adductor problems,

xMarket1year['injurytype_ankle'] = 0 #Ankle Injury, Sprained ankle, Ankle problems, Torn ankle ligament, Ankle Surgery

xMarket1year['injurytype_arm'] = 0 #Arm Injury, Wirst Injury, Shoulder Injury

xMarket1year['injurytype_back'] = 0 #Back Injury, Back trouble, Back bruise

xMarket1year['injurytype_bruise'] = 0 #Bruise

xMarket1year['injurytype_calf'] = 0 #Calf Problems, Calf Injury, Calf Strain

xMarket1year['injurytype_chest'] = 0 #Angina,

xMarket1year['injurytype_collarbone'] = 0 #Collarbone fracture

xMarket1year['injurytype_concussion'] = 0 #Concussion

xMarket1year['injurytype_cruciateligament'] = 0 #Cruciate Ligament Rupture, Cruciate Ligament Surgery, Cruciate Ligament Strain

xMarket1year['injurytype_face'] = 0 #Facial Injury

xMarket1year['injurytype_finger'] = 0 #Finger Injury

xMarket1year['injurytype_fitness'] = 0 #Fitness

xMarket1year['injurytype_foot'] = 0 #Foot Injury, Bruised Foot, Fractured Foot,

xMarket1year['injurytype_groin'] = 0 #groin strain, Groin Strain (w/Capital), Groin Injury,

xMarket1year['injurytype_hamstring'] = 0 #Hamstring Injury

xMarket1year['injurytype_hand'] = 0 #Hand Injury, Hand Problems,

xMarket1year['injurytype_head'] = 0 #Head Injury

xMarket1year['injurytype_heart'] = 0 # Heart Condition

xMarket1year['injurytype_hip'] = 0 #Hip problems, Hip Injury,

xMarket1year['injurytype_ill'] = 0 #Cold, Influenza, Flu, Abdominal Influenza, Ill, Corona virus, Infection, Virus Infection, Quaratine, Testicular Cancer
#include abdominal influenza, cold, ill, covid, influenza, fever, quarantine, infection

xMarket1year['injurytype_inflammation'] = 0 #Inflammation

xMarket1year['injurytype_knee'] = 0 #Knee Injury, Knee Problems, Knee inflammation, Bruised Knee, Ruptured intraarticular ligament initiation in knee,

xMarket1year['injurytype_knock'] = 0 #Knock, Minor Knock,

xMarket1year['injurytype_leg'] = 0 #Leg Injury

xMarket1year['injurytype_muscular'] = 0 #Muscular problems, Muscle Injury, Biceps femoris muscle injury, Torn Muscle Fibre, Muscle Fatigue

xMarket1year['injurytype_pelvis'] = 0 #Pubitis, bruised pelvis, Pelvis Injury, Pubis bone irritation

xMarket1year['injurytype_rest'] = 0 #Rest

xMarket1year['injurytype_rib'] = 0 #Fractured Rib

xMarket1year['injurytype_spine'] = 0 #Lumbar Vertebra Fracture, Lumbar vertebrae problems, Blockade in the spinal, Cervical spine injury

xMarket1year['injurytype_thigh'] = 0 #Thigh Muscle Strain, Thigh Problems,

xMarket1year['injurytype_toe'] = 0 #Toe Injury

xMarket1year['injurytype_unknown'] = 0 #Unknown Injury,

xMarket1year['injurytype_other'] = 0 #Surgery, Contracture, Sprain, Laceration

for i, row in xMarket1year.iterrows(): #len(xMarket1year)
  s = row['injuryType']
  if (s == 'nan'):
    xMarket1year.at[i, 'injurytype_nan'] += 1
  elif type(s) == str:
    injury_list = s.split(" ,")
    injury_list = injury_list[:-1]
    for injury in injury_list:
      injury = injury.strip()
      if (injury == 'Abdominal Strain' or injury == 'Abdominal muscles injury' or injury == 'Appendectomy' or injury == 'Umbilical hernia'):
        xMarket1year.at[i, 'injurytype_abdomen'] += 1
      elif (injury == 'Tear in the abductor muscle'):
        xMarket1year.at[i, 'injurytype_abductor'] += 1
      elif (injury == 'Achilles tendon rupture' or injury == 'Achilles tendon problems' or injury == 'Achilles Irritation' or injury == 'Achilles tendon surgery'):
        xMarket1year.at[i, 'injurytype_achilles'] += 1
      elif (injury == 'Adductor problems'):
        xMarket1year.at[i, 'injurytype_adductor'] += 1
      elif (injury == 'Ankle Injury' or injury == 'Sprained ankle' or injury == 'Ankle problems' or injury == 'Torn ankle ligament' or injury == 'Ankle Surgery' or injury == 'Bruised Ankle' or injury == 'Distortion of the ankle' or injury == 'Ruptured syndesmotic ligament' or injury == 'Torn Ankle Ligament' or injury == 'Fracture-dislocation of the ankle' or injury == 'Sprained Ankle' or injury == 'Injury to the ankle' or injury == 'Ankle fracture' or injury == 'Ankle Inflammation' or injury == 'Syndesmotic ligament tear' or injury == 'Ankle Fracture' or injury == 'Capsular rupture in the ankle' or injury == 'Ruptured ankle ligament' or injury == 'Ruptured ankle aigament' or injury == 'Bruise on ankle' or injury == 'Ruptured intraarticular ligament initiation in the ankle' or injury == 'Peroneus tendon injury'):
        xMarket1year.at[i, 'injurytype_ankle'] += 1
      elif (injury == 'Arm Injury' or injury == 'Shoulder Injury' or injury == 'Shoulder fracture' or injury == 'Elbow Injury' or injury == 'Bruised Acromioclavicular' or injury == 'Fractured Arm' or injury == 'Acromioclavicular Separation'):
        xMarket1year.at[i, 'injurytype_arm'] += 1
      elif (injury == 'Back Injury' or injury == 'Back trouble' or injury == 'Back bruise' or injury == 'Lumbago'):
        xMarket1year.at[i, 'injurytype_back'] += 1
      elif (injury == 'Bruise' or injury == 'Bruised Rib' or injury == 'Metatarsal bone bruise' or injury == 'Bone Bruise' or injury == 'Muscle bruise'):
        xMarket1year.at[i, 'injurytype_bruise'] += 1
      elif (injury == 'Calf Problems' or injury == 'Calf Injury' or injury == 'Calf Strain' or injury == 'Disrupted Calf Muscle' or injury == 'Hairline crack in calfbone' or injury == 'Calf muscle strain'):
        xMarket1year.at[i, 'injurytype_calf'] += 1
      elif (injury == 'Angina' or injury == 'Chest injury' or injury == 'Pneumothorax' or injury == 'Lung contusion'):
        xMarket1year.at[i, 'injurytype_chest'] += 1
      elif (injury == 'Collarbone fracture'):
        xMarket1year.at[i, 'injurytype_collarbone'] += 1
      elif (injury == 'Concussion'):
        xMarket1year.at[i, 'injurytype_concussion'] += 1
      elif (injury == 'Cruciate Ligament Rupture' or injury == 'Cruciate Ligament Surgery' or injury == 'Cruciate Ligament Strain' or injury == 'Cruciate Ligament Injury' or injury == 'Ruptured cruciate ligament' or injury == 'Partial damage to the cruciate ligament' or injury == 'Cruciate ligament stretch'):
        xMarket1year.at[i, 'injurytype_cruciateligament'] += 1
      elif (injury == 'Facial Injury' or injury == 'Nose surgery' or injury == 'Dental Surgery' or injury == 'Nasal Bone Fracture' or injury == 'Facial Fracture' or injury == 'Cheekbone Fracture' or injury == 'Eye Injury' or injury == 'Toothache' or injury == 'Frontal bone fracture' or injury == 'Fractured Jaw' or injury == 'Fractured Skull' or injury == 'Fracture of the orbit' or injury == 'Tooth Inflammation' or injury == 'Nose Injury'):
        xMarket1year.at[i, 'injurytype_face'] += 1
      elif (injury == 'Finger Injury'):
        xMarket1year.at[i, 'injurytype_finger'] += 1
      elif (injury == 'Fitness'):
        xMarket1year.at[i, 'injurytype_fitness'] += 1
      elif (injury == 'Foot Injury' or injury == 'Bruised Foot' or injury == 'Fractured Foot' or injury == 'Arch pain' or injury == 'Heel Injury' or injury == 'Hell pain' or injury == 'Heelspur' or injury == 'Heel Bone Injury' or injury == 'Metatarsal Fracture' or injury == 'Heel pain' or injury == 'Plantar fascia' or injury == 'Foot surgery' or injury == 'Hairline crack in the foot' or injury == 'Partial demolition of the plantar fascia'):
        xMarket1year.at[i, 'injurytype_foot'] += 1
      elif (injury == 'groin strain' or injury == 'Groin Strain' or injury == 'Groin Injury' or injury == 'Groin Surgery' or injury == 'Inguinal Hernia' or injury == 'Pubalgia' or injury == 'Testicular disruption'):
        xMarket1year.at[i, 'injurytype_groin'] += 1
      elif (injury == 'Hamstring Injury' or injury == 'Pulled hamstring at the adductors' or injury == 'Hamstring contusion'):
        xMarket1year.at[i, 'injurytype_hamstring'] += 1
      elif (injury == 'Hand Injury' or injury == 'Fractured Hand' or injury == 'Broken wrist' or injury == 'Wirst Injury' or injury == 'scaphoid operation' or injury == 'Hand fracture' or injury == 'Fractured Finger' or injury == 'Thumb Injury'):
        xMarket1year.at[i, 'injurytype_hand'] += 1
      elif (injury == 'Head Injury' or injury == 'Neck Injury' or injury == 'Neck bruise'):
        xMarket1year.at[i, 'injurytype_head'] += 1
      elif (injury == 'Heart Condition'):
        xMarket1year.at[i, 'injurytype_heart'] += 1
      elif (injury == 'Hip problems' or injury == 'Hip Injury' or injury == 'Problems with the hip flexor' or injury == 'Bruised Hip' or injury == 'Pubis bone contusion' or injury == 'Problems with the right hip flexor'):
        xMarket1year.at[i, 'injurytype_hip'] += 1
      elif (injury == 'Cold' or injury == 'Influenza' or injury == 'Flu' or injury == 'Abdominal Influenza' or injury == 'Ill' or injury == 'Corona virus' or injury == 'Infection' or injury == 'Virus Infection' or injury == 'Quarantine' or injury == 'Fever' or injury == 'Tonsillitis' or injury == 'Testicular Cancer' or injury == 'Pneumonia' or injury == 'Gastric problems' or injury == 'Mononucleosis' or injury == 'Chickenpox' or injury == 'Bronchitis' or injury == 'Stomach complaints' or injury == 'Food Poisoning' or injury == 'intestial virus' or injury == 'Infected wound' or injury == 'Malaria' or injury == 'cancer' or injury == 'Lymphoma' or injury == 'Depression' or injury == 'Cals Sclerosis' or injury == 'Kidney problems'):
        xMarket1year.at[i, 'injurytype_ill'] += 1
      elif (injury == 'Inflammation'):
        xMarket1year.at[i, 'injurytype_inflammation'] += 1
      elif (injury == 'Knee Injury' or injury == 'Knee Problems' or injury == 'Knee inflammation' or injury == 'Bruised Knee' or injury == 'Ruptured intraarticular ligament initiation in knee' or injury == 'Knee Surgery' or injury == 'Medial Collateral Ligament Tear' or injury == 'Medial Collateral Ligament Injury' or injury == 'Meniscal Injury' or injury == 'Sideband strain in the knee' or injury == 'Patella tendon irritation' or injury == 'Mensical Laceration' or injury == 'Twisted knee' or injury == 'Torn Knee Ligament' or injury == 'Medial Collateral Ligament Knee Injury' or injury == 'Patella problems' or injury == 'Ruptured lateral collateral ligament' or injury == 'Edema in the knee' or injury == 'Fractured Kneecap' or injury == 'Torn lateral collateral ligament' or injury == 'Torn Meniscus' or injury == 'Torn Collateral Ligament' or injury == 'Patella rupture' or injury == 'Torn knee ligament' or injury == 'Patella tendon luxation' or injury == 'Ruptured knee ligament' or injury == 'Rupture of Outer Meniscus' or injury == 'Double Torn Ligament' or injury == 'Meniscus Damage' or injury == 'Medial Collateral Ligament avulsion' or injury == 'Meniscus irritation' or injury == 'Inflamed ligaments of the knee' or injury == 'Rupture of the pattella'):
        xMarket1year.at[i, 'injurytype_knee'] += 1
      elif (injury == 'Knock' or injury == 'Minor Knock' or injury == 'Dead Leg' or injury == 'Stiffness'):
        xMarket1year.at[i, 'injurytype_knock'] += 1
      elif (injury == 'Leg Injury' or injury == 'Strain in the thigh and gluteal muscles' or injury == 'Shinbone injury' or injury == 'Shin bone bruise' or injury == 'Biceps femoris muscle injury' or injury == 'Fibula Fracture' or injury == 'Fractured Leg' or injury == 'Fissure of the fibula' or injury == 'Sciatic Problem' or injury == 'Tibia and Fibula Fracture' or injury == 'Fracture of the lower leg' or injury == 'Tibia Fracture' or injury == 'Inflamed head of fibula'):
        xMarket1year.at[i, 'injurytype_leg'] += 1
      elif (injury == 'Muscular problems' or injury == 'Muscle Injury' or injury == 'Torn Muscle Fibre' or injury == 'Muscle Fatigue' or injury == 'Torn muscle bundle' or injury == 'Torn Muscle' or injury == 'Muscle fiber tear'):
        xMarket1year.at[i, 'injurytype_muscular'] += 1
      elif (injury == 'Pubitis' or injury == 'Bruised pelvis' or injury == 'Pelvis Injury' or injury == 'Pubis bone irritation' or injury == 'bruised pelvis' or injury == 'bruised pelvis' or injury == 'Pelvic obliquity'):
        xMarket1year.at[i, 'injurytype_pelvis'] += 1
      elif (injury == 'Rest'):
        xMarket1year.at[i, 'injurytype_rest'] += 1
      elif (injury == 'Fractured Rib'):
        xMarket1year.at[i, 'injurytype_rib'] += 1
      elif (injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar vertebrae problems' or injury == 'Blockade in the spinal' or injury == 'Cervical spine injury' or injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar Vertebra Fracture' or injury == 'Vertebra injury'):
        xMarket1year.at[i, 'injurytype_spine'] += 1
      elif (injury == 'Thigh Muscle Strain' or injury == 'Thigh Problems' or injury == 'Thigh muscle rupture'):
        xMarket1year.at[i, 'injurytype_thigh'] += 1
      elif (injury == 'Toe Injury' or injury == 'Fractured Toe'):
        xMarket1year.at[i, 'injurytype_toe'] += 1
      elif (injury == 'Unknown Injury'):
        xMarket1year.at[i, 'injurytype_unknown'] += 1
      elif (injury == 'Surgery' or injury == 'Contracture' or injury == 'Sprain' or injury == 'Laceration' or injury == 'Fracture' or injury == 'Ligament Injury' or injury == 'Stress response of the bone' or injury == 'Strain' or injury == 'Contused laceration' or injury == 'Muscle partial avulsion' or injury == 'torn tendon' or injury == 'Tear in a joint capsule' or injury == 'tendon irritation' or injury == 'Ligament Problems' or injury == 'Sideband injury' or injury == 'Bone buckling' or injury == 'Stretched Ligament' or injury == 'Pinched nerve' or injury == 'Overstretching' or injury == 'Tendonitis' or injury == 'laceration' or injury == 'Arthroscopie' or injury == 'Herniated Disc' or injury == 'Fatigue fracture' or injury == 'Torn Ligament' or injury == 'Cartilage Damage' or injury == 'Bursitis' or injury == 'Circulation Problems' or injury == 'Tendon crack' or injury == 'Cut' or injury == 'sprain' or injury == 'Burns' or injury == 'open wound' or injury == 'Sideband tear' or injury == 'Marrow bulge' or injury == 'Compartment syndrome' or injury == 'Vestibular disorder' or injury == 'Intraarticular ligament fissure' or injury == 'Ruptured ligaments' or injury == 'Muscular hairline crack' or injury == 'capsular tear' or injury == 'Traffic Accident' or injury == 'Flesh Wound'):
        xMarket1year.at[i, 'injurytype_other'] += 1
      else:
        print(injury)

xMarket1year.insert(5, "GK", 0)
xMarket1year.insert(6, "DF", 0)
xMarket1year.insert(7, "MF", 0)
xMarket1year.insert(8, "FW", 0)

for i, row in xMarket1year.iterrows(): #len(xMarket1year)
  s = row['Pos']
  if type(s) == str:
    pos_list = s.split(",")
    for pos in pos_list:
      pos = pos.strip()
      if (pos == 'GK'):
        xMarket1year.at[i, 'GK'] += 1
      elif (pos == 'DF'):
        xMarket1year.at[i, 'DF'] += 1
      elif (pos == 'MF'):
        xMarket1year.at[i, 'MF'] += 1
      elif (pos == 'FW'):
        xMarket1year.at[i, 'FW'] += 1

transferValue = xMarket1year.pop("player_market_value_euro")
xMarket1year["player_market_value_euro"] = transferValue

xMarket2year['injurytype_nan'] = 0 #nan

xMarket2year['injurytype_abdomen'] = 0 #Abdominal Strain

xMarket2year['injurytype_abductor'] = 0 #Tear in the abductor muscle,

xMarket2year['injurytype_achilles'] = 0 #Achilles tendon rupture, Achilles tendon problems,

xMarket2year['injurytype_adductor'] = 0 #Adductor problems,

xMarket2year['injurytype_ankle'] = 0 #Ankle Injury, Sprained ankle, Ankle problems, Torn ankle ligament, Ankle Surgery

xMarket2year['injurytype_arm'] = 0 #Arm Injury, Wirst Injury, Shoulder Injury

xMarket2year['injurytype_back'] = 0 #Back Injury, Back trouble, Back bruise

xMarket2year['injurytype_bruise'] = 0 #Bruise

xMarket2year['injurytype_calf'] = 0 #Calf Problems, Calf Injury, Calf Strain

xMarket2year['injurytype_chest'] = 0 #Angina,

xMarket2year['injurytype_collarbone'] = 0 #Collarbone fracture

xMarket2year['injurytype_concussion'] = 0 #Concussion

xMarket2year['injurytype_cruciateligament'] = 0 #Cruciate Ligament Rupture, Cruciate Ligament Surgery, Cruciate Ligament Strain

xMarket2year['injurytype_face'] = 0 #Facial Injury

xMarket2year['injurytype_finger'] = 0 #Finger Injury

xMarket2year['injurytype_fitness'] = 0 #Fitness

xMarket2year['injurytype_foot'] = 0 #Foot Injury, Bruised Foot, Fractured Foot,

xMarket2year['injurytype_groin'] = 0 #groin strain, Groin Strain (w/Capital), Groin Injury,

xMarket2year['injurytype_hamstring'] = 0 #Hamstring Injury

xMarket2year['injurytype_hand'] = 0 #Hand Injury, Hand Problems,

xMarket2year['injurytype_head'] = 0 #Head Injury

xMarket2year['injurytype_heart'] = 0 # Heart Condition

xMarket2year['injurytype_hip'] = 0 #Hip problems, Hip Injury,

xMarket2year['injurytype_ill'] = 0 #Cold, Influenza, Flu, Abdominal Influenza, Ill, Corona virus, Infection, Virus Infection, Quaratine, Testicular Cancer
#include abdominal influenza, cold, ill, covid, influenza, fever, quarantine, infection

xMarket2year['injurytype_inflammation'] = 0 #Inflammation

xMarket2year['injurytype_knee'] = 0 #Knee Injury, Knee Problems, Knee inflammation, Bruised Knee, Ruptured intraarticular ligament initiation in knee,

xMarket2year['injurytype_knock'] = 0 #Knock, Minor Knock,

xMarket2year['injurytype_leg'] = 0 #Leg Injury

xMarket2year['injurytype_muscular'] = 0 #Muscular problems, Muscle Injury, Biceps femoris muscle injury, Torn Muscle Fibre, Muscle Fatigue

xMarket2year['injurytype_pelvis'] = 0 #Pubitis, bruised pelvis, Pelvis Injury, Pubis bone irritation

xMarket2year['injurytype_rest'] = 0 #Rest

xMarket2year['injurytype_rib'] = 0 #Fractured Rib

xMarket2year['injurytype_spine'] = 0 #Lumbar Vertebra Fracture, Lumbar vertebrae problems, Blockade in the spinal, Cervical spine injury

xMarket2year['injurytype_thigh'] = 0 #Thigh Muscle Strain, Thigh Problems,

xMarket2year['injurytype_toe'] = 0 #Toe Injury

xMarket2year['injurytype_unknown'] = 0 #Unknown Injury,

xMarket2year['injurytype_other'] = 0 #Surgery, Contracture, Sprain, Laceration

for i, row in xMarket2year.iterrows(): #len(x1year)
  s = row['injuryType']
  if (s == 'nan'):
    xMarket2year.at[i, 'injurytype_nan'] += 1
  elif type(s) == str:
    injury_list = s.split(" ,")
    injury_list = injury_list[:-1]
    for injury in injury_list:
      injury = injury.strip()
      if (injury == 'Abdominal Strain' or injury == 'Abdominal muscles injury' or injury == 'Appendectomy' or injury == 'Umbilical hernia'):
        xMarket2year.at[i, 'injurytype_abdomen'] += 1
      elif (injury == 'Tear in the abductor muscle'):
        xMarket2year.at[i, 'injurytype_abductor'] += 1
      elif (injury == 'Achilles tendon rupture' or injury == 'Achilles tendon problems' or injury == 'Achilles Irritation' or injury == 'Achilles tendon surgery'):
        xMarket2year.at[i, 'injurytype_achilles'] += 1
      elif (injury == 'Adductor problems'):
        xMarket2year.at[i, 'injurytype_adductor'] += 1
      elif (injury == 'Ankle Injury' or injury == 'Sprained ankle' or injury == 'Ankle problems' or injury == 'Torn ankle ligament' or injury == 'Ankle Surgery' or injury == 'Bruised Ankle' or injury == 'Distortion of the ankle' or injury == 'Ruptured syndesmotic ligament' or injury == 'Torn Ankle Ligament' or injury == 'Fracture-dislocation of the ankle' or injury == 'Sprained Ankle' or injury == 'Injury to the ankle' or injury == 'Ankle fracture' or injury == 'Ankle Inflammation' or injury == 'Syndesmotic ligament tear' or injury == 'Ankle Fracture' or injury == 'Capsular rupture in the ankle' or injury == 'Ruptured ankle ligament' or injury == 'Ruptured ankle aigament' or injury == 'Bruise on ankle' or injury == 'Ruptured intraarticular ligament initiation in the ankle' or injury == 'Peroneus tendon injury'):
        xMarket2year.at[i, 'injurytype_ankle'] += 1
      elif (injury == 'Arm Injury' or injury == 'Shoulder Injury' or injury == 'Shoulder fracture' or injury == 'Elbow Injury' or injury == 'Bruised Acromioclavicular' or injury == 'Fractured Arm' or injury == 'Acromioclavicular Separation'):
        xMarket2year.at[i, 'injurytype_arm'] += 1
      elif (injury == 'Back Injury' or injury == 'Back trouble' or injury == 'Back bruise' or injury == 'Lumbago'):
        xMarket2year.at[i, 'injurytype_back'] += 1
      elif (injury == 'Bruise' or injury == 'Bruised Rib' or injury == 'Metatarsal bone bruise' or injury == 'Bone Bruise' or injury == 'Muscle bruise'):
        xMarket2year.at[i, 'injurytype_bruise'] += 1
      elif (injury == 'Calf Problems' or injury == 'Calf Injury' or injury == 'Calf Strain' or injury == 'Disrupted Calf Muscle' or injury == 'Hairline crack in calfbone' or injury == 'Calf muscle strain'):
        xMarket2year.at[i, 'injurytype_calf'] += 1
      elif (injury == 'Angina' or injury == 'Chest injury' or injury == 'Pneumothorax' or injury == 'Lung contusion'):
        xMarket2year.at[i, 'injurytype_chest'] += 1
      elif (injury == 'Collarbone fracture'):
        xMarket2year.at[i, 'injurytype_collarbone'] += 1
      elif (injury == 'Concussion'):
        xMarket2year.at[i, 'injurytype_concussion'] += 1
      elif (injury == 'Cruciate Ligament Rupture' or injury == 'Cruciate Ligament Surgery' or injury == 'Cruciate Ligament Strain' or injury == 'Cruciate Ligament Injury' or injury == 'Ruptured cruciate ligament' or injury == 'Partial damage to the cruciate ligament' or injury == 'Cruciate ligament stretch'):
        xMarket2year.at[i, 'injurytype_cruciateligament'] += 1
      elif (injury == 'Facial Injury' or injury == 'Nose surgery' or injury == 'Dental Surgery' or injury == 'Nasal Bone Fracture' or injury == 'Facial Fracture' or injury == 'Cheekbone Fracture' or injury == 'Eye Injury' or injury == 'Toothache' or injury == 'Frontal bone fracture' or injury == 'Fractured Jaw' or injury == 'Fractured Skull' or injury == 'Fracture of the orbit' or injury == 'Tooth Inflammation' or injury == 'Nose Injury'):
        xMarket2year.at[i, 'injurytype_face'] += 1
      elif (injury == 'Finger Injury'):
        xMarket2year.at[i, 'injurytype_finger'] += 1
      elif (injury == 'Fitness'):
        xMarket2year.at[i, 'injurytype_fitness'] += 1
      elif (injury == 'Foot Injury' or injury == 'Bruised Foot' or injury == 'Fractured Foot' or injury == 'Arch pain' or injury == 'Heel Injury' or injury == 'Hell pain' or injury == 'Heelspur' or injury == 'Heel Bone Injury' or injury == 'Metatarsal Fracture' or injury == 'Heel pain' or injury == 'Plantar fascia' or injury == 'Foot surgery' or injury == 'Hairline crack in the foot' or injury == 'Partial demolition of the plantar fascia'):
        xMarket2year.at[i, 'injurytype_foot'] += 1
      elif (injury == 'groin strain' or injury == 'Groin Strain' or injury == 'Groin Injury' or injury == 'Groin Surgery' or injury == 'Inguinal Hernia' or injury == 'Pubalgia' or injury == 'Testicular disruption'):
        xMarket2year.at[i, 'injurytype_groin'] += 1
      elif (injury == 'Hamstring Injury' or injury == 'Pulled hamstring at the adductors' or injury == 'Hamstring contusion'):
        xMarket2year.at[i, 'injurytype_hamstring'] += 1
      elif (injury == 'Hand Injury' or injury == 'Fractured Hand' or injury == 'Broken wrist' or injury == 'Wirst Injury' or injury == 'scaphoid operation' or injury == 'Hand fracture' or injury == 'Fractured Finger' or injury == 'Thumb Injury'):
        xMarket2year.at[i, 'injurytype_hand'] += 1
      elif (injury == 'Head Injury' or injury == 'Neck Injury' or injury == 'Neck bruise'):
        xMarket2year.at[i, 'injurytype_head'] += 1
      elif (injury == 'Heart Condition'):
        xMarket2year.at[i, 'injurytype_heart'] += 1
      elif (injury == 'Hip problems' or injury == 'Hip Injury' or injury == 'Problems with the hip flexor' or injury == 'Bruised Hip' or injury == 'Pubis bone contusion' or injury == 'Problems with the right hip flexor'):
        xMarket2year.at[i, 'injurytype_hip'] += 1
      elif (injury == 'Cold' or injury == 'Influenza' or injury == 'Flu' or injury == 'Abdominal Influenza' or injury == 'Ill' or injury == 'Corona virus' or injury == 'Infection' or injury == 'Virus Infection' or injury == 'Quarantine' or injury == 'Fever' or injury == 'Tonsillitis' or injury == 'Testicular Cancer' or injury == 'Pneumonia' or injury == 'Gastric problems' or injury == 'Mononucleosis' or injury == 'Chickenpox' or injury == 'Bronchitis' or injury == 'Stomach complaints' or injury == 'Food Poisoning' or injury == 'intestial virus' or injury == 'Infected wound' or injury == 'Malaria' or injury == 'cancer' or injury == 'Lymphoma' or injury == 'Depression' or injury == 'Cals Sclerosis' or injury == 'Kidney problems'):
        xMarket2year.at[i, 'injurytype_ill'] += 1
      elif (injury == 'Inflammation'):
        xMarket2year.at[i, 'injurytype_inflammation'] += 1
      elif (injury == 'Knee Injury' or injury == 'Knee Problems' or injury == 'Knee inflammation' or injury == 'Bruised Knee' or injury == 'Ruptured intraarticular ligament initiation in knee' or injury == 'Knee Surgery' or injury == 'Medial Collateral Ligament Tear' or injury == 'Medial Collateral Ligament Injury' or injury == 'Meniscal Injury' or injury == 'Sideband strain in the knee' or injury == 'Patella tendon irritation' or injury == 'Mensical Laceration' or injury == 'Twisted knee' or injury == 'Torn Knee Ligament' or injury == 'Medial Collateral Ligament Knee Injury' or injury == 'Patella problems' or injury == 'Ruptured lateral collateral ligament' or injury == 'Edema in the knee' or injury == 'Fractured Kneecap' or injury == 'Torn lateral collateral ligament' or injury == 'Torn Meniscus' or injury == 'Torn Collateral Ligament' or injury == 'Patella rupture' or injury == 'Torn knee ligament' or injury == 'Patella tendon luxation' or injury == 'Ruptured knee ligament' or injury == 'Rupture of Outer Meniscus' or injury == 'Double Torn Ligament' or injury == 'Meniscus Damage' or injury == 'Medial Collateral Ligament avulsion' or injury == 'Meniscus irritation' or injury == 'Inflamed ligaments of the knee' or injury == 'Rupture of the pattella'):
        xMarket2year.at[i, 'injurytype_knee'] += 1
      elif (injury == 'Knock' or injury == 'Minor Knock' or injury == 'Dead Leg' or injury == 'Stiffness'):
        xMarket2year.at[i, 'injurytype_knock'] += 1
      elif (injury == 'Leg Injury' or injury == 'Strain in the thigh and gluteal muscles' or injury == 'Shinbone injury' or injury == 'Shin bone bruise' or injury == 'Biceps femoris muscle injury' or injury == 'Fibula Fracture' or injury == 'Fractured Leg' or injury == 'Fissure of the fibula' or injury == 'Sciatic Problem' or injury == 'Tibia and Fibula Fracture' or injury == 'Fracture of the lower leg' or injury == 'Tibia Fracture' or injury == 'Inflamed head of fibula'):
        xMarket2year.at[i, 'injurytype_leg'] += 1
      elif (injury == 'Muscular problems' or injury == 'Muscle Injury' or injury == 'Torn Muscle Fibre' or injury == 'Muscle Fatigue' or injury == 'Torn muscle bundle' or injury == 'Torn Muscle' or injury == 'Muscle fiber tear'):
        xMarket2year.at[i, 'injurytype_muscular'] += 1
      elif (injury == 'Pubitis' or injury == 'Bruised pelvis' or injury == 'Pelvis Injury' or injury == 'Pubis bone irritation' or injury == 'bruised pelvis' or injury == 'bruised pelvis' or injury == 'Pelvic obliquity'):
        xMarket2year.at[i, 'injurytype_pelvis'] += 1
      elif (injury == 'Rest'):
        xMarket2year.at[i, 'injurytype_rest'] += 1
      elif (injury == 'Fractured Rib'):
        xMarket2year.at[i, 'injurytype_rib'] += 1
      elif (injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar vertebrae problems' or injury == 'Blockade in the spinal' or injury == 'Cervical spine injury' or injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar Vertebra Fracture' or injury == 'Vertebra injury'):
        xMarket2year.at[i, 'injurytype_spine'] += 1
      elif (injury == 'Thigh Muscle Strain' or injury == 'Thigh Problems' or injury == 'Thigh muscle rupture'):
        xMarket2year.at[i, 'injurytype_thigh'] += 1
      elif (injury == 'Toe Injury' or injury == 'Fractured Toe'):
        xMarket2year.at[i, 'injurytype_toe'] += 1
      elif (injury == 'Unknown Injury'):
        xMarket2year.at[i, 'injurytype_unknown'] += 1
      elif (injury == 'Surgery' or injury == 'Contracture' or injury == 'Sprain' or injury == 'Laceration' or injury == 'Fracture' or injury == 'Ligament Injury' or injury == 'Stress response of the bone' or injury == 'Strain' or injury == 'Contused laceration' or injury == 'Muscle partial avulsion' or injury == 'torn tendon' or injury == 'Tear in a joint capsule' or injury == 'tendon irritation' or injury == 'Ligament Problems' or injury == 'Sideband injury' or injury == 'Bone buckling' or injury == 'Stretched Ligament' or injury == 'Pinched nerve' or injury == 'Overstretching' or injury == 'Tendonitis' or injury == 'laceration' or injury == 'Arthroscopie' or injury == 'Herniated Disc' or injury == 'Fatigue fracture' or injury == 'Torn Ligament' or injury == 'Cartilage Damage' or injury == 'Bursitis' or injury == 'Circulation Problems' or injury == 'Tendon crack' or injury == 'Cut' or injury == 'sprain' or injury == 'Burns' or injury == 'open wound' or injury == 'Sideband tear' or injury == 'Marrow bulge' or injury == 'Compartment syndrome' or injury == 'Vestibular disorder' or injury == 'Intraarticular ligament fissure' or injury == 'Ruptured ligaments' or injury == 'Muscular hairline crack' or injury == 'capsular tear' or injury == 'Traffic Accident' or injury == 'Flesh Wound'):
        xMarket2year.at[i, 'injurytype_other'] += 1
      else:
        print(injury)

xMarket2year.insert(5, "GK", 0)
xMarket2year.insert(6, "DF", 0)
xMarket2year.insert(7, "MF", 0)
xMarket2year.insert(8, "FW", 0)

for i, row in xMarket2year.iterrows(): #len(xMarket2year)
  s = row['Pos']
  if type(s) == str:
    pos_list = s.split(",")
    for pos in pos_list:
      pos = pos.strip()
      if (pos == 'GK'):
        xMarket2year.at[i, 'GK'] += 1
      elif (pos == 'DF'):
        xMarket2year.at[i, 'DF'] += 1
      elif (pos == 'MF'):
        xMarket2year.at[i, 'MF'] += 1
      elif (pos == 'FW'):
        xMarket2year.at[i, 'FW'] += 1

transferValue = xMarket2year.pop("player_market_value_euro")
xMarket2year["player_market_value_euro"] = transferValue

#nan, lower body, upper body, ill, head, other
xMarket3year['injurytype_nan'] = 0 #nan

xMarket3year['injurytype_abdomen'] = 0 #Abdominal Strain

xMarket3year['injurytype_abductor'] = 0 #Tear in the abductor muscle,

xMarket3year['injurytype_achilles'] = 0 #Achilles tendon rupture, Achilles tendon problems,

xMarket3year['injurytype_adductor'] = 0 #Adductor problems,

xMarket3year['injurytype_ankle'] = 0 #Ankle Injury, Sprained ankle, Ankle problems, Torn ankle ligament, Ankle Surgery

xMarket3year['injurytype_arm'] = 0 #Arm Injury, Wirst Injury, Shoulder Injury

xMarket3year['injurytype_back'] = 0 #Back Injury, Back trouble, Back bruise

xMarket3year['injurytype_bruise'] = 0 #Bruise

xMarket3year['injurytype_calf'] = 0 #Calf Problems, Calf Injury, Calf Strain

xMarket3year['injurytype_chest'] = 0 #Angina,

xMarket3year['injurytype_collarbone'] = 0 #Collarbone fracture

xMarket3year['injurytype_concussion'] = 0 #Concussion

xMarket3year['injurytype_cruciateligament'] = 0 #Cruciate Ligament Rupture, Cruciate Ligament Surgery, Cruciate Ligament Strain

xMarket3year['injurytype_face'] = 0 #Facial Injury

xMarket3year['injurytype_finger'] = 0 #Finger Injury

xMarket3year['injurytype_fitness'] = 0 #Fitness

xMarket3year['injurytype_foot'] = 0 #Foot Injury, Bruised Foot, Fractured Foot,

xMarket3year['injurytype_groin'] = 0 #groin strain, Groin Strain (w/Capital), Groin Injury,

xMarket3year['injurytype_hamstring'] = 0 #Hamstring Injury

xMarket3year['injurytype_hand'] = 0 #Hand Injury, Hand Problems,

xMarket3year['injurytype_head'] = 0 #Head Injury

xMarket3year['injurytype_heart'] = 0 # Heart Condition

xMarket3year['injurytype_hip'] = 0 #Hip problems, Hip Injury,

xMarket3year['injurytype_ill'] = 0 #Cold, Influenza, Flu, Abdominal Influenza, Ill, Corona virus, Infection, Virus Infection, Quaratine, Testicular Cancer
#include abdominal influenza, cold, ill, covid, influenza, fever, quarantine, infection

xMarket3year['injurytype_inflammation'] = 0 #Inflammation

xMarket3year['injurytype_knee'] = 0 #Knee Injury, Knee Problems, Knee inflammation, Bruised Knee, Ruptured intraarticular ligament initiation in knee,

xMarket3year['injurytype_knock'] = 0 #Knock, Minor Knock,

xMarket3year['injurytype_leg'] = 0 #Leg Injury

xMarket3year['injurytype_muscular'] = 0 #Muscular problems, Muscle Injury, Biceps femoris muscle injury, Torn Muscle Fibre, Muscle Fatigue

xMarket3year['injurytype_pelvis'] = 0 #Pubitis, bruised pelvis, Pelvis Injury, Pubis bone irritation

xMarket3year['injurytype_rest'] = 0 #Rest

xMarket3year['injurytype_rib'] = 0 #Fractured Rib

xMarket3year['injurytype_spine'] = 0 #Lumbar Vertebra Fracture, Lumbar vertebrae problems, Blockade in the spinal, Cervical spine injury

xMarket3year['injurytype_thigh'] = 0 #Thigh Muscle Strain, Thigh Problems,

xMarket3year['injurytype_toe'] = 0 #Toe Injury

xMarket3year['injurytype_unknown'] = 0 #Unknown Injury,

xMarket3year['injurytype_other'] = 0 #Surgery, Contracture, Sprain, Laceration

for i, row in xMarket3year.iterrows(): #len(x1year)
  s = row['injuryType']
  if (s == 'nan'):
    xMarket3year.at[i, 'injurytype_nan'] += 1
  elif type(s) == str:
    injury_list = s.split(" ,")
    injury_list = injury_list[:-1]
    for injury in injury_list:
      injury = injury.strip()
      if (injury == 'Abdominal Strain' or injury == 'Abdominal muscles injury' or injury == 'Appendectomy' or injury == 'Umbilical hernia'):
        xMarket3year.at[i, 'injurytype_abdomen'] += 1
      elif (injury == 'Tear in the abductor muscle'):
        xMarket3year.at[i, 'injurytype_abductor'] += 1
      elif (injury == 'Achilles tendon rupture' or injury == 'Achilles tendon problems' or injury == 'Achilles Irritation' or injury == 'Achilles tendon surgery'):
        xMarket3year.at[i, 'injurytype_achilles'] += 1
      elif (injury == 'Adductor problems'):
        xMarket3year.at[i, 'injurytype_adductor'] += 1
      elif (injury == 'Ankle Injury' or injury == 'Sprained ankle' or injury == 'Ankle problems' or injury == 'Torn ankle ligament' or injury == 'Ankle Surgery' or injury == 'Bruised Ankle' or injury == 'Distortion of the ankle' or injury == 'Ruptured syndesmotic ligament' or injury == 'Torn Ankle Ligament' or injury == 'Fracture-dislocation of the ankle' or injury == 'Sprained Ankle' or injury == 'Injury to the ankle' or injury == 'Ankle fracture' or injury == 'Ankle Inflammation' or injury == 'Syndesmotic ligament tear' or injury == 'Ankle Fracture' or injury == 'Capsular rupture in the ankle' or injury == 'Ruptured ankle ligament' or injury == 'Ruptured ankle aigament' or injury == 'Bruise on ankle' or injury == 'Ruptured intraarticular ligament initiation in the ankle' or injury == 'Peroneus tendon injury'):
        xMarket3year.at[i, 'injurytype_ankle'] += 1
      elif (injury == 'Arm Injury' or injury == 'Shoulder Injury' or injury == 'Shoulder fracture' or injury == 'Elbow Injury' or injury == 'Bruised Acromioclavicular' or injury == 'Fractured Arm' or injury == 'Acromioclavicular Separation'):
        xMarket3year.at[i, 'injurytype_arm'] += 1
      elif (injury == 'Back Injury' or injury == 'Back trouble' or injury == 'Back bruise' or injury == 'Lumbago'):
        xMarket3year.at[i, 'injurytype_back'] += 1
      elif (injury == 'Bruise' or injury == 'Bruised Rib' or injury == 'Metatarsal bone bruise' or injury == 'Bone Bruise' or injury == 'Muscle bruise'):
        xMarket3year.at[i, 'injurytype_bruise'] += 1
      elif (injury == 'Calf Problems' or injury == 'Calf Injury' or injury == 'Calf Strain' or injury == 'Disrupted Calf Muscle' or injury == 'Hairline crack in calfbone' or injury == 'Calf muscle strain'):
        xMarket3year.at[i, 'injurytype_calf'] += 1
      elif (injury == 'Angina' or injury == 'Chest injury' or injury == 'Pneumothorax' or injury == 'Lung contusion'):
        xMarket3year.at[i, 'injurytype_chest'] += 1
      elif (injury == 'Collarbone fracture'):
        xMarket3year.at[i, 'injurytype_collarbone'] += 1
      elif (injury == 'Concussion'):
        xMarket3year.at[i, 'injurytype_concussion'] += 1
      elif (injury == 'Cruciate Ligament Rupture' or injury == 'Cruciate Ligament Surgery' or injury == 'Cruciate Ligament Strain' or injury == 'Cruciate Ligament Injury' or injury == 'Ruptured cruciate ligament' or injury == 'Partial damage to the cruciate ligament' or injury == 'Cruciate ligament stretch'):
        xMarket3year.at[i, 'injurytype_cruciateligament'] += 1
      elif (injury == 'Facial Injury' or injury == 'Nose surgery' or injury == 'Dental Surgery' or injury == 'Nasal Bone Fracture' or injury == 'Facial Fracture' or injury == 'Cheekbone Fracture' or injury == 'Eye Injury' or injury == 'Toothache' or injury == 'Frontal bone fracture' or injury == 'Fractured Jaw' or injury == 'Fractured Skull' or injury == 'Fracture of the orbit' or injury == 'Tooth Inflammation' or injury == 'Nose Injury'):
        xMarket3year.at[i, 'injurytype_face'] += 1
      elif (injury == 'Finger Injury'):
        xMarket3year.at[i, 'injurytype_finger'] += 1
      elif (injury == 'Fitness'):
        xMarket3year.at[i, 'injurytype_fitness'] += 1
      elif (injury == 'Foot Injury' or injury == 'Bruised Foot' or injury == 'Fractured Foot' or injury == 'Arch pain' or injury == 'Heel Injury' or injury == 'Hell pain' or injury == 'Heelspur' or injury == 'Heel Bone Injury' or injury == 'Metatarsal Fracture' or injury == 'Heel pain' or injury == 'Plantar fascia' or injury == 'Foot surgery' or injury == 'Hairline crack in the foot' or injury == 'Partial demolition of the plantar fascia'):
        xMarket3year.at[i, 'injurytype_foot'] += 1
      elif (injury == 'groin strain' or injury == 'Groin Strain' or injury == 'Groin Injury' or injury == 'Groin Surgery' or injury == 'Inguinal Hernia' or injury == 'Pubalgia' or injury == 'Testicular disruption'):
        xMarket3year.at[i, 'injurytype_groin'] += 1
      elif (injury == 'Hamstring Injury' or injury == 'Pulled hamstring at the adductors' or injury == 'Hamstring contusion'):
        xMarket3year.at[i, 'injurytype_hamstring'] += 1
      elif (injury == 'Hand Injury' or injury == 'Fractured Hand' or injury == 'Broken wrist' or injury == 'Wirst Injury' or injury == 'scaphoid operation' or injury == 'Hand fracture' or injury == 'Fractured Finger' or injury == 'Thumb Injury'):
        xMarket3year.at[i, 'injurytype_hand'] += 1
      elif (injury == 'Head Injury' or injury == 'Neck Injury' or injury == 'Neck bruise'):
        xMarket3year.at[i, 'injurytype_head'] += 1
      elif (injury == 'Heart Condition'):
        xMarket3year.at[i, 'injurytype_heart'] += 1
      elif (injury == 'Hip problems' or injury == 'Hip Injury' or injury == 'Problems with the hip flexor' or injury == 'Bruised Hip' or injury == 'Pubis bone contusion' or injury == 'Problems with the right hip flexor'):
        xMarket3year.at[i, 'injurytype_hip'] += 1
      elif (injury == 'Cold' or injury == 'Influenza' or injury == 'Flu' or injury == 'Abdominal Influenza' or injury == 'Ill' or injury == 'Corona virus' or injury == 'Infection' or injury == 'Virus Infection' or injury == 'Quarantine' or injury == 'Fever' or injury == 'Tonsillitis' or injury == 'Testicular Cancer' or injury == 'Pneumonia' or injury == 'Gastric problems' or injury == 'Mononucleosis' or injury == 'Chickenpox' or injury == 'Bronchitis' or injury == 'Stomach complaints' or injury == 'Food Poisoning' or injury == 'intestial virus' or injury == 'Infected wound' or injury == 'Malaria' or injury == 'cancer' or injury == 'Lymphoma' or injury == 'Depression' or injury == 'Cals Sclerosis' or injury == 'Kidney problems'):
        xMarket3year.at[i, 'injurytype_ill'] += 1
      elif (injury == 'Inflammation'):
        xMarket3year.at[i, 'injurytype_inflammation'] += 1
      elif (injury == 'Knee Injury' or injury == 'Knee Problems' or injury == 'Knee inflammation' or injury == 'Bruised Knee' or injury == 'Ruptured intraarticular ligament initiation in knee' or injury == 'Knee Surgery' or injury == 'Medial Collateral Ligament Tear' or injury == 'Medial Collateral Ligament Injury' or injury == 'Meniscal Injury' or injury == 'Sideband strain in the knee' or injury == 'Patella tendon irritation' or injury == 'Mensical Laceration' or injury == 'Twisted knee' or injury == 'Torn Knee Ligament' or injury == 'Medial Collateral Ligament Knee Injury' or injury == 'Patella problems' or injury == 'Ruptured lateral collateral ligament' or injury == 'Edema in the knee' or injury == 'Fractured Kneecap' or injury == 'Torn lateral collateral ligament' or injury == 'Torn Meniscus' or injury == 'Torn Collateral Ligament' or injury == 'Patella rupture' or injury == 'Torn knee ligament' or injury == 'Patella tendon luxation' or injury == 'Ruptured knee ligament' or injury == 'Rupture of Outer Meniscus' or injury == 'Double Torn Ligament' or injury == 'Meniscus Damage' or injury == 'Medial Collateral Ligament avulsion' or injury == 'Meniscus irritation' or injury == 'Inflamed ligaments of the knee' or injury == 'Rupture of the pattella'):
        xMarket3year.at[i, 'injurytype_knee'] += 1
      elif (injury == 'Knock' or injury == 'Minor Knock' or injury == 'Dead Leg' or injury == 'Stiffness'):
        xMarket3year.at[i, 'injurytype_knock'] += 1
      elif (injury == 'Leg Injury' or injury == 'Strain in the thigh and gluteal muscles' or injury == 'Shinbone injury' or injury == 'Shin bone bruise' or injury == 'Biceps femoris muscle injury' or injury == 'Fibula Fracture' or injury == 'Fractured Leg' or injury == 'Fissure of the fibula' or injury == 'Sciatic Problem' or injury == 'Tibia and Fibula Fracture' or injury == 'Fracture of the lower leg' or injury == 'Tibia Fracture' or injury == 'Inflamed head of fibula'):
        xMarket3year.at[i, 'injurytype_leg'] += 1
      elif (injury == 'Muscular problems' or injury == 'Muscle Injury' or injury == 'Torn Muscle Fibre' or injury == 'Muscle Fatigue' or injury == 'Torn muscle bundle' or injury == 'Torn Muscle' or injury == 'Muscle fiber tear'):
        xMarket3year.at[i, 'injurytype_muscular'] += 1
      elif (injury == 'Pubitis' or injury == 'Bruised pelvis' or injury == 'Pelvis Injury' or injury == 'Pubis bone irritation' or injury == 'bruised pelvis' or injury == 'bruised pelvis' or injury == 'Pelvic obliquity'):
        xMarket3year.at[i, 'injurytype_pelvis'] += 1
      elif (injury == 'Rest'):
        xMarket3year.at[i, 'injurytype_rest'] += 1
      elif (injury == 'Fractured Rib'):
        xMarket3year.at[i, 'injurytype_rib'] += 1
      elif (injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar vertebrae problems' or injury == 'Blockade in the spinal' or injury == 'Cervical spine injury' or injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar Vertebra Fracture' or injury == 'Vertebra injury'):
        xMarket3year.at[i, 'injurytype_spine'] += 1
      elif (injury == 'Thigh Muscle Strain' or injury == 'Thigh Problems' or injury == 'Thigh muscle rupture'):
        xMarket3year.at[i, 'injurytype_thigh'] += 1
      elif (injury == 'Toe Injury' or injury == 'Fractured Toe'):
        xMarket3year.at[i, 'injurytype_toe'] += 1
      elif (injury == 'Unknown Injury'):
        xMarket3year.at[i, 'injurytype_unknown'] += 1
      elif (injury == 'Surgery' or injury == 'Contracture' or injury == 'Sprain' or injury == 'Laceration' or injury == 'Fracture' or injury == 'Ligament Injury' or injury == 'Stress response of the bone' or injury == 'Strain' or injury == 'Contused laceration' or injury == 'Muscle partial avulsion' or injury == 'torn tendon' or injury == 'Tear in a joint capsule' or injury == 'tendon irritation' or injury == 'Ligament Problems' or injury == 'Sideband injury' or injury == 'Bone buckling' or injury == 'Stretched Ligament' or injury == 'Pinched nerve' or injury == 'Overstretching' or injury == 'Tendonitis' or injury == 'laceration' or injury == 'Arthroscopie' or injury == 'Herniated Disc' or injury == 'Fatigue fracture' or injury == 'Torn Ligament' or injury == 'Cartilage Damage' or injury == 'Bursitis' or injury == 'Circulation Problems' or injury == 'Tendon crack' or injury == 'Cut' or injury == 'sprain' or injury == 'Burns' or injury == 'open wound' or injury == 'Sideband tear' or injury == 'Marrow bulge' or injury == 'Compartment syndrome' or injury == 'Vestibular disorder' or injury == 'Intraarticular ligament fissure' or injury == 'Ruptured ligaments' or injury == 'Muscular hairline crack' or injury == 'capsular tear' or injury == 'Traffic Accident' or injury == 'Flesh Wound'):
        xMarket3year.at[i, 'injurytype_other'] += 1
      else:
        print(injury)

xMarket3year.insert(5, "GK", 0)
xMarket3year.insert(6, "DF", 0)
xMarket3year.insert(7, "MF", 0)
xMarket3year.insert(8, "FW", 0)

for i, row in xMarket3year.iterrows(): #len(xMarket3year)
  s = row['Pos']
  if type(s) == str:
    pos_list = s.split(",")
    for pos in pos_list:
      pos = pos.strip()
      if (pos == 'GK'):
        xMarket3year.at[i, 'GK'] += 1
      elif (pos == 'DF'):
        xMarket3year.at[i, 'DF'] += 1
      elif (pos == 'MF'):
        xMarket3year.at[i, 'MF'] += 1
      elif (pos == 'FW'):
        xMarket3year.at[i, 'FW'] += 1

transferValue = xMarket3year.pop("player_market_value_euro")
xMarket3year["player_market_value_euro"] = transferValue

#nan, lower body, upper body, ill, head, other
xMarket4year['injurytype_nan'] = 0 #nan

xMarket4year['injurytype_abdomen'] = 0 #Abdominal Strain

xMarket4year['injurytype_abductor'] = 0 #Tear in the abductor muscle,

xMarket4year['injurytype_achilles'] = 0 #Achilles tendon rupture, Achilles tendon problems,

xMarket4year['injurytype_adductor'] = 0 #Adductor problems,

xMarket4year['injurytype_ankle'] = 0 #Ankle Injury, Sprained ankle, Ankle problems, Torn ankle ligament, Ankle Surgery

xMarket4year['injurytype_arm'] = 0 #Arm Injury, Wirst Injury, Shoulder Injury

xMarket4year['injurytype_back'] = 0 #Back Injury, Back trouble, Back bruise

xMarket4year['injurytype_bruise'] = 0 #Bruise

xMarket4year['injurytype_calf'] = 0 #Calf Problems, Calf Injury, Calf Strain

xMarket4year['injurytype_chest'] = 0 #Angina,

xMarket4year['injurytype_collarbone'] = 0 #Collarbone fracture

xMarket4year['injurytype_concussion'] = 0 #Concussion

xMarket4year['injurytype_cruciateligament'] = 0 #Cruciate Ligament Rupture, Cruciate Ligament Surgery, Cruciate Ligament Strain

xMarket4year['injurytype_face'] = 0 #Facial Injury

xMarket4year['injurytype_finger'] = 0 #Finger Injury

xMarket4year['injurytype_fitness'] = 0 #Fitness

xMarket4year['injurytype_foot'] = 0 #Foot Injury, Bruised Foot, Fractured Foot,

xMarket4year['injurytype_groin'] = 0 #groin strain, Groin Strain (w/Capital), Groin Injury,

xMarket4year['injurytype_hamstring'] = 0 #Hamstring Injury

xMarket4year['injurytype_hand'] = 0 #Hand Injury, Hand Problems,

xMarket4year['injurytype_head'] = 0 #Head Injury

xMarket4year['injurytype_heart'] = 0 # Heart Condition

xMarket4year['injurytype_hip'] = 0 #Hip problems, Hip Injury,

xMarket4year['injurytype_ill'] = 0 #Cold, Influenza, Flu, Abdominal Influenza, Ill, Corona virus, Infection, Virus Infection, Quaratine, Testicular Cancer
#include abdominal influenza, cold, ill, covid, influenza, fever, quarantine, infection

xMarket4year['injurytype_inflammation'] = 0 #Inflammation

xMarket4year['injurytype_knee'] = 0 #Knee Injury, Knee Problems, Knee inflammation, Bruised Knee, Ruptured intraarticular ligament initiation in knee,

xMarket4year['injurytype_knock'] = 0 #Knock, Minor Knock,

xMarket4year['injurytype_leg'] = 0 #Leg Injury

xMarket4year['injurytype_muscular'] = 0 #Muscular problems, Muscle Injury, Biceps femoris muscle injury, Torn Muscle Fibre, Muscle Fatigue

xMarket4year['injurytype_pelvis'] = 0 #Pubitis, bruised pelvis, Pelvis Injury, Pubis bone irritation

xMarket4year['injurytype_rest'] = 0 #Rest

xMarket4year['injurytype_rib'] = 0 #Fractured Rib

xMarket4year['injurytype_spine'] = 0 #Lumbar Vertebra Fracture, Lumbar vertebrae problems, Blockade in the spinal, Cervical spine injury

xMarket4year['injurytype_thigh'] = 0 #Thigh Muscle Strain, Thigh Problems,

xMarket4year['injurytype_toe'] = 0 #Toe Injury

xMarket4year['injurytype_unknown'] = 0 #Unknown Injury,

xMarket4year['injurytype_other'] = 0 #Surgery, Contracture, Sprain, Laceration

for i, row in xMarket4year.iterrows(): #len(x1year)
  s = row['injuryType']
  if (s == 'nan'):
    xMarket4year.at[i, 'injurytype_nan'] += 1
  elif type(s) == str:
    injury_list = s.split(" ,")
    injury_list = injury_list[:-1]
    for injury in injury_list:
      injury = injury.strip()
      if (injury == 'Abdominal Strain' or injury == 'Abdominal muscles injury' or injury == 'Appendectomy' or injury == 'Umbilical hernia'):
        xMarket4year.at[i, 'injurytype_abdomen'] += 1
      elif (injury == 'Tear in the abductor muscle'):
        xMarket4year.at[i, 'injurytype_abductor'] += 1
      elif (injury == 'Achilles tendon rupture' or injury == 'Achilles tendon problems' or injury == 'Achilles Irritation' or injury == 'Achilles tendon surgery'):
        xMarket4year.at[i, 'injurytype_achilles'] += 1
      elif (injury == 'Adductor problems'):
        xMarket4year.at[i, 'injurytype_adductor'] += 1
      elif (injury == 'Ankle Injury' or injury == 'Sprained ankle' or injury == 'Ankle problems' or injury == 'Torn ankle ligament' or injury == 'Ankle Surgery' or injury == 'Bruised Ankle' or injury == 'Distortion of the ankle' or injury == 'Ruptured syndesmotic ligament' or injury == 'Torn Ankle Ligament' or injury == 'Fracture-dislocation of the ankle' or injury == 'Sprained Ankle' or injury == 'Injury to the ankle' or injury == 'Ankle fracture' or injury == 'Ankle Inflammation' or injury == 'Syndesmotic ligament tear' or injury == 'Ankle Fracture' or injury == 'Capsular rupture in the ankle' or injury == 'Ruptured ankle ligament' or injury == 'Ruptured ankle aigament' or injury == 'Bruise on ankle' or injury == 'Ruptured intraarticular ligament initiation in the ankle' or injury == 'Peroneus tendon injury'):
        xMarket4year.at[i, 'injurytype_ankle'] += 1
      elif (injury == 'Arm Injury' or injury == 'Shoulder Injury' or injury == 'Shoulder fracture' or injury == 'Elbow Injury' or injury == 'Bruised Acromioclavicular' or injury == 'Fractured Arm' or injury == 'Acromioclavicular Separation'):
        xMarket4year.at[i, 'injurytype_arm'] += 1
      elif (injury == 'Back Injury' or injury == 'Back trouble' or injury == 'Back bruise' or injury == 'Lumbago'):
        xMarket4year.at[i, 'injurytype_back'] += 1
      elif (injury == 'Bruise' or injury == 'Bruised Rib' or injury == 'Metatarsal bone bruise' or injury == 'Bone Bruise' or injury == 'Muscle bruise'):
        xMarket4year.at[i, 'injurytype_bruise'] += 1
      elif (injury == 'Calf Problems' or injury == 'Calf Injury' or injury == 'Calf Strain' or injury == 'Disrupted Calf Muscle' or injury == 'Hairline crack in calfbone' or injury == 'Calf muscle strain'):
        xMarket4year.at[i, 'injurytype_calf'] += 1
      elif (injury == 'Angina' or injury == 'Chest injury' or injury == 'Pneumothorax' or injury == 'Lung contusion'):
        xMarket4year.at[i, 'injurytype_chest'] += 1
      elif (injury == 'Collarbone fracture'):
        xMarket4year.at[i, 'injurytype_collarbone'] += 1
      elif (injury == 'Concussion'):
        xMarket4year.at[i, 'injurytype_concussion'] += 1
      elif (injury == 'Cruciate Ligament Rupture' or injury == 'Cruciate Ligament Surgery' or injury == 'Cruciate Ligament Strain' or injury == 'Cruciate Ligament Injury' or injury == 'Ruptured cruciate ligament' or injury == 'Partial damage to the cruciate ligament' or injury == 'Cruciate ligament stretch'):
        xMarket4year.at[i, 'injurytype_cruciateligament'] += 1
      elif (injury == 'Facial Injury' or injury == 'Nose surgery' or injury == 'Dental Surgery' or injury == 'Nasal Bone Fracture' or injury == 'Facial Fracture' or injury == 'Cheekbone Fracture' or injury == 'Eye Injury' or injury == 'Toothache' or injury == 'Frontal bone fracture' or injury == 'Fractured Jaw' or injury == 'Fractured Skull' or injury == 'Fracture of the orbit' or injury == 'Tooth Inflammation' or injury == 'Nose Injury'):
        xMarket4year.at[i, 'injurytype_face'] += 1
      elif (injury == 'Finger Injury'):
        xMarket4year.at[i, 'injurytype_finger'] += 1
      elif (injury == 'Fitness'):
        xMarket4year.at[i, 'injurytype_fitness'] += 1
      elif (injury == 'Foot Injury' or injury == 'Bruised Foot' or injury == 'Fractured Foot' or injury == 'Arch pain' or injury == 'Heel Injury' or injury == 'Hell pain' or injury == 'Heelspur' or injury == 'Heel Bone Injury' or injury == 'Metatarsal Fracture' or injury == 'Heel pain' or injury == 'Plantar fascia' or injury == 'Foot surgery' or injury == 'Hairline crack in the foot' or injury == 'Partial demolition of the plantar fascia'):
        xMarket4year.at[i, 'injurytype_foot'] += 1
      elif (injury == 'groin strain' or injury == 'Groin Strain' or injury == 'Groin Injury' or injury == 'Groin Surgery' or injury == 'Inguinal Hernia' or injury == 'Pubalgia' or injury == 'Testicular disruption'):
        xMarket4year.at[i, 'injurytype_groin'] += 1
      elif (injury == 'Hamstring Injury' or injury == 'Pulled hamstring at the adductors' or injury == 'Hamstring contusion'):
        xMarket4year.at[i, 'injurytype_hamstring'] += 1
      elif (injury == 'Hand Injury' or injury == 'Fractured Hand' or injury == 'Broken wrist' or injury == 'Wirst Injury' or injury == 'scaphoid operation' or injury == 'Hand fracture' or injury == 'Fractured Finger' or injury == 'Thumb Injury'):
        xMarket4year.at[i, 'injurytype_hand'] += 1
      elif (injury == 'Head Injury' or injury == 'Neck Injury' or injury == 'Neck bruise'):
        xMarket4year.at[i, 'injurytype_head'] += 1
      elif (injury == 'Heart Condition'):
        xMarket4year.at[i, 'injurytype_heart'] += 1
      elif (injury == 'Hip problems' or injury == 'Hip Injury' or injury == 'Problems with the hip flexor' or injury == 'Bruised Hip' or injury == 'Pubis bone contusion' or injury == 'Problems with the right hip flexor'):
        xMarket4year.at[i, 'injurytype_hip'] += 1
      elif (injury == 'Cold' or injury == 'Influenza' or injury == 'Flu' or injury == 'Abdominal Influenza' or injury == 'Ill' or injury == 'Corona virus' or injury == 'Infection' or injury == 'Virus Infection' or injury == 'Quarantine' or injury == 'Fever' or injury == 'Tonsillitis' or injury == 'Testicular Cancer' or injury == 'Pneumonia' or injury == 'Gastric problems' or injury == 'Mononucleosis' or injury == 'Chickenpox' or injury == 'Bronchitis' or injury == 'Stomach complaints' or injury == 'Food Poisoning' or injury == 'intestial virus' or injury == 'Infected wound' or injury == 'Malaria' or injury == 'cancer' or injury == 'Lymphoma' or injury == 'Depression' or injury == 'Cals Sclerosis' or injury == 'Kidney problems'):
        xMarket4year.at[i, 'injurytype_ill'] += 1
      elif (injury == 'Inflammation'):
        xMarket4year.at[i, 'injurytype_inflammation'] += 1
      elif (injury == 'Knee Injury' or injury == 'Knee Problems' or injury == 'Knee inflammation' or injury == 'Bruised Knee' or injury == 'Ruptured intraarticular ligament initiation in knee' or injury == 'Knee Surgery' or injury == 'Medial Collateral Ligament Tear' or injury == 'Medial Collateral Ligament Injury' or injury == 'Meniscal Injury' or injury == 'Sideband strain in the knee' or injury == 'Patella tendon irritation' or injury == 'Mensical Laceration' or injury == 'Twisted knee' or injury == 'Torn Knee Ligament' or injury == 'Medial Collateral Ligament Knee Injury' or injury == 'Patella problems' or injury == 'Ruptured lateral collateral ligament' or injury == 'Edema in the knee' or injury == 'Fractured Kneecap' or injury == 'Torn lateral collateral ligament' or injury == 'Torn Meniscus' or injury == 'Torn Collateral Ligament' or injury == 'Patella rupture' or injury == 'Torn knee ligament' or injury == 'Patella tendon luxation' or injury == 'Ruptured knee ligament' or injury == 'Rupture of Outer Meniscus' or injury == 'Double Torn Ligament' or injury == 'Meniscus Damage' or injury == 'Medial Collateral Ligament avulsion' or injury == 'Meniscus irritation' or injury == 'Inflamed ligaments of the knee' or injury == 'Rupture of the pattella'):
        xMarket4year.at[i, 'injurytype_knee'] += 1
      elif (injury == 'Knock' or injury == 'Minor Knock' or injury == 'Dead Leg' or injury == 'Stiffness'):
        xMarket4year.at[i, 'injurytype_knock'] += 1
      elif (injury == 'Leg Injury' or injury == 'Strain in the thigh and gluteal muscles' or injury == 'Shinbone injury' or injury == 'Shin bone bruise' or injury == 'Biceps femoris muscle injury' or injury == 'Fibula Fracture' or injury == 'Fractured Leg' or injury == 'Fissure of the fibula' or injury == 'Sciatic Problem' or injury == 'Tibia and Fibula Fracture' or injury == 'Fracture of the lower leg' or injury == 'Tibia Fracture' or injury == 'Inflamed head of fibula'):
        xMarket4year.at[i, 'injurytype_leg'] += 1
      elif (injury == 'Muscular problems' or injury == 'Muscle Injury' or injury == 'Torn Muscle Fibre' or injury == 'Muscle Fatigue' or injury == 'Torn muscle bundle' or injury == 'Torn Muscle' or injury == 'Muscle fiber tear'):
        xMarket4year.at[i, 'injurytype_muscular'] += 1
      elif (injury == 'Pubitis' or injury == 'Bruised pelvis' or injury == 'Pelvis Injury' or injury == 'Pubis bone irritation' or injury == 'bruised pelvis' or injury == 'bruised pelvis' or injury == 'Pelvic obliquity'):
        xMarket4year.at[i, 'injurytype_pelvis'] += 1
      elif (injury == 'Rest'):
        xMarket4year.at[i, 'injurytype_rest'] += 1
      elif (injury == 'Fractured Rib'):
        xMarket4year.at[i, 'injurytype_rib'] += 1
      elif (injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar vertebrae problems' or injury == 'Blockade in the spinal' or injury == 'Cervical spine injury' or injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar Vertebra Fracture' or injury == 'Vertebra injury'):
        xMarket4year.at[i, 'injurytype_spine'] += 1
      elif (injury == 'Thigh Muscle Strain' or injury == 'Thigh Problems' or injury == 'Thigh muscle rupture'):
        xMarket4year.at[i, 'injurytype_thigh'] += 1
      elif (injury == 'Toe Injury' or injury == 'Fractured Toe'):
        xMarket4year.at[i, 'injurytype_toe'] += 1
      elif (injury == 'Unknown Injury'):
        xMarket4year.at[i, 'injurytype_unknown'] += 1
      elif (injury == 'Surgery' or injury == 'Contracture' or injury == 'Sprain' or injury == 'Laceration' or injury == 'Fracture' or injury == 'Ligament Injury' or injury == 'Stress response of the bone' or injury == 'Strain' or injury == 'Contused laceration' or injury == 'Muscle partial avulsion' or injury == 'torn tendon' or injury == 'Tear in a joint capsule' or injury == 'tendon irritation' or injury == 'Ligament Problems' or injury == 'Sideband injury' or injury == 'Bone buckling' or injury == 'Stretched Ligament' or injury == 'Pinched nerve' or injury == 'Overstretching' or injury == 'Tendonitis' or injury == 'laceration' or injury == 'Arthroscopie' or injury == 'Herniated Disc' or injury == 'Fatigue fracture' or injury == 'Torn Ligament' or injury == 'Cartilage Damage' or injury == 'Bursitis' or injury == 'Circulation Problems' or injury == 'Tendon crack' or injury == 'Cut' or injury == 'sprain' or injury == 'Burns' or injury == 'open wound' or injury == 'Sideband tear' or injury == 'Marrow bulge' or injury == 'Compartment syndrome' or injury == 'Vestibular disorder' or injury == 'Intraarticular ligament fissure' or injury == 'Ruptured ligaments' or injury == 'Muscular hairline crack' or injury == 'capsular tear' or injury == 'Traffic Accident' or injury == 'Flesh Wound'):
        xMarket4year.at[i, 'injurytype_other'] += 1
      else:
        print(injury)

xMarket4year.insert(5, "GK", 0)
xMarket4year.insert(6, "DF", 0)
xMarket4year.insert(7, "MF", 0)
xMarket4year.insert(8, "FW", 0)

for i, row in xMarket4year.iterrows(): #len(xMarket4year)
  s = row['Pos']
  if type(s) == str:
    pos_list = s.split(",")
    for pos in pos_list:
      pos = pos.strip()
      if (pos == 'GK'):
        xMarket4year.at[i, 'GK'] += 1
      elif (pos == 'DF'):
        xMarket4year.at[i, 'DF'] += 1
      elif (pos == 'MF'):
        xMarket4year.at[i, 'MF'] += 1
      elif (pos == 'FW'):
        xMarket4year.at[i, 'FW'] += 1

transferValue = xMarket4year.pop("player_market_value_euro")
xMarket4year["player_market_value_euro"] = transferValue

#nan, lower body, upper body, ill, head, other
xMarket5year['injurytype_nan'] = 0 #nan

xMarket5year['injurytype_abdomen'] = 0 #Abdominal Strain

xMarket5year['injurytype_abductor'] = 0 #Tear in the abductor muscle,

xMarket5year['injurytype_achilles'] = 0 #Achilles tendon rupture, Achilles tendon problems,

xMarket5year['injurytype_adductor'] = 0 #Adductor problems,

xMarket5year['injurytype_ankle'] = 0 #Ankle Injury, Sprained ankle, Ankle problems, Torn ankle ligament, Ankle Surgery

xMarket5year['injurytype_arm'] = 0 #Arm Injury, Wirst Injury, Shoulder Injury

xMarket5year['injurytype_back'] = 0 #Back Injury, Back trouble, Back bruise

xMarket5year['injurytype_bruise'] = 0 #Bruise

xMarket5year['injurytype_calf'] = 0 #Calf Problems, Calf Injury, Calf Strain

xMarket5year['injurytype_chest'] = 0 #Angina,

xMarket5year['injurytype_collarbone'] = 0 #Collarbone fracture

xMarket5year['injurytype_concussion'] = 0 #Concussion

xMarket5year['injurytype_cruciateligament'] = 0 #Cruciate Ligament Rupture, Cruciate Ligament Surgery, Cruciate Ligament Strain

xMarket5year['injurytype_face'] = 0 #Facial Injury

xMarket5year['injurytype_finger'] = 0 #Finger Injury

xMarket5year['injurytype_fitness'] = 0 #Fitness

xMarket5year['injurytype_foot'] = 0 #Foot Injury, Bruised Foot, Fractured Foot,

xMarket5year['injurytype_groin'] = 0 #groin strain, Groin Strain (w/Capital), Groin Injury,

xMarket5year['injurytype_hamstring'] = 0 #Hamstring Injury

xMarket5year['injurytype_hand'] = 0 #Hand Injury, Hand Problems,

xMarket5year['injurytype_head'] = 0 #Head Injury

xMarket5year['injurytype_heart'] = 0 # Heart Condition

xMarket5year['injurytype_hip'] = 0 #Hip problems, Hip Injury,

xMarket5year['injurytype_ill'] = 0 #Cold, Influenza, Flu, Abdominal Influenza, Ill, Corona virus, Infection, Virus Infection, Quaratine, Testicular Cancer
#include abdominal influenza, cold, ill, covid, influenza, fever, quarantine, infection

xMarket5year['injurytype_inflammation'] = 0 #Inflammation

xMarket5year['injurytype_knee'] = 0 #Knee Injury, Knee Problems, Knee inflammation, Bruised Knee, Ruptured intraarticular ligament initiation in knee,

xMarket5year['injurytype_knock'] = 0 #Knock, Minor Knock,

xMarket5year['injurytype_leg'] = 0 #Leg Injury

xMarket5year['injurytype_muscular'] = 0 #Muscular problems, Muscle Injury, Biceps femoris muscle injury, Torn Muscle Fibre, Muscle Fatigue

xMarket5year['injurytype_pelvis'] = 0 #Pubitis, bruised pelvis, Pelvis Injury, Pubis bone irritation

xMarket5year['injurytype_rest'] = 0 #Rest

xMarket5year['injurytype_rib'] = 0 #Fractured Rib

xMarket5year['injurytype_spine'] = 0 #Lumbar Vertebra Fracture, Lumbar vertebrae problems, Blockade in the spinal, Cervical spine injury

xMarket5year['injurytype_thigh'] = 0 #Thigh Muscle Strain, Thigh Problems,

xMarket5year['injurytype_toe'] = 0 #Toe Injury

xMarket5year['injurytype_unknown'] = 0 #Unknown Injury,

xMarket5year['injurytype_other'] = 0 #Surgery, Contracture, Sprain, Laceration

for i, row in xMarket5year.iterrows(): #len(x1year)
  s = row['injuryType']
  if (s == 'nan'):
    xMarket5year.at[i, 'injurytype_nan'] += 1
  elif type(s) == str:
    injury_list = s.split(" ,")
    injury_list = injury_list[:-1]
    for injury in injury_list:
      injury = injury.strip()
      if (injury == 'Abdominal Strain' or injury == 'Abdominal muscles injury' or injury == 'Appendectomy' or injury == 'Umbilical hernia'):
        xMarket5year.at[i, 'injurytype_abdomen'] += 1
      elif (injury == 'Tear in the abductor muscle'):
        xMarket5year.at[i, 'injurytype_abductor'] += 1
      elif (injury == 'Achilles tendon rupture' or injury == 'Achilles tendon problems' or injury == 'Achilles Irritation' or injury == 'Achilles tendon surgery'):
        xMarket5year.at[i, 'injurytype_achilles'] += 1
      elif (injury == 'Adductor problems'):
        xMarket5year.at[i, 'injurytype_adductor'] += 1
      elif (injury == 'Ankle Injury' or injury == 'Sprained ankle' or injury == 'Ankle problems' or injury == 'Torn ankle ligament' or injury == 'Ankle Surgery' or injury == 'Bruised Ankle' or injury == 'Distortion of the ankle' or injury == 'Ruptured syndesmotic ligament' or injury == 'Torn Ankle Ligament' or injury == 'Fracture-dislocation of the ankle' or injury == 'Sprained Ankle' or injury == 'Injury to the ankle' or injury == 'Ankle fracture' or injury == 'Ankle Inflammation' or injury == 'Syndesmotic ligament tear' or injury == 'Ankle Fracture' or injury == 'Capsular rupture in the ankle' or injury == 'Ruptured ankle ligament' or injury == 'Ruptured ankle aigament' or injury == 'Bruise on ankle' or injury == 'Ruptured intraarticular ligament initiation in the ankle' or injury == 'Peroneus tendon injury'):
        xMarket5year.at[i, 'injurytype_ankle'] += 1
      elif (injury == 'Arm Injury' or injury == 'Shoulder Injury' or injury == 'Shoulder fracture' or injury == 'Elbow Injury' or injury == 'Bruised Acromioclavicular' or injury == 'Fractured Arm' or injury == 'Acromioclavicular Separation'):
        xMarket5year.at[i, 'injurytype_arm'] += 1
      elif (injury == 'Back Injury' or injury == 'Back trouble' or injury == 'Back bruise' or injury == 'Lumbago'):
        xMarket5year.at[i, 'injurytype_back'] += 1
      elif (injury == 'Bruise' or injury == 'Bruised Rib' or injury == 'Metatarsal bone bruise' or injury == 'Bone Bruise' or injury == 'Muscle bruise'):
        xMarket5year.at[i, 'injurytype_bruise'] += 1
      elif (injury == 'Calf Problems' or injury == 'Calf Injury' or injury == 'Calf Strain' or injury == 'Disrupted Calf Muscle' or injury == 'Hairline crack in calfbone' or injury == 'Calf muscle strain'):
        xMarket5year.at[i, 'injurytype_calf'] += 1
      elif (injury == 'Angina' or injury == 'Chest injury' or injury == 'Pneumothorax' or injury == 'Lung contusion'):
        xMarket5year.at[i, 'injurytype_chest'] += 1
      elif (injury == 'Collarbone fracture'):
        xMarket5year.at[i, 'injurytype_collarbone'] += 1
      elif (injury == 'Concussion'):
        xMarket5year.at[i, 'injurytype_concussion'] += 1
      elif (injury == 'Cruciate Ligament Rupture' or injury == 'Cruciate Ligament Surgery' or injury == 'Cruciate Ligament Strain' or injury == 'Cruciate Ligament Injury' or injury == 'Ruptured cruciate ligament' or injury == 'Partial damage to the cruciate ligament' or injury == 'Cruciate ligament stretch'):
        xMarket5year.at[i, 'injurytype_cruciateligament'] += 1
      elif (injury == 'Facial Injury' or injury == 'Nose surgery' or injury == 'Dental Surgery' or injury == 'Nasal Bone Fracture' or injury == 'Facial Fracture' or injury == 'Cheekbone Fracture' or injury == 'Eye Injury' or injury == 'Toothache' or injury == 'Frontal bone fracture' or injury == 'Fractured Jaw' or injury == 'Fractured Skull' or injury == 'Fracture of the orbit' or injury == 'Tooth Inflammation' or injury == 'Nose Injury'):
        xMarket5year.at[i, 'injurytype_face'] += 1
      elif (injury == 'Finger Injury'):
        xMarket5year.at[i, 'injurytype_finger'] += 1
      elif (injury == 'Fitness'):
        xMarket5year.at[i, 'injurytype_fitness'] += 1
      elif (injury == 'Foot Injury' or injury == 'Bruised Foot' or injury == 'Fractured Foot' or injury == 'Arch pain' or injury == 'Heel Injury' or injury == 'Hell pain' or injury == 'Heelspur' or injury == 'Heel Bone Injury' or injury == 'Metatarsal Fracture' or injury == 'Heel pain' or injury == 'Plantar fascia' or injury == 'Foot surgery' or injury == 'Hairline crack in the foot' or injury == 'Partial demolition of the plantar fascia'):
        xMarket5year.at[i, 'injurytype_foot'] += 1
      elif (injury == 'groin strain' or injury == 'Groin Strain' or injury == 'Groin Injury' or injury == 'Groin Surgery' or injury == 'Inguinal Hernia' or injury == 'Pubalgia' or injury == 'Testicular disruption'):
        xMarket5year.at[i, 'injurytype_groin'] += 1
      elif (injury == 'Hamstring Injury' or injury == 'Pulled hamstring at the adductors' or injury == 'Hamstring contusion'):
        xMarket5year.at[i, 'injurytype_hamstring'] += 1
      elif (injury == 'Hand Injury' or injury == 'Fractured Hand' or injury == 'Broken wrist' or injury == 'Wirst Injury' or injury == 'scaphoid operation' or injury == 'Hand fracture' or injury == 'Fractured Finger' or injury == 'Thumb Injury'):
        xMarket5year.at[i, 'injurytype_hand'] += 1
      elif (injury == 'Head Injury' or injury == 'Neck Injury' or injury == 'Neck bruise'):
        xMarket5year.at[i, 'injurytype_head'] += 1
      elif (injury == 'Heart Condition'):
        xMarket5year.at[i, 'injurytype_heart'] += 1
      elif (injury == 'Hip problems' or injury == 'Hip Injury' or injury == 'Problems with the hip flexor' or injury == 'Bruised Hip' or injury == 'Pubis bone contusion' or injury == 'Problems with the right hip flexor'):
        xMarket5year.at[i, 'injurytype_hip'] += 1
      elif (injury == 'Cold' or injury == 'Influenza' or injury == 'Flu' or injury == 'Abdominal Influenza' or injury == 'Ill' or injury == 'Corona virus' or injury == 'Infection' or injury == 'Virus Infection' or injury == 'Quarantine' or injury == 'Fever' or injury == 'Tonsillitis' or injury == 'Testicular Cancer' or injury == 'Pneumonia' or injury == 'Gastric problems' or injury == 'Mononucleosis' or injury == 'Chickenpox' or injury == 'Bronchitis' or injury == 'Stomach complaints' or injury == 'Food Poisoning' or injury == 'intestial virus' or injury == 'Infected wound' or injury == 'Malaria' or injury == 'cancer' or injury == 'Lymphoma' or injury == 'Depression' or injury == 'Cals Sclerosis' or injury == 'Kidney problems'):
        xMarket5year.at[i, 'injurytype_ill'] += 1
      elif (injury == 'Inflammation'):
        xMarket5year.at[i, 'injurytype_inflammation'] += 1
      elif (injury == 'Knee Injury' or injury == 'Knee Problems' or injury == 'Knee inflammation' or injury == 'Bruised Knee' or injury == 'Ruptured intraarticular ligament initiation in knee' or injury == 'Knee Surgery' or injury == 'Medial Collateral Ligament Tear' or injury == 'Medial Collateral Ligament Injury' or injury == 'Meniscal Injury' or injury == 'Sideband strain in the knee' or injury == 'Patella tendon irritation' or injury == 'Mensical Laceration' or injury == 'Twisted knee' or injury == 'Torn Knee Ligament' or injury == 'Medial Collateral Ligament Knee Injury' or injury == 'Patella problems' or injury == 'Ruptured lateral collateral ligament' or injury == 'Edema in the knee' or injury == 'Fractured Kneecap' or injury == 'Torn lateral collateral ligament' or injury == 'Torn Meniscus' or injury == 'Torn Collateral Ligament' or injury == 'Patella rupture' or injury == 'Torn knee ligament' or injury == 'Patella tendon luxation' or injury == 'Ruptured knee ligament' or injury == 'Rupture of Outer Meniscus' or injury == 'Double Torn Ligament' or injury == 'Meniscus Damage' or injury == 'Medial Collateral Ligament avulsion' or injury == 'Meniscus irritation' or injury == 'Inflamed ligaments of the knee' or injury == 'Rupture of the pattella'):
        xMarket5year.at[i, 'injurytype_knee'] += 1
      elif (injury == 'Knock' or injury == 'Minor Knock' or injury == 'Dead Leg' or injury == 'Stiffness'):
        xMarket5year.at[i, 'injurytype_knock'] += 1
      elif (injury == 'Leg Injury' or injury == 'Strain in the thigh and gluteal muscles' or injury == 'Shinbone injury' or injury == 'Shin bone bruise' or injury == 'Biceps femoris muscle injury' or injury == 'Fibula Fracture' or injury == 'Fractured Leg' or injury == 'Fissure of the fibula' or injury == 'Sciatic Problem' or injury == 'Tibia and Fibula Fracture' or injury == 'Fracture of the lower leg' or injury == 'Tibia Fracture' or injury == 'Inflamed head of fibula'):
        xMarket5year.at[i, 'injurytype_leg'] += 1
      elif (injury == 'Muscular problems' or injury == 'Muscle Injury' or injury == 'Torn Muscle Fibre' or injury == 'Muscle Fatigue' or injury == 'Torn muscle bundle' or injury == 'Torn Muscle' or injury == 'Muscle fiber tear'):
        xMarket5year.at[i, 'injurytype_muscular'] += 1
      elif (injury == 'Pubitis' or injury == 'Bruised pelvis' or injury == 'Pelvis Injury' or injury == 'Pubis bone irritation' or injury == 'bruised pelvis' or injury == 'bruised pelvis' or injury == 'Pelvic obliquity'):
        xMarket5year.at[i, 'injurytype_pelvis'] += 1
      elif (injury == 'Rest'):
        xMarket5year.at[i, 'injurytype_rest'] += 1
      elif (injury == 'Fractured Rib'):
        xMarket5year.at[i, 'injurytype_rib'] += 1
      elif (injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar vertebrae problems' or injury == 'Blockade in the spinal' or injury == 'Cervical spine injury' or injury == 'Lumbar Vertebra Fracture' or injury == 'Lumbar Vertebra Fracture' or injury == 'Vertebra injury'):
        xMarket5year.at[i, 'injurytype_spine'] += 1
      elif (injury == 'Thigh Muscle Strain' or injury == 'Thigh Problems' or injury == 'Thigh muscle rupture'):
        xMarket5year.at[i, 'injurytype_thigh'] += 1
      elif (injury == 'Toe Injury' or injury == 'Fractured Toe'):
        xMarket5year.at[i, 'injurytype_toe'] += 1
      elif (injury == 'Unknown Injury'):
        xMarket5year.at[i, 'injurytype_unknown'] += 1
      elif (injury == 'Surgery' or injury == 'Contracture' or injury == 'Sprain' or injury == 'Laceration' or injury == 'Fracture' or injury == 'Ligament Injury' or injury == 'Stress response of the bone' or injury == 'Strain' or injury == 'Contused laceration' or injury == 'Muscle partial avulsion' or injury == 'torn tendon' or injury == 'Tear in a joint capsule' or injury == 'tendon irritation' or injury == 'Ligament Problems' or injury == 'Sideband injury' or injury == 'Bone buckling' or injury == 'Stretched Ligament' or injury == 'Pinched nerve' or injury == 'Overstretching' or injury == 'Tendonitis' or injury == 'laceration' or injury == 'Arthroscopie' or injury == 'Herniated Disc' or injury == 'Fatigue fracture' or injury == 'Torn Ligament' or injury == 'Cartilage Damage' or injury == 'Bursitis' or injury == 'Circulation Problems' or injury == 'Tendon crack' or injury == 'Cut' or injury == 'sprain' or injury == 'Burns' or injury == 'open wound' or injury == 'Sideband tear' or injury == 'Marrow bulge' or injury == 'Compartment syndrome' or injury == 'Vestibular disorder' or injury == 'Intraarticular ligament fissure' or injury == 'Ruptured ligaments' or injury == 'Muscular hairline crack' or injury == 'capsular tear' or injury == 'Traffic Accident' or injury == 'Flesh Wound'):
        xMarket5year.at[i, 'injurytype_other'] += 1
      else:
        print(injury)

xMarket5year.insert(5, "GK", 0)
xMarket5year.insert(6, "DF", 0)
xMarket5year.insert(7, "MF", 0)
xMarket5year.insert(8, "FW", 0)

for i, row in xMarket5year.iterrows(): #len(xMarket5year)
  s = row['Pos']
  if type(s) == str:
    pos_list = s.split(",")
    for pos in pos_list:
      pos = pos.strip()
      if (pos == 'GK'):
        xMarket5year.at[i, 'GK'] += 1
      elif (pos == 'DF'):
        xMarket5year.at[i, 'DF'] += 1
      elif (pos == 'MF'):
        xMarket5year.at[i, 'MF'] += 1
      elif (pos == 'FW'):
        xMarket5year.at[i, 'FW'] += 1

transferValue = xMarket5year.pop("player_market_value_euro")
xMarket5year["player_market_value_euro"] = transferValue

xMarket1year.rename(columns = {"Player_x": "Player"}, inplace = True)
xMarket2year.rename(columns = {"Player_x": "Player"}, inplace = True)
xMarket3year.rename(columns = {"Player_x": "Player"}, inplace = True)
xMarket4year.rename(columns = {"Player_x": "Player"}, inplace = True)
xMarket5year.rename(columns = {"Player_x": "Player"}, inplace = True)
xMarket1year.drop(columns = ["injuryType", "player_age", "Url"], inplace = True)
xMarket2year.drop(columns = ["injuryType", "player_age", "Url"], inplace = True)
xMarket3year.drop(columns = ["injuryType", "player_age", "Url"], inplace = True)
xMarket4year.drop(columns = ["injuryType", "player_age", "Url"], inplace = True)
xMarket5year.drop(columns = ["injuryType", "player_age", "Url"], inplace = True)

xMarket1year.drop(columns = ["Pos"], inplace = True)
xMarket2year.drop(columns = ["Pos"], inplace = True)
xMarket3year.drop(columns = ["Pos"], inplace = True)
xMarket4year.drop(columns = ["Pos"], inplace = True)
xMarket5year.drop(columns = ["Pos"], inplace = True)

xMarket1year.to_csv('/content/drive/MyDrive/PlayerValue/Market1year.csv')
xMarket2year.to_csv('/content/drive/MyDrive/PlayerValue/Market2year.csv')
xMarket3year.to_csv('/content/drive/MyDrive/PlayerValue/Market3year.csv')
xMarket4year.to_csv('/content/drive/MyDrive/PlayerValue/Market4year.csv')
xMarket5year.to_csv('/content/drive/MyDrive/PlayerValue/Market5year.csv')

MarketModelTwoYeardf.drop(columns = ['Posyear2', 'Urlyear2', 'Player_urlyear2', 'regionyear2', 'squadyear2', 'player_numyear2', 'player_nameyear2', 'player_positionyear2', 'player_dobyear2', 'player_nationalityyear2', 'current_clubyear2', 'date_joinedyear2', 'joined_fromyear2', 'contract_expiryyear2', 'Player_yyear2', 'Type_2019year2', 'Type_2020year2', 'Type_2021year2', 'Type_2022year2', 'comp_nameyear2'], axis=1, inplace=True)

MarketModelThreeYeardf.drop(columns = ['Posyear2', 'Urlyear2', 'Player_urlyear2', 'regionyear2', 'squadyear2', 'player_numyear2', 'player_nameyear2', 'player_positionyear2', 'player_dobyear2', 'player_nationalityyear2', 'current_clubyear2', 'date_joinedyear2', 'joined_fromyear2', 'contract_expiryyear2', 'Player_yyear2', 'Type_2019year2', 'Type_2020year2', 'Type_2021year2', 'Type_2022year2', 'comp_nameyear2',
                                       'Posyear3', 'Urlyear3', 'Player_urlyear3', 'regionyear3', 'squadyear3', 'player_numyear3', 'player_nameyear3', 'player_positionyear3', 'player_dobyear3', 'player_nationalityyear3', 'current_clubyear3', 'date_joinedyear3', 'joined_fromyear3', 'contract_expiryyear3', 'Player_yyear3', 'Type_2019year3', 'Type_2020year3', 'Type_2021year3', 'Type_2022year3', 'comp_nameyear3'], axis=1, inplace=True)

MarketModelFourYeardf.drop(columns = ['Posyear2', 'Urlyear2', 'Player_urlyear2', 'regionyear2', 'squadyear2', 'player_numyear2', 'player_nameyear2', 'player_positionyear2', 'player_dobyear2', 'player_nationalityyear2', 'current_clubyear2', 'date_joinedyear2', 'joined_fromyear2', 'contract_expiryyear2', 'Player_yyear2', 'Type_2019year2', 'Type_2020year2', 'Type_2021year2', 'Type_2022year2', 'comp_nameyear2',
                                       'Posyear3', 'Urlyear3', 'Player_urlyear3', 'regionyear3', 'squadyear3', 'player_numyear3', 'player_nameyear3', 'player_positionyear3', 'player_dobyear3', 'player_nationalityyear3', 'current_clubyear3', 'date_joinedyear3', 'joined_fromyear3', 'contract_expiryyear3', 'Player_yyear3', 'Type_2019year3', 'Type_2020year3', 'Type_2021year3', 'Type_2022year3', 'comp_nameyear3',
                                       'Posyear4', 'Urlyear4', 'Player_urlyear4', 'regionyear4', 'squadyear4', 'player_numyear4', 'player_nameyear4', 'player_positionyear4', 'player_dobyear4', 'player_nationalityyear4', 'current_clubyear4', 'date_joinedyear4', 'joined_fromyear4', 'contract_expiryyear4', 'Player_yyear4', 'Type_2019year4', 'Type_2020year4', 'Type_2021year4', 'Type_2022year4', 'comp_nameyear4'], axis=1, inplace=True)

MarketModelFiveYeardf.drop(columns = ['Posyear2', 'Urlyear2', 'Player_urlyear2', 'regionyear2', 'squadyear2', 'player_numyear2', 'player_nameyear2', 'player_positionyear2', 'player_dobyear2', 'player_nationalityyear2', 'current_clubyear2', 'date_joinedyear2', 'joined_fromyear2', 'contract_expiryyear2', 'Player_yyear2', 'Type_2019year2', 'Type_2020year2', 'Type_2021year2', 'Type_2022year2', 'comp_nameyear2',
                                       'Posyear3', 'Urlyear3', 'Player_urlyear3', 'regionyear3', 'squadyear3', 'player_numyear3', 'player_nameyear3', 'player_positionyear3', 'player_dobyear3', 'player_nationalityyear3', 'current_clubyear3', 'date_joinedyear3', 'joined_fromyear3', 'contract_expiryyear3', 'Player_yyear3', 'Type_2019year3', 'Type_2020year3', 'Type_2021year3', 'Type_2022year3', 'comp_nameyear3',
                                       'Posyear4', 'Urlyear4', 'Player_urlyear4', 'regionyear4', 'squadyear4', 'player_numyear4', 'player_nameyear4', 'player_positionyear4', 'player_dobyear4', 'player_nationalityyear4', 'current_clubyear4', 'date_joinedyear4', 'joined_fromyear4', 'contract_expiryyear4', 'Player_yyear4', 'Type_2019year4', 'Type_2020year4', 'Type_2021year4', 'Type_2022year4', 'comp_nameyear4',
                                       'Posyear5', 'Urlyear5', 'Player_urlyear5', 'regionyear5', 'squadyear5', 'player_numyear5', 'player_nameyear5', 'player_positionyear5', 'player_dobyear5', 'player_nationalityyear5', 'current_clubyear5', 'date_joinedyear5', 'joined_fromyear5', 'contract_expiryyear5', 'Player_yyear5', 'Type_2019year5', 'Type_2020year5', 'Type_2021year5', 'Type_2022year5', 'comp_nameyear5'], axis=1, inplace=True)

print(MarketModelTwoYeardf['Compyear2'])

MarketModelTwoYeardf.to_csv('/content/drive/MyDrive/PlayerValue/Market2year.csv')

MarketModelThreeYeardf.to_csv('/content/drive/MyDrive/PlayerValue/Market3year.csv')

MarketModelFourYeardf.to_csv('/content/drive/MyDrive/PlayerValue/Market4year.csv')

MarketModelFiveYeardf.to_csv('/content/drive/MyDrive/PlayerValue/Market5year.csv')

#One year model
xMarket1yearEncoding = pd.DataFrame()
xMarket1year["Squad"] = xMarket1year["Squad"].astype("category")
extractedMarket1year = xMarket1year["Squad"]
xMarket1year.drop("Squad", axis = 1)
xMarket1yearEncoding = pd.concat([xMarket1yearEncoding, extractedMarket1year], axis=1)
xMarket1year["Comp"] = xMarket1year["Comp"].astype("category")
extractedMarket1year = xMarket1year["Comp"]
xMarket1year.drop("Comp", axis = 1)
xMarket1yearEncoding = pd.concat([xMarket1yearEncoding, extractedMarket1year], axis=1)
xMarket1year["Nation"] = xMarket1year["Nation"].astype("category")
extractedMarket1year = xMarket1year["Nation"]
xMarket1year.drop("Nation", axis = 1)
xMarket1yearEncoding = pd.concat([xMarket1yearEncoding, extractedMarket1year], axis=1)
xMarket1year["player_foot"] = xMarket1year["player_foot"].astype("category")
extractedMarket1year = xMarket1year["player_foot"]
xMarket1year.drop("player_foot", axis = 1)
xMarket1yearEncoding = pd.concat([xMarket1yearEncoding, extractedMarket1year], axis=1)
xMarket1year["country"] = xMarket1year["country"].astype("category")
extractedMarket1year = xMarket1year["country"]
xMarket1year.drop("country", axis = 1)
xMarket1yearEncoding = pd.concat([xMarket1yearEncoding, extractedMarket1year], axis=1)

encMarket1year = OneHotEncoder()
encMarket1year.fit(xMarket1yearEncoding)

#transform categorical features
XMarket1year_encoded = encMarket1year.transform(xMarket1yearEncoding).toarray()
feature_namesMarket1year = xMarket1yearEncoding.columns
new_feature_namesMarket1year = encMarket1year.get_feature_names_out(feature_namesMarket1year)
XMarket1encoded = pd.DataFrame(XMarket1year_encoded, columns= new_feature_namesMarket1year)
xMarket1yearFinal = xMarket1year.join(XMarket1encoded)

xMarket1year_train, xMarket1year_test, yMarket1year_train, yMarket1year_test = train_test_split(xMarket1yearFinal, yMarket1year, random_state=1, train_size=0.8)
MarketModelXGB1year = xgb.XGBRegressor(tree_method="approx", seed = 1, enable_categorical=True, n_estimators = 15, eval_metric = 'mae',)
MarketModelXGB1year.fit(xMarket1year_train, yMarket1year_train, verbose = True, eval_set = [(xMarket1year_train, yMarket1year_train), (xMarket1year_test, yMarket1year_test)])

feat_imp_list = list(zip ( list(MarketModelXGB1year.feature_importances_) , xMarket1year_train.columns.to_list()) )
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])

pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(400))

yMarket1year_pred = MarketModelXGB1year.predict(xMarket1year_test)
print(r2_score(yMarket1year_test, yMarket1year_pred))
print(mean_absolute_error(yMarket1year_test, yMarket1year_pred))

MarketModelXGB1year.get_booster().dump_model("out.txt")

plt.figure(figsize=(30,20))
xgb.plot_tree(MarketModelXGB1year)
plt.show()

#Two year model
xMarket2yearEncoding = pd.DataFrame()
xMarket2year["Squad"] = xMarket2year["Squad"].astype("category")
extractedMarket2year = xMarket2year["Squad"]
xMarket2year.drop("Squad", axis = 1)
xMarket2yearEncoding = pd.concat([xMarket2yearEncoding, extractedMarket2year], axis=1)
xMarket2year["Comp"] = xMarket2year["Comp"].astype("category")
extractedMarket2year = xMarket2year["Comp"]
xMarket2year.drop("Comp", axis = 1)
xMarket2yearEncoding = pd.concat([xMarket2yearEncoding, extractedMarket2year], axis=1)
xMarket2year["Nation"] = xMarket2year["Nation"].astype("category")
extractedMarket2year = xMarket2year["Nation"]
xMarket2year.drop("Nation", axis = 1)
xMarket2yearEncoding = pd.concat([xMarket2yearEncoding, extractedMarket2year], axis=1)
xMarket2year["player_foot"] = xMarket2year["player_foot"].astype("category")
extractedMarket2year = xMarket2year["player_foot"]
xMarket2year.drop("player_foot", axis = 1)
xMarket2yearEncoding = pd.concat([xMarket2yearEncoding, extractedMarket2year], axis=1)
xMarket2year["country"] = xMarket2year["country"].astype("category")
extractedMarket2year = xMarket2year["country"]
xMarket2year.drop("country", axis = 1)

xMarket2year["Squadyear2"] = xMarket2year["Squadyear2"].astype("category")
extractedMarket2year = xMarket2year["Squadyear2"]
xMarket2year.drop("Squadyear2", axis = 1)
xMarket2yearEncoding = pd.concat([xMarket2yearEncoding, extractedMarket2year], axis=1)
xMarket2year["Compyear2"] = xMarket2year["Compyear2"].astype("category")
extractedMarket2year = xMarket2year["Compyear2"]
xMarket2year.drop("Compyear2", axis = 1)
xMarket2yearEncoding = pd.concat([xMarket2yearEncoding, extractedMarket2year], axis=1)
xMarket2year["Nationyear2"] = xMarket2year["Nationyear2"].astype("category")
extractedMarket2year = xMarket2year["Nationyear2"]
xMarket2year.drop("Nationyear2", axis = 1)
xMarket2yearEncoding = pd.concat([xMarket2yearEncoding, extractedMarket2year], axis=1)
xMarket2year["player_footyear2"] = xMarket2year["player_footyear2"].astype("category")
extractedMarket2year = xMarket2year["player_footyear2"]
xMarket2year.drop("player_footyear2", axis = 1)
xMarket2yearEncoding = pd.concat([xMarket2yearEncoding, extractedMarket2year], axis=1)
xMarket2year["countryyear2"] = xMarket2year["countryyear2"].astype("category")
extractedMarket2year = xMarket2year["countryyear2"]
xMarket2year.drop("countryyear2", axis = 1)
xMarket2yearEncoding = pd.concat([xMarket2yearEncoding, extractedMarket2year], axis=1)

encMarket2year = OneHotEncoder()
encMarket2year.fit(xMarket2yearEncoding)

#transform categorical features
XMarket2year_encoded = encMarket2year.transform(xMarket2yearEncoding).toarray()
feature_namesMarket2year = xMarket2yearEncoding.columns
new_feature_namesMarket2year = encMarket2year.get_feature_names_out(feature_namesMarket2year)
XMarket2encoded = pd.DataFrame(XMarket2year_encoded, columns= new_feature_namesMarket2year)
xMarket2yearFinal = xMarket2year.join(XMarket2encoded)

xMarket2year_train, xMarket2year_test, yMarket2year_train, yMarket2year_test = train_test_split(xMarket2yearFinal, yMarket2year, random_state=1, train_size=0.8)
MarketModelXGB2year = xgb.XGBRegressor(tree_method="approx", seed = 1, enable_categorical=True, n_estimators = 200, eval_metric = 'mae',)
MarketModelXGB2year.fit(xMarket2year_train, yMarket2year_train, verbose = True, eval_set = [(xMarket2year_train, yMarket2year_train), (xMarket2year_test, yMarket2year_test)])

feat_imp_list = list(zip ( list(MarketModelXGB2year.feature_importances_) , xMarket2year_train.columns.to_list()) )
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])

pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(400))

yMarket2year_pred = MarketModelXGB2year.predict(xMarket2year_test)
print(r2_score(yMarket2year_test, yMarket2year_pred))
print(mean_absolute_error(yMarket2year_test, yMarket2year_pred))

#Three year model
xMarket3yearEncoding = pd.DataFrame()

xMarket3year["Squad"] = xMarket3year["Squad"].astype("category")
extractedMarket3year = xMarket3year["Squad"]
xMarket3year.drop("Squad", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["Comp"] = xMarket3year["Comp"].astype("category")
extractedMarket3year = xMarket3year["Comp"]
xMarket3year.drop("Comp", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["Nation"] = xMarket3year["Nation"].astype("category")
extractedMarket3year = xMarket3year["Nation"]
xMarket3year.drop("Nation", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["player_foot"] = xMarket3year["player_foot"].astype("category")
extractedMarket3year = xMarket3year["player_foot"]
xMarket3year.drop("player_foot", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["country"] = xMarket3year["country"].astype("category")
extractedMarket3year = xMarket3year["country"]
xMarket3year.drop("country", axis=1, inplace=True)

xMarket3year["Squadyear2"] = xMarket3year["Squadyear2"].astype("category")
extractedMarket3year = xMarket3year["Squadyear2"]
xMarket3year.drop("Squadyear2", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["Compyear2"] = xMarket3year["Compyear2"].astype("category")
extractedMarket3year = xMarket3year["Compyear2"]
xMarket3year.drop("Compyear2", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["Nationyear2"] = xMarket3year["Nationyear2"].astype("category")
extractedMarket3year = xMarket3year["Nationyear2"]
xMarket3year.drop("Nationyear2", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["player_footyear2"] = xMarket3year["player_footyear2"].astype("category")
extractedMarket3year = xMarket3year["player_footyear2"]
xMarket3year.drop("player_footyear2", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["countryyear2"] = xMarket3year["countryyear2"].astype("category")
extractedMarket3year = xMarket3year["countryyear2"]
xMarket3year.drop("countryyear2", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["Squadyear3"] = xMarket3year["Squadyear3"].astype("category")
extractedMarket3year = xMarket3year["Squadyear3"]
xMarket3year.drop("Squadyear3", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["Compyear3"] = xMarket3year["Compyear3"].astype("category")
extractedMarket3year = xMarket3year["Compyear3"]
xMarket3year.drop("Compyear3", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["Nationyear3"] = xMarket3year["Nationyear3"].astype("category")
extractedMarket3year = xMarket3year["Nationyear3"]
xMarket3year.drop("Nationyear3", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["player_footyear3"] = xMarket3year["player_footyear3"].astype("category")
extractedMarket3year = xMarket3year["player_footyear3"]
xMarket3year.drop("player_footyear3", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

xMarket3year["countryyear3"] = xMarket3year["countryyear3"].astype("category")
extractedMarket3year = xMarket3year["countryyear3"]
xMarket3year.drop("countryyear3", axis=1, inplace=True)
xMarket3yearEncoding = pd.concat([xMarket3yearEncoding, extractedMarket3year], axis=1)

encMarket3year = OneHotEncoder()
encMarket3year.fit(xMarket3yearEncoding)

#transform categorical features
XMarket3year_encoded = encMarket3year.transform(xMarket3yearEncoding).toarray()
feature_namesMarket3year = xMarket3yearEncoding.columns
new_feature_namesMarket3year = encMarket3year.get_feature_names_out(feature_namesMarket3year)
XMarket3encoded = pd.DataFrame(XMarket3year_encoded, columns= new_feature_namesMarket3year)
xMarket3yearFinal = xMarket3year.join(XMarket3encoded)

xMarket3year_train, xMarket3year_test, yMarket3year_train, yMarket3year_test = train_test_split(xMarket3yearFinal, yMarket3year, random_state=1, train_size=0.8)
MarketModelXGB3year = xgb.XGBRegressor(tree_method="approx", seed = 1, enable_categorical=True, n_estimators = 200, eval_metric = 'mae',)
MarketModelXGB3year.fit(xMarket3year_train, yMarket3year_train, verbose = True, eval_set = [(xMarket3year_train, yMarket3year_train), (xMarket3year_test, yMarket3year_test)])

feat_imp_list = list(zip ( list(MarketModelXGB3year.feature_importances_) , xMarket3year_train.columns.to_list()) )
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])

#unSub_Subs, err - search up
pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(100))

yMarket3year_pred = MarketModelXGB3year.predict(xMarket3year_test)
print(r2_score(yMarket3year_test, yMarket3year_pred))
print(mean_absolute_error(yMarket3year_test, yMarket3year_pred))

#4 year model
xMarket4yearEncoding = pd.DataFrame()

xMarket4year["Squad"] = xMarket4year["Squad"].astype("category")
extractedMarket4year = xMarket4year["Squad"]
xMarket4year.drop("Squad", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["Comp"] = xMarket4year["Comp"].astype("category")
extractedMarket4year = xMarket4year["Comp"]
xMarket4year.drop("Comp", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["Nation"] = xMarket4year["Nation"].astype("category")
extractedMarket4year = xMarket4year["Nation"]
xMarket4year.drop("Nation", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["player_foot"] = xMarket4year["player_foot"].astype("category")
extractedMarket4year = xMarket4year["player_foot"]
xMarket4year.drop("player_foot", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["country"] = xMarket4year["country"].astype("category")
extractedMarket4year = xMarket4year["country"]
xMarket4year.drop("country", axis=1, inplace=True)

xMarket4year["Squadyear2"] = xMarket4year["Squadyear2"].astype("category")
extractedMarket4year = xMarket4year["Squadyear2"]
xMarket4year.drop("Squadyear2", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["Compyear2"] = xMarket4year["Compyear2"].astype("category")
extractedMarket4year = xMarket4year["Compyear2"]
xMarket4year.drop("Compyear2", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["Nationyear2"] = xMarket4year["Nationyear2"].astype("category")
extractedMarket4year = xMarket4year["Nationyear2"]
xMarket4year.drop("Nationyear2", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["player_footyear2"] = xMarket4year["player_footyear2"].astype("category")
extractedMarket4year = xMarket4year["player_footyear2"]
xMarket4year.drop("player_footyear2", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["countryyear2"] = xMarket4year["countryyear2"].astype("category")
extractedMarket4year = xMarket4year["countryyear2"]
xMarket4year.drop("countryyear2", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["Squadyear3"] = xMarket4year["Squadyear3"].astype("category")
extractedMarket4year = xMarket4year["Squadyear3"]
xMarket4year.drop("Squadyear3", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["Compyear3"] = xMarket4year["Compyear3"].astype("category")
extractedMarket4year = xMarket4year["Compyear3"]
xMarket4year.drop("Compyear3", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["Nationyear3"] = xMarket4year["Nationyear3"].astype("category")
extractedMarket4year = xMarket4year["Nationyear3"]
xMarket4year.drop("Nationyear3", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["player_footyear3"] = xMarket4year["player_footyear3"].astype("category")
extractedMarket4year = xMarket4year["player_footyear3"]
xMarket4year.drop("player_footyear3", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["countryyear3"] = xMarket4year["countryyear3"].astype("category")
extractedMarket4year = xMarket4year["countryyear3"]
xMarket4year.drop("countryyear3", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["Squadyear4"] = xMarket4year["Squadyear4"].astype("category")
extractedMarket4year = xMarket4year["Squadyear4"]
xMarket4year.drop("Squadyear4", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["Compyear4"] = xMarket4year["Compyear4"].astype("category")
extractedMarket4year = xMarket4year["Compyear4"]
xMarket4year.drop("Compyear4", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["Nationyear4"] = xMarket4year["Nationyear4"].astype("category")
extractedMarket4year = xMarket4year["Nationyear4"]
xMarket4year.drop("Nationyear4", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["player_footyear4"] = xMarket4year["player_footyear4"].astype("category")
extractedMarket4year = xMarket4year["player_footyear4"]
xMarket4year.drop("player_footyear4", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

xMarket4year["countryyear4"] = xMarket4year["countryyear4"].astype("category")
extractedMarket4year = xMarket4year["countryyear4"]
xMarket4year.drop("countryyear4", axis=1, inplace=True)
xMarket4yearEncoding = pd.concat([xMarket4yearEncoding, extractedMarket4year], axis=1)

encMarket4year = OneHotEncoder()
encMarket4year.fit(xMarket4yearEncoding)

#transform categorical features
XMarket4year_encoded = encMarket4year.transform(xMarket4yearEncoding).toarray()
feature_namesMarket4year = xMarket4yearEncoding.columns
new_feature_namesMarket4year = encMarket4year.get_feature_names_out(feature_namesMarket4year)
XMarket4encoded = pd.DataFrame(XMarket4year_encoded, columns= new_feature_namesMarket4year)
xMarket4yearFinal = xMarket4year.join(XMarket4encoded)

xMarket4year_train, xMarket4year_test, yMarket4year_train, yMarket4year_test = train_test_split(xMarket4yearFinal, yMarket4year, random_state=1, train_size=0.8)
MarketModelXGB4year = xgb.XGBRegressor(tree_method="approx", seed = 1, enable_categorical=True, n_estimators = 200, eval_metric = 'mae',)
MarketModelXGB4year.fit(xMarket4year_train, yMarket4year_train, verbose = True, eval_set = [(xMarket4year_train, yMarket4year_train), (xMarket4year_test, yMarket4year_test)])

feat_imp_list = list(zip ( list(MarketModelXGB4year.feature_importances_) , xMarket4year_train.columns.to_list()) )
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])

#unSub_Subs, err - search up
pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(100))

yMarket4year_pred = MarketModelXGB4year.predict(xMarket4year_test)
print(r2_score(yMarket4year_test, yMarket4year_pred))
print(mean_absolute_error(yMarket4year_test, yMarket4year_pred))

#5 year model
xMarket5yearEncoding = pd.DataFrame()

xMarket5year["Squad"] = xMarket5year["Squad"].astype("category")
extractedMarket5year = xMarket5year["Squad"]
xMarket5year.drop("Squad", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Comp"] = xMarket5year["Comp"].astype("category")
extractedMarket5year = xMarket5year["Comp"]
xMarket5year.drop("Comp", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Nation"] = xMarket5year["Nation"].astype("category")
extractedMarket5year = xMarket5year["Nation"]
xMarket5year.drop("Nation", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["player_foot"] = xMarket5year["player_foot"].astype("category")
extractedMarket5year = xMarket5year["player_foot"]
xMarket5year.drop("player_foot", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["country"] = xMarket5year["country"].astype("category")
extractedMarket5year = xMarket5year["country"]
xMarket5year.drop("country", axis=1, inplace=True)

xMarket5year["Squadyear2"] = xMarket5year["Squadyear2"].astype("category")
extractedMarket5year = xMarket5year["Squadyear2"]
xMarket5year.drop("Squadyear2", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Compyear2"] = xMarket5year["Compyear2"].astype("category")
extractedMarket5year = xMarket5year["Compyear2"]
xMarket5year.drop("Compyear2", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Nationyear2"] = xMarket5year["Nationyear2"].astype("category")
extractedMarket5year = xMarket5year["Nationyear2"]
xMarket5year.drop("Nationyear2", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["player_footyear2"] = xMarket5year["player_footyear2"].astype("category")
extractedMarket5year = xMarket5year["player_footyear2"]
xMarket5year.drop("player_footyear2", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["countryyear2"] = xMarket5year["countryyear2"].astype("category")
extractedMarket5year = xMarket5year["countryyear2"]
xMarket5year.drop("countryyear2", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Squadyear3"] = xMarket5year["Squadyear3"].astype("category")
extractedMarket5year = xMarket5year["Squadyear3"]
xMarket5year.drop("Squadyear3", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Compyear3"] = xMarket5year["Compyear3"].astype("category")
extractedMarket5year = xMarket5year["Compyear3"]
xMarket5year.drop("Compyear3", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Nationyear3"] = xMarket5year["Nationyear3"].astype("category")
extractedMarket5year = xMarket5year["Nationyear3"]
xMarket5year.drop("Nationyear3", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["player_footyear3"] = xMarket5year["player_footyear3"].astype("category")
extractedMarket5year = xMarket5year["player_footyear3"]
xMarket5year.drop("player_footyear3", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["countryyear3"] = xMarket5year["countryyear3"].astype("category")
extractedMarket5year = xMarket5year["countryyear3"]
xMarket5year.drop("countryyear3", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Squadyear4"] = xMarket5year["Squadyear4"].astype("category")
extractedMarket5year = xMarket5year["Squadyear4"]
xMarket5year.drop("Squadyear4", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Compyear4"] = xMarket5year["Compyear4"].astype("category")
extractedMarket5year = xMarket5year["Compyear4"]
xMarket5year.drop("Compyear4", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Nationyear4"] = xMarket5year["Nationyear4"].astype("category")
extractedMarket5year = xMarket5year["Nationyear4"]
xMarket5year.drop("Nationyear4", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["player_footyear4"] = xMarket5year["player_footyear4"].astype("category")
extractedMarket5year = xMarket5year["player_footyear4"]
xMarket5year.drop("player_footyear4", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["countryyear4"] = xMarket5year["countryyear4"].astype("category")
extractedMarket5year = xMarket5year["countryyear4"]
xMarket5year.drop("countryyear4", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Squadyear5"] = xMarket5year["Squadyear5"].astype("category")
extractedMarket5year = xMarket5year["Squadyear5"]
xMarket5year.drop("Squadyear5", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Compyear5"] = xMarket5year["Compyear5"].astype("category")
extractedMarket5year = xMarket5year["Compyear5"]
xMarket5year.drop("Compyear5", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["Nationyear5"] = xMarket5year["Nationyear5"].astype("category")
extractedMarket5year = xMarket5year["Nationyear5"]
xMarket5year.drop("Nationyear5", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["player_footyear5"] = xMarket5year["player_footyear5"].astype("category")
extractedMarket5year = xMarket5year["player_footyear5"]
xMarket5year.drop("player_footyear5", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)

xMarket5year["countryyear5"] = xMarket5year["countryyear5"].astype("category")
extractedMarket5year = xMarket5year["countryyear5"]
xMarket5year.drop("countryyear5", axis=1, inplace=True)
xMarket5yearEncoding = pd.concat([xMarket5yearEncoding, extractedMarket5year], axis=1)


encMarket5year = OneHotEncoder()
encMarket5year.fit(xMarket5yearEncoding)

#transform categorical features
XMarket5year_encoded = encMarket5year.transform(xMarket5yearEncoding).toarray()
feature_namesMarket5year = xMarket5yearEncoding.columns
new_feature_namesMarket5year = encMarket5year.get_feature_names_out(feature_namesMarket5year)
XMarket5encoded = pd.DataFrame(XMarket5year_encoded, columns= new_feature_namesMarket5year)
xMarket5yearFinal = xMarket5year.join(XMarket5encoded)

xMarket5year_train, xMarket5year_test, yMarket5year_train, yMarket5year_test = train_test_split(xMarket5yearFinal, yMarket5year, random_state=1, train_size=0.8)
MarketModelXGB5year = xgb.XGBRegressor(tree_method="approx", seed = 1, enable_categorical=True, n_estimators = 200, eval_metric = 'mae',)
MarketModelXGB5year.fit(xMarket5year_train, yMarket5year_train, verbose = True, eval_set = [(xMarket5year_train, yMarket5year_train), (xMarket5year_test, yMarket5year_test)])

feat_imp_list = list(zip ( list(MarketModelXGB5year.feature_importances_) , xMarket5year_train.columns.to_list()) )
feature_imp_df = pd.DataFrame(sorted(feat_imp_list, key=lambda x: x[0], reverse=True) , columns = ['feature_value','feature_name'])

#unSub_Subs, err - search up
pd.set_option('display.max_rows', 500)
print(feature_imp_df.head(100))

yMarket5year_pred = MarketModelXGB5year.predict(xMarket5year_test)
print(r2_score(yMarket5year_test, yMarket5year_pred))
print(mean_absolute_error(yMarket5year_test, yMarket5year_pred))

#Iterating thorugh hyperparameters of model
x1year_train, x1year_test, y1year_train, y1year_test = train_test_split(x1year, y1year, random_state=1, train_size = 0.8)
for depth in [None, 5, 10, 15, 20]:
  for learning_rate in [0.1, 0.01, 0.001]:
    for min_child_weight in [1,2,3]:
      for gamma in [0.1, 0.2, 0.3]:
        InjuryModelXGB1year = xgb.XGBRegressor(tree_method="approx", seed=1, max_depth = depth, learning_rate = learning_rate, min_child_weight = min_child_weight, gamma = gamma, enable_categorical = True)
        InjuryModelXGB1year.fit(x1year_train, y1year_train, verbose = True, eval_set = [(x1year_test, y1year_test)])
        y1year_pred = InjuryModelXGB1year.predict(x1year_test)
        print(r2_score(y1year_test, y1year_pred))
        InjuryModelXGB1year.save_model('xgb_injury_1year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y1year_test, y1year_pred)) + '.json')
        files.download('xgb_injury_1year_depth' + str(depth) + '_learningrate' + str(learning_rate) + '_childweight' + str(min_child_weight) + '_gamma' + str(gamma) + '_r2score' + str(r2_score(y1year_test, y1year_pred)) + '.json')
