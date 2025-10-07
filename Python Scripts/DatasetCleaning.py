#import libraries
!pip install --upgrade pip
!pip install pandas --upgrade pip
import pandas as pd
!pip install pyreadr --upgrade pip
import pyreadr as pyr

from google.colab import drive
drive.mount('/content/drive')

mappingdf = pd.read_csv('fbref_to_tm_mapping.csv', encoding='latin-1')
mappingdf.rename(columns = {'PlayerFBref':'Player', 'UrlFBref':'Url', 'UrlTmarkt':'player_url'}, inplace = True)
mappingdf = mappingdf[['Url', 'player_url']]
mappingdf.head()
mappingdf.to_csv('/drive/My Drive/PlayerValue/mappingdf.csv')

#read all data from 2018 freb
#Defense
playerDefense2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/defense2018.RData')
playerDefense2018df = playerDefense2018["defense2018"]
#GCA
playerGCA2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/gca2018.RData')
playerGCA2018df = playerGCA2018["gca2018"]
#Keepers
playerKeepers2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/keepers2018.RData')
playerKeepers2018df = playerKeepers2018["keepers2018"]
#Keepers Advanced
playerKeepersAdv2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/keepersAdvanced2018.RData')
playerKeepersAdv2018df = playerKeepersAdv2018["keepersAdvanced2018"]
#misc
playerMisc2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/misc2018.RData')
playerMisc2018df = playerMisc2018["misc2018"]
#passing
playerPassing2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/passing2018.RData')
playerPassing2018df = playerPassing2018["passing2018"]
#passing types
playerPassingTypes2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/passingTypes2018.RData')
playerPassingTypes2018df = playerPassingTypes2018["passingTypes2018"]
#playing time
playerPlayingTime2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/playingTime2018.RData')
playerPlayingTime2018df = playerPlayingTime2018["playingTime2018"]
playerPlayingTime2018df.count()
#possession
playerPossession2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/possession2018.RData')
playerPossession2018df = playerPossession2018["possession2018"]
#shooting
playerShooting2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/shooting2018.RData')
playerShooting2018df = playerShooting2018["shooting2018"]
#Standard
playerStandard2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/standard2018.RData')
playerStandard2018df = playerStandard2018["standard2018"]
#Valuations
playerValuation2018 = pyr.read_r('/Users/25AlecZ/Documents/PlayerValue/2018/valuations2018.RData')
playerValuation2018df = playerValuation2018["valuations2018"]

#merge to create 2018 player dataframe
#merge defense and gca
playerdf2018 = pd.merge(playerDefense2018df, playerGCA2018df)
#merge player and standard
playerdf2018 = pd.merge(playerdf2018, playerStandard2018df, how = 'left')
#merge player and keepers
playerdf2018 = pd.merge(playerdf2018, playerKeepers2018df, how = 'left')
#merge player and keepers advanced
playerdf2018 = pd.merge(playerdf2018, playerKeepersAdv2018df, how = 'left')
#merge player and playing time
playerdf2018 = pd.merge(playerdf2018, playerPlayingTime2018df, how = 'left')
#merge player and misc
playerdf2018 = pd.merge(playerdf2018, playerMisc2018df)
#merge player and passing
playerdf2018 = pd.merge(playerdf2018, playerPassing2018df)
#merge player and passing types
playerdf2018 = pd.merge(playerdf2018, playerPassingTypes2018df)
#merge player and possession
playerdf2018 = pd.merge(playerdf2018, playerPossession2018df)
#merge player and shooting
playerdf2018 = pd.merge(playerdf2018, playerShooting2018df)
#merge mapping csv
playerdf2018 = pd.merge(playerdf2018, mappingdf, on = 'Url', how = 'left')
#merge valuations
playerdf2018 = pd.merge(playerdf2018, playerValuation2018df, how = 'left')
playerdf2018.to_csv('/drive/My Drive/PlayerValue/playerdf2018.csv')

#read all data from 2019 freb
#Defense
playerDefense2019 = pyr.read_r('defense2019.RData')
playerDefense2019df = playerDefense2019["defense2019"]
#GCA
playerGCA2019 = pyr.read_r('gca2019.RData')
playerGCA2019df = playerGCA2019["gca2019"]
#Keepers
playerKeepers2019 = pyr.read_r('keepers2019.RData')
playerKeepers2019df = playerKeepers2019["keepers2019"]
#Keepers Advanced
playerKeepersAdv2019 = pyr.read_r('keepersAdvanced2019.RData')
playerKeepersAdv2019df = playerKeepersAdv2019["keepersAdvanced2019"]
#misc
playerMisc2019 = pyr.read_r('misc2019.RData')
playerMisc2019df = playerMisc2019["misc2019"]
#passing
playerPassing2019 = pyr.read_r('passing2019.RData')
playerPassing2019df = playerPassing2019["passing2019"]
#passing types
playerPassingTypes2019 = pyr.read_r('passingTypes2019.RData')
playerPassingTypes2019df = playerPassingTypes2019["passingTypes2019"]
#playing time
playerPlayingTime2019 = pyr.read_r('playingTime2019.RData')
playerPlayingTime2019df = playerPlayingTime2019["playingTime2019"]
playerPlayingTime2019df.count()
#possession
playerPossession2019 = pyr.read_r('possession2019.RData')
playerPossession2019df = playerPossession2019["possession2019"]
#shooting
playerShooting2019 = pyr.read_r('shooting2019.RData')
playerShooting2019df = playerShooting2019["shooting2019"]
#Standard
playerStandard2019 = pyr.read_r('standard2019.RData')
playerStandard2019df = playerStandard2019["standard2019"]
#Valuations
playerValuation2019 = pyr.read_r('valuations2019.RData')
playerValuation2019df = playerValuation2019["valuations2019"]

#merge to create 2019 player dataframe
#merge defense and gca
playerdf2019 = pd.merge(playerDefense2019df, playerGCA2019df)
#merge player and standard
playerdf2019 = pd.merge(playerdf2019, playerStandard2019df, how = 'left')
#merge player and keepers
playerdf2019 = pd.merge(playerdf2019, playerKeepers2019df, how = 'left')
#merge player and keepers advanced
playerdf2019 = pd.merge(playerdf2019, playerKeepersAdv2019df, how = 'left')
#merge player and playing time
playerdf2019 = pd.merge(playerdf2019, playerPlayingTime2019df, how = 'left')
#merge player and misc
playerdf2019 = pd.merge(playerdf2019, playerMisc2019df)
#merge player and passing
playerdf2019 = pd.merge(playerdf2019, playerPassing2019df)
#merge player and passing types
playerdf2019 = pd.merge(playerdf2019, playerPassingTypes2019df)
#merge player and possession
playerdf2019 = pd.merge(playerdf2019, playerPossession2019df)
#merge player and shooting
playerdf2019 = pd.merge(playerdf2019, playerShooting2019df)
#merge mapping csv
playerdf2019 = pd.merge(playerdf2019, mappingdf, on = 'Url', how = 'left')
#merge valuations
playerdf2019 = pd.merge(playerdf2019, playerValuation2019df, how = 'left')
playerdf2019.to_csv('/drive/My Drive/PlayerValue/playerdf2019.csv')

#read all data from 2020 freb
#Defense
playerDefense2020 = pyr.read_r('defense2020.RData')
playerDefense2020df = playerDefense2020["defense2020"]
#GCA
playerGCA2020 = pyr.read_r('gca2020.RData')
playerGCA2020df = playerGCA2020["gca2020"]
#Keepers
playerKeepers2020 = pyr.read_r('keepers2020.RData')
playerKeepers2020df = playerKeepers2020["keepers2020"]
#Keepers Advanced
playerKeepersAdv2020 = pyr.read_r('keepersAdvanced2020.RData')
playerKeepersAdv2020df = playerKeepersAdv2020["keepersAdvanced2020"]
#misc
playerMisc2020 = pyr.read_r('misc2020.RData')
playerMisc2020df = playerMisc2020["misc2020"]
#passing
playerPassing2020 = pyr.read_r('passing2020.RData')
playerPassing2020df = playerPassing2020["passing2020"]
#passing types
playerPassingTypes2020 = pyr.read_r('passingTypes2020.RData')
playerPassingTypes2020df = playerPassingTypes2020["passingTypes2020"]
#playing time
playerPlayingTime2020 = pyr.read_r('playingTime2020.RData')
playerPlayingTime2020df = playerPlayingTime2020["playingTime2020"]
playerPlayingTime2020df.count()
#possession
playerPossession2020 = pyr.read_r('possession2020.RData')
playerPossession2020df = playerPossession2020["possession2020"]
#shooting
playerShooting2020 = pyr.read_r('shooting2020.RData')
playerShooting2020df = playerShooting2020["shooting2020"]
#Standard
playerStandard2020 = pyr.read_r('standard2020.RData')
playerStandard2020df = playerStandard2020["standard2020"]
#Valuations
playerValuation2020 = pyr.read_r('valuations2020.RData')
playerValuation2020df = playerValuation2020["valuations2020"]

#merge to create 2020 player dataframe
#merge defense and gca
playerdf2020 = pd.merge(playerDefense2020df, playerGCA2020df)
#merge player and standard
playerdf2020 = pd.merge(playerdf2020, playerStandard2020df, how = 'left')
#merge player and keepers
playerdf2020 = pd.merge(playerdf2020, playerKeepers2020df, how = 'left')
#merge player and keepers advanced
playerdf2020 = pd.merge(playerdf2020, playerKeepersAdv2020df, how = 'left')
#merge player and playing time
playerdf2020 = pd.merge(playerdf2020, playerPlayingTime2020df, how = 'left')
#merge player and misc
playerdf2020 = pd.merge(playerdf2020, playerMisc2020df)
#merge player and passing
playerdf2020 = pd.merge(playerdf2020, playerPassing2020df)
#merge player and passing types
playerdf2020 = pd.merge(playerdf2020, playerPassingTypes2020df)
#merge player and possession
playerdf2020 = pd.merge(playerdf2020, playerPossession2020df)
#merge player and shooting
playerdf2020 = pd.merge(playerdf2020, playerShooting2020df)
#merge mapping csv
playerdf2020 = pd.merge(playerdf2020, mappingdf, on = 'Url', how = 'left')
#merge valuations
playerdf2020 = pd.merge(playerdf2020, playerValuation2020df, how = 'left')
playerdf2020.to_csv('/drive/My Drive/PlayerValue/playerdf2020.csv')

#read all data from 2021 freb
#Defense
playerDefense2021 = pyr.read_r('defense2021.RData')
playerDefense2021df = playerDefense2021["defense2021"]
#GCA
playerGCA2021 = pyr.read_r('gca2021.RData')
playerGCA2021df = playerGCA2021["gca2021"]
#Keepers
playerKeepers2021 = pyr.read_r('keepers2021.RData')
playerKeepers2021df = playerKeepers2021["keepers2021"]
#Keepers Advanced
playerKeepersAdv2021 = pyr.read_r('keepersAdvanced2021.RData')
playerKeepersAdv2021df = playerKeepersAdv2021["keepersAdvanced2021"]
#misc
playerMisc2021 = pyr.read_r('misc2021.RData')
playerMisc2021df = playerMisc2021["misc2021"]
#passing
playerPassing2021 = pyr.read_r('passing2021.RData')
playerPassing2021df = playerPassing2021["passing2021"]
#passing types
playerPassingTypes2021 = pyr.read_r('passingTypes2021.RData')
playerPassingTypes2021df = playerPassingTypes2021["passingTypes2021"]
#playing time
playerPlayingTime2021 = pyr.read_r('playingTime2021.RData')
playerPlayingTime2021df = playerPlayingTime2021["playingTime2021"]
playerPlayingTime2021df.count()
#possession
playerPossession2021 = pyr.read_r('possession2021.RData')
playerPossession2021df = playerPossession2021["possession2021"]
#shooting
playerShooting2021 = pyr.read_r('shooting2021.RData')
playerShooting2021df = playerShooting2021["shooting2021"]
#Standard
playerStandard2021 = pyr.read_r('standard2021.RData')
playerStandard2021df = playerStandard2021["standard2021"]
#Valuations
playerValuation2021 = pyr.read_r('valuations2021.RData')
playerValuation2021df = playerValuation2021["valuations2021"]

#merge to create 2021 player dataframe
#merge defense and gca
playerdf2021 = pd.merge(playerDefense2021df, playerGCA2021df)
#merge player and standard
playerdf2021 = pd.merge(playerdf2021, playerStandard2021df, how = 'left')
#merge player and keepers
playerdf2021 = pd.merge(playerdf2021, playerKeepers2021df, how = 'left')
#merge player and keepers advanced
playerdf2021 = pd.merge(playerdf2021, playerKeepersAdv2021df, how = 'left')
#merge player and playing time
playerdf2021 = pd.merge(playerdf2021, playerPlayingTime2021df, how = 'left')
#merge player and misc
playerdf2021 = pd.merge(playerdf2021, playerMisc2021df)
#merge player and passing
playerdf2021 = pd.merge(playerdf2021, playerPassing2021df)
#merge player and passing types
playerdf2021 = pd.merge(playerdf2021, playerPassingTypes2021df)
#merge player and possession
playerdf2021 = pd.merge(playerdf2021, playerPossession2021df)
#merge player and shooting
playerdf2021 = pd.merge(playerdf2021, playerShooting2021df)
#merge mapping csv
playerdf2021 = pd.merge(playerdf2021, mappingdf, on = 'Url', how = 'left')
#merge valuations
playerdf2021 = pd.merge(playerdf2021, playerValuation2021df, how = 'left')
playerdf2021.to_csv('/drive/My Drive/PlayerValue/playerdf2021.csv')

#read all data from 2022 freb
#Defense
playerDefense2022 = pyr.read_r('defense2022.RData')
playerDefense2022df = playerDefense2022["defense2022"]
#GCA
playerGCA2022 = pyr.read_r('gca2022.RData')
playerGCA2022df = playerGCA2022["gca2022"]
#Keepers
playerKeepers2022 = pyr.read_r('keepers2022.RData')
playerKeepers2022df = playerKeepers2022["keepers2022"]
#Keepers Advanced
playerKeepersAdv2022 = pyr.read_r('keepersAdvanced2022.RData')
playerKeepersAdv2022df = playerKeepersAdv2022["keepersAdvanced2022"]
#misc
playerMisc2022 = pyr.read_r('misc2022.RData')
playerMisc2022df = playerMisc2022["misc2022"]
#passing
playerPassing2022 = pyr.read_r('passing2022.RData')
playerPassing2022df = playerPassing2022["passing2022"]
#passing types
playerPassingTypes2022 = pyr.read_r('passingTypes2022.RData')
playerPassingTypes2022df = playerPassingTypes2022["passingTypes2022"]
#playing time
playerPlayingTime2022 = pyr.read_r('playingTime2022.RData')
playerPlayingTime2022df = playerPlayingTime2022["playingTime2022"]
playerPlayingTime2022df.count()
#possession
playerPossession2022 = pyr.read_r('possession2022.RData')
playerPossession2022df = playerPossession2022["possession2022"]
#shooting
playerShooting2022 = pyr.read_r('shooting2022.RData')
playerShooting2022df = playerShooting2022["shooting2022"]
#Standard
playerStandard2022 = pyr.read_r('standard2022.RData')
playerStandard2022df = playerStandard2022["standard2022"]
#Valuations
playerValuation2022 = pyr.read_r('valuations2022.RData')
playerValuation2022df = playerValuation2022["valuations2022"]

#merge to create 2022 player dataframe
#merge defense and gca
playerdf2022 = pd.merge(playerDefense2022df, playerGCA2022df)
#merge player and standard
playerdf2022 = pd.merge(playerdf2022, playerStandard2022df, how = 'left')
#merge player and keepers
playerdf2022 = pd.merge(playerdf2022, playerKeepers2022df, how = 'left')
#merge player and keepers advanced
playerdf2022 = pd.merge(playerdf2022, playerKeepersAdv2022df, how = 'left')
#merge player and playing time
playerdf2022 = pd.merge(playerdf2022, playerPlayingTime2022df, how = 'left')
#merge player and misc
playerdf2022 = pd.merge(playerdf2022, playerMisc2022df)
#merge player and passing
playerdf2022 = pd.merge(playerdf2022, playerPassing2022df)
#merge player and passing types
playerdf2022 = pd.merge(playerdf2022, playerPassingTypes2022df)
#merge player and possession
playerdf2022 = pd.merge(playerdf2022, playerPossession2022df)
#merge player and shooting
playerdf2022 = pd.merge(playerdf2022, playerShooting2022df)
#merge mapping csv
playerdf2022 = pd.merge(playerdf2022, mappingdf, on = 'Url', how = 'left')
#merge valuations
playerdf2022 = pd.merge(playerdf2022, playerValuation2022df, how = 'left')
playerdf2022.to_csv('/drive/My Drive/PlayerValue/playerdf2022.csv')

#read all data from 2023 freb
#Defense
playerDefense2023 = pyr.read_r('defense2023.RData')
playerDefense2023df = playerDefense2023["defense2023"]
#GCA
playerGCA2023 = pyr.read_r('gca2023.RData')
playerGCA2023df = playerGCA2023["gca2023"]
#Keepers
playerKeepers2023 = pyr.read_r('keepers2023.RData')
playerKeepers2023df = playerKeepers2023["keepers2023"]
#Keepers Advanced
playerKeepersAdv2023 = pyr.read_r('keepersAdvanced2023.RData')
playerKeepersAdv2023df = playerKeepersAdv2023["keepersAdvanced2023"]
#misc
playerMisc2023 = pyr.read_r('misc2023.RData')
playerMisc2023df = playerMisc2023["misc2023"]
#passing
playerPassing2023 = pyr.read_r('passing2023.RData')
playerPassing2023df = playerPassing2023["passing2023"]
#passing types
playerPassingTypes2023 = pyr.read_r('passingTypes2023.RData')
playerPassingTypes2023df = playerPassingTypes2023["passingTypes2023"]
#playing time
playerPlayingTime2023 = pyr.read_r('playingTime2023.RData')
playerPlayingTime2023df = playerPlayingTime2023["playingTime2023"]
#possession
playerPossession2023 = pyr.read_r('possession2023.RData')
playerPossession2023df = playerPossession2023["possession2023"]
#shooting
playerShooting2023 = pyr.read_r('shooting2023.RData')
playerShooting2023df = playerShooting2023["shooting2023"]
#Standard
playerStandard2023 = pyr.read_r('standard2023.RData')
playerStandard2023df = playerStandard2023["standard2023"]
#Valuations
playerValuation2023 = pyr.read_r('valuations2023.RData')
playerValuation2023df = playerValuation2023["valuations2023"]

#merge to create 2023 player dataframe
#merge defense and gca
playerdf2023 = pd.merge(playerDefense2023df, playerGCA2023df)
#merge player and standard
playerdf2023 = pd.merge(playerdf2023, playerStandard2023df, how = 'left')
#merge player and keepers
playerdf2023 = pd.merge(playerdf2023, playerKeepers2023df, how = 'left')
#merge player and keepers advanced
playerdf2023 = pd.merge(playerdf2023, playerKeepersAdv2023df, how = 'left')
#merge player and playing time
playerdf2023 = pd.merge(playerdf2023, playerPlayingTime2023df, how = 'left')
#merge player and misc
playerdf2023 = pd.merge(playerdf2023, playerMisc2023df)
#merge player and passing
playerdf2023 = pd.merge(playerdf2023, playerPassing2023df)
#merge player and passing types
playerdf2023 = pd.merge(playerdf2023, playerPassingTypes2023df)
#merge player and possession
playerdf2023 = pd.merge(playerdf2023, playerPossession2023df)
#merge player and shooting
playerdf2023 = pd.merge(playerdf2023, playerShooting2023df)
#merge mapping csv
playerdf2023 = pd.merge(playerdf2023, mappingdf, on = 'Url', how = 'left')
#merge valuations
playerdf2023 = pd.merge(playerdf2023, playerValuation2023df, how = 'left')
playerdf2023.to_csv('/drive/My Drive/PlayerValue/playerdf2023.csv')

Injury2018df = pd.read_csv('Injuries 2018 - Sheet1.csv')
Injury2019df = pd.read_csv('Injuries 2019 - Sheet1.csv')
Injury2020df = pd.read_csv('Injuries 2020 - Sheet1.csv')
Injury2021df = pd.read_csv('Injuries 2021 - Sheet1.csv')
Injury2022df = pd.read_csv('Injuries 2022 - Sheet1.csv')
Injury2023df = pd.read_csv('Injuries 2023 - Sheet1.csv')

Injury2023df.head()

Injury2018df.drop_duplicates(inplace = True)
Injury2018df.to_csv('/content/drive/MyDrive/PlayerValue/Injury2018df.csv')

Injury2019df.drop_duplicates(inplace = True)
Injury2019df.to_csv('/content/drive/MyDrive/PlayerValue/Injury2019df.csv')

Injury2020df.drop_duplicates(inplace = True)
Injury2020df.to_csv('/content/drive/MyDrive/PlayerValue/Injury2020df.csv')

Injury2021df.drop_duplicates(inplace = True)
Injury2021df.to_csv('/content/drive/MyDrive/PlayerValue/Injury2021df.csv')

Injury2022df.drop_duplicates(inplace = True)
Injury2022df.to_csv('/content/drive/MyDrive/PlayerValue/Injury2022df.csv')

Injury2023df.drop_duplicates(inplace = True)
Injury2023df.to_csv('/content/drive/MyDrive/PlayerValue/Injury2023df.csv')
