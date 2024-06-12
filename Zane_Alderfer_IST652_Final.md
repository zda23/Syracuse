```python
#I had to create a path for pymongo because Jupyter was claiming it wasn't being found in the directory
import sys
sys.path.append("/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages")

import os
import json
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
```


```python
#This function reads the file into a json file from a folder that I call below
def get_fantasy(folder_name):
    stats = {}
    
    for filename in os.listdir(folder_name):
        with open(os.path.join(folder_name, filename), 'r') as f: 
            content = f.read() # open in readonly mode
            #added a substitute line as python didn't like seeing $ in the input file
            content = re.sub('\$','',content)
            content_json = json.loads(content)
            filename = filename.replace('.json', '')
            content_json2 = {}
            #content_json2[filename] = content_json
            stats = content_json
    return stats
```


```python
#This function allows us to connect to the collection within MongoDB
def save_fantasy(stats, database_name, collection_name, host_name, host_port):  
    try:
        client = pymongo.MongoClient(host_name, host_port)
        print ("Connected successfully!!!")
    except e:
        print ("Could not connect to MongoDB: %s" % e )
    else:
        client.drop_database(database_name)
        #This line allows to rerun the called functions as many times as we want without obtaining any errors
        stats_db = client[database_name]
        stats_coll = stats_db[collection_name]
        stats_coll.insert_many(stats)
        print(f"Added {len(stats)} stats to collection {collection_name}")
        # close the database connection
        client.close()
```


```python
#This function loads the mongodb file into a pandas dataframe for easy use
def load_stats(database_name, collection_name, host_name, host_port):  
    try:
        client = pymongo.MongoClient(host_name, host_port)
        print ("Connected successfully!!!")
    except e:
        print ("Could not connect to MongoDB: %s" % e )
    else:
        
        fantasy_db = client[database_name]
        # create collection named stats
        fantasy_coll = fantasy_db[collection_name]
        #This creates the dictionary of values for the database
        stats_list = {}
        for doc in fantasy_coll.find():
            ID = doc['_id']
            year = doc['Year']
            name = doc['name']
            position = doc['position']
            team = doc['team']
            adp = doc['adp']
            player_id = doc['PlayerID']
            stats_list[ID['oid']] = ([ID['oid'], year, name, position, team, adp, player_id])
        client.close()
        #This line creates the pandas database
        df = pd.DataFrame(stats_list.values(), columns=['_id', 'year', 'name', 'position', 'team', 'adp', 'PlayerID'])
        return df
```


```python
folder_name = 'Fantasy'
database_name = 'fantasy_football'
collection_name = 'stats'
host_name = 'localhost'
host_port = 27017
stats = get_fantasy(folder_name)
save_fantasy(stats, database_name, collection_name, host_name, host_port)
```

    Connected successfully!!!
    Added 1206 stats to collection stats



```python
#This first function allows us to see the team that has the best average of players with the lowest ADP or average draft position
#This helps us indicate which team has the best succes'adp',s in terms of fantasy players
def team_adp_average(stats_list):
    team_group = stats_list.groupby(stats_list.team).mean().sort_values(ascending=True)
    print(team_group)

#After seeing the best team is Cincinatti, this function allows us to see which of their players have had the best adp over the 7 years
def team_top_players(stats_list, team):
    players = stats_list[stats_list.team == team].sort_values('adp')
    print(players)
    
    
stats_data = load_stats(database_name, collection_name, host_name, host_port)
team_adp_average(stats_data)
team_top_players(stats_data, 'CIN')
```

    Connected successfully!!!
                 year         adp
    team                         
    CIN   2020.058824   71.661765
    KC    2020.294118   72.390196
    MIN   2020.125000   73.342500
    LAR   2020.052632   74.152632
    GB    2019.926829   74.317073
    DAL   2020.073171   74.356098
    PIT   2020.189189   75.902703
    ATL   2019.710526   77.207895
    PHI   2019.933333   77.268889
    LAC   2020.075000   77.807500
    CLE   2020.176471   77.894118
    BAL   2020.218750   78.662500
    SEA   2019.717949   81.041026
    TEN   2019.562500   81.134375
    SF    2020.255814   83.302326
    NO    2019.833333   83.740476
    CAR   2019.689655   84.137931
    TB    2019.956522   84.852174
    ARI   2020.157895   86.328947
    DEN   2020.235294   86.529412
    HOU   2020.031250   88.459375
    LV    2020.235294   88.485294
    BUF   2020.567568   89.300000
    NYG   2020.030303   92.521212
    JAX   2020.655172   92.555172
    WAS   2019.878788   92.796970
    CHI   2020.171429   92.877143
    DET   2019.594595   94.635135
    NE    2019.549020   97.233333
    IND   2019.694444  100.530556
    MIA   2020.344828  101.741379
    NYJ   2020.333333  106.643333
    FA    2023.000000  128.618750
                               _id  year             name position team    adp  \
    1007  64f3c96b9f6a10f940744caf  2023    Ja'Marr Chase       WR  CIN    3.8   
    866   64f3c96b9f6a10f940744c22  2022    Ja'Marr Chase       WR  CIN    8.7   
    9     64f3c96b9f6a10f9407448c9  2017       A.J. Green       WR  CIN   10.0   
    867   64f3c96b9f6a10f940744c23  2022        Joe Mixon       RB  CIN   10.1   
    513   64f3c96b9f6a10f940744ac1  2020        Joe Mixon       RB  CIN   13.1   
    349   64f3c96b9f6a10f940744a1d  2019        Joe Mixon       RB  CIN   18.4   
    693   64f3c96b9f6a10f940744b75  2021        Joe Mixon       RB  CIN   19.2   
    184   64f3c96b9f6a10f940744978  2018       A.J. Green       WR  CIN   21.3   
    185   64f3c96b9f6a10f940744979  2018        Joe Mixon       RB  CIN   23.3   
    1031  64f3c96b9f6a10f940744cc7  2023        Joe Mixon       RB  CIN   27.5   
    1034  64f3c96b9f6a10f940744cca  2023      Tee Higgins       WR  CIN   30.5   
    890   64f3c96b9f6a10f940744c3a  2022      Tee Higgins       WR  CIN   32.5   
    43    64f3c96b9f6a10f9407448eb  2017        Joe Mixon       RB  CIN   42.5   
    905   64f3c96b9f6a10f940744c49  2022       Joe Burrow       QB  CIN   46.1   
    382   64f3c96b9f6a10f940744a3e  2019       Tyler Boyd       WR  CIN   51.5   
    1058  64f3c96b9f6a10f940744ce2  2023       Joe Burrow       QB  CIN   52.8   
    738   64f3c96b9f6a10f940744ba2  2021      Tee Higgins       WR  CIN   61.3   
    397   64f3c96b9f6a10f940744a4d  2019       A.J. Green       WR  CIN   63.8   
    571   64f3c96b9f6a10f940744afb  2020       A.J. Green       WR  CIN   69.8   
    71    64f3c96b9f6a10f940744907  2017     Tyler Eifert       TE  CIN   70.4   
    752   64f3c96b9f6a10f940744bb0  2021    Ja'Marr Chase       WR  CIN   77.4   
    579   64f3c96b9f6a10f940744b03  2020       Tyler Boyd       WR  CIN   78.5   
    776   64f3c96b9f6a10f940744bc8  2021       Tyler Boyd       WR  CIN   97.3   
    787   64f3c96b9f6a10f940744bd3  2021       Joe Burrow       QB  CIN  112.4   
    1128  64f3c96b9f6a10f940744d28  2023       Tyler Boyd       WR  CIN  122.4   
    984   64f3c96b9f6a10f940744c98  2022       Tyler Boyd       WR  CIN  123.0   
    126   64f3c96b9f6a10f94074493e  2017      Andy Dalton       QB  CIN  126.5   
    134   64f3c96b9f6a10f940744946  2017      Jeremy Hill       RB  CIN  134.5   
    637   64f3c96b9f6a10f940744b3d  2020       Joe Burrow       QB  CIN  135.7   
    300   64f3c96b9f6a10f9407449ec  2018  Giovani Bernard       RB  CIN  140.9   
    306   64f3c96b9f6a10f9407449f2  2018     Tyler Eifert       TE  CIN  147.0   
    309   64f3c96b9f6a10f9407449f5  2018        John Ross       WR  CIN  152.0   
    1171  64f3c96b9f6a10f940744d53  2023    Irv Smith Jr.       TE  CIN  152.7   
    159   64f3c96b9f6a10f94074495f  2017  Giovani Bernard       RB  CIN  159.6   
    
          PlayerID  
    1007  ChasJa00  
    866   ChasJa00  
    9     GreeA.00  
    867   MixoJo00  
    513   MixoJo00  
    349   MixoJo00  
    693   MixoJo00  
    184   GreeA.00  
    185   MixoJo00  
    1031  MixoJo00  
    1034  HiggTe00  
    890   HiggTe00  
    43    MixoJo00  
    905   BurrJo01  
    382   BoydTy00  
    1058  BurrJo01  
    738   HiggTe00  
    397   GreeA.00  
    571   GreeA.00  
    71    EifeTy00  
    752   ChasJa00  
    579   BoydTy00  
    776   BoydTy00  
    787   BurrJo01  
    1128  BoydTy00  
    984   BoydTy00  
    126   DaltAn00  
    134   HillJe01  
    637   BurrJo01  
    300   BernGi00  
    306   EifeTy00  
    309   RossJo00  
    1171  SmitIr01  
    159   BernGi00  



```python

```
