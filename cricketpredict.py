import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn import preprocessing,svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

def create_cric_df():
    cric_df = pd.read_csv('ContinousDataset.csv')
    cric_df = cric_df[['Team 1','Team 2','Match Date','Venue_Team1','Venue_Team2','Winner']]
##        print(cric_df[['Team 1','Team 2','Match Date','Winner']].tail())
##        print("")
##        print(cric_df[['Team 1','Team 2','Match Date','Winner']].head())
    cric_df['Last10MatchesTeam1'] = np.nan  #Stores no. of matches won
    cric_df['Last10MatchesTeam2'] = np.nan  #Stores no. of matches won
    cric_df['Last10ConflictsTeam1'] = np.nan #Same competitors #Stores no. of matches won
    cric_df['Last10ConflictsTeam2'] = np.nan #Same competitors #Stores no. of matches won
    cric_df['label'] = np.nan #0 for first team win and 1 for second

    cric_df.rename(columns = {'Match Date':'Date'}, inplace = True)
    regex = re.compile(r'-\d{1,2}', re.IGNORECASE)
    for index,row in cric_df.iterrows():
        date = row['Date']
        #regex = re.findall(r'-\d{1,2}',date)
        date = regex.sub('',date)
        #print(date)
        cric_df.loc[index,'Date'] = datetime.strptime(date,'%b %d, %Y') 
    
    cric_df['Date'] = pd.to_datetime(cric_df.Date,format='%b %d, %Y')
    #print(cric_df['Team 1'].value_counts())
    #print(cric_df['Team 2'].value_counts())

    for index,row in cric_df.iterrows():

        team1_df = cric_df.loc[(cric_df['Date'] < row['Date']) & ((cric_df['Team 1'] == row['Team 1']) | (cric_df['Team 2'] == row['Team 1']))].copy()
        team1_df.sort_values(by='Date',ascending=False,inplace=True)
        n = 0
        if len(team1_df.index) >=10:
            n=10
        else:
            n = len(team1_df.index)
        if n > 0:
            team1_df = team1_df[:n]
            
            wins = 0
            for i,r in team1_df.iterrows():
                
                if r['Winner'] == row['Team 1']:
                    wins+=1
            #print(wins)            
            cric_df.loc[index,'Last10MatchesTeam1'] = wins        
        

        team2_df = cric_df.loc[(cric_df['Date'] < row['Date']) & ((cric_df['Team 1'] == row['Team 2']) | (cric_df['Team 2'] == row['Team 2']))].copy()
        team2_df.sort_values(by='Date',ascending=False,inplace=True)

        n = 0
        if len(team2_df.index) >=10:
            n = 10
        else:
            n = len(team2_df.index)

        if n > 0 :
            team2_df = team2_df[:n]
            wins = 0
            
            for i,r in team2_df.iterrows():
                if r['Winner'] == row['Team 2']:
                    wins+=1

            #print(wins)
            cric_df.loc[index,'Last10MatchesTeam2'] = wins

        teamvs_df = cric_df.loc[(cric_df['Date'] < row['Date']) &
                                (((cric_df['Team 1'] == row['Team 1']) & (cric_df['Team 2'] == row['Team 2'])) |
                                                                   ((cric_df['Team 1'] == row['Team 2']) & (cric_df['Team 2'] == row['Team 1'])))].copy()
        teamvs_df.sort_values(by='Date',ascending=False,inplace=True)

        n = 0
        if len(teamvs_df.index) > 10:
            n = 10
        else:
            n = len(teamvs_df.index)

        if n > 0:
            teamvs_df = teamvs_df[:n]
            wins = 0
            for i,r in teamvs_df.iterrows():
                if r['Winner'] == row['Team 1']:
                    wins+=1

            #print(wins)
            cric_df.loc[index,'Last10ConflictsTeam1'] = wins
            cric_df.loc[index,'Last10ConflictsTeam2'] = n-wins

        if row['Winner'] == row['Team 1']:
            cric_df.loc[index,'label'] = 0
        else:
            cric_df.loc[index,'label'] = 1

        if row['Venue_Team1'] == 'Home':
            cric_df.loc[index,'Venue_Team1'] = 0
        elif row['Venue_Team1'] == 'Away':
            cric_df.loc[index,'Venue_Team1'] = 1
        else:
            cric_df.loc[index,'Venue_Team1'] = 2

        if row['Venue_Team2'] == 'Home':
            cric_df.loc[index,'Venue_Team2'] = 0
        elif row['Venue_Team2'] == 'Away':
            cric_df.loc[index,'Venue_Team2'] = 1
        else:
            cric_df.loc[index,'Venue_Team2'] = 2
            
  
    
    #cric_df.sort_values(by='Date',ascending=False,inplace=True)
    #print(cric_df[['Team 1','Team 2','Winner','Date']].head())
    #print(cric_df[['Team 1','Team 2','Winner','Date']].tail())
    cric_df.fillna(0,inplace=True)
    cric_df.to_csv('main_cric_csv.csv')

def get_cric_df():
    cric_df = pd.read_csv('main_cric_csv.csv')
    return cric_df

def create_team_df():
    cric_df = get_cric_df()
    #print(cric_df['Team 1'].unique())
    team_df = pd.DataFrame(data = cric_df['Team 1'].unique(),columns=['Team_Name'])
##    print(team_df)

    team_df.to_csv('Team_csv.csv')

def get_team_df():
    team_df = pd.read_csv('Team_csv.csv')
    return team_df


def main():
    #create_cric_df()
    
    #create_team_df()

    cric_df = get_cric_df()
    cric_df.set_index('Date',inplace=True)
    #print(cric_df.columns)
    X = np.array(cric_df.drop(['label','Winner','Team 1','Team 2'],axis = 1))
    y = np.array(cric_df['label'])

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05)

    # clf = svm.SVC()
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    accuracy = clf.score(X_test,y_test)

    print(accuracy)

    

main()
