import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from collections import Counter # Keep track of our term counts
import seaborn as sns

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    ) 

def survive_and_feature(feature):
    #feature_data=str(np.zeros(len(data_train)))
    feature_data=data_train[feature].values   
    feature_count=Counter(feature_data)
    index=np.argsort(np.array(feature_count.keys()))
    feature_survive=list([])
    for i in range(0,len(feature_count )):
        print i,feature_count.keys()[index[i]]
        temp=np.nansum(data_train['Survived'].values[feature_data==feature_count.keys()[index[i]]])
        feature_survive.append(temp)
    
    feature_survive_rate=list(1.0*np.array(feature_survive)/(np.array(feature_count.values())[index])) 
    ind = np.arange(len(feature_count))    # the x locations for the groups
    width = 0.35
    #plt.figure(2)
    
    feature_elem=np.array(feature_count.keys())[index]
    if feature=='Sex':
        feature_elem=('male','female')
    plt.bar(ind,feature_survive_rate,width)
    plt.xticks(ind, feature_elem)
    plt.xticks(rotation=90)
    plt.title(feature+' vs Survive')
    plt.ylabel('survive rate')
    plt.xlabel(feature)
   # plt.legend((p1[0], p2[0]), ('survived', 'not survived'))
    plt.show()

def Name_surffix(name):
    name=name.split()
    name_s=[]
    for n in range(0,len(name)):
        if ('.' in name[n]) & (len(name[n])>2):
            name_s.append(name[n])
    name_s=''.join(name_s)
    return name_s
def Cabin_firstletter(cabin):
        
        if ( type(cabin[0])==str ):
            return cabin[0][0] 
        else:
            return cabin[0]
            
def Ticket_group(ticket):
    
        if ( ticket.isdigit()):
            return 100
        else:
            ticket0=ticket.split(' ')
            return ticket0[0]


def Age_group(age):
        
        if (age>=0) & (age<1):
            return 0.5
        if (age>=1) & (age<5):
            return 3   
        if (age>=5) & (age<65 ):
            return 10  
        if (age>=15) & (age<25):
            return 20  
        if (age>=25) & (age<35):
            return 30  
        if (age>=35) & (age<45):
            return 40  
        if (age>=45) & (age<55):
            return 50  
        if (age>=55) & (age<65):
            return 60  
        if (age>=65) :
            return 70  
def Fare_group(fare):
        return int(fare/20)*20+10         

#modify training data     
data_train = pd.read_csv("D:/Kaggle/Titanic/train.csv")
data_train = data_train.fillna(-1)    
[nperson,nterm]=data_train.shape
data_train.info()
headers=pd.read_csv('D:/Kaggle/Titanic/train.csv', header=None,nrows=1)


#modify variables
#modify Sex, set female=1 and male=0
data_train['Sex'][data_train['Sex']=='female']=1 #change gender to number
data_train['Sex'][data_train['Sex']=='male']=0
data_train['Sex']=data_train['Sex'].astype(float)

for i in range(0,len(data_train)):
    print i
    data_train['Name'][i]=Name_surffix(data_train['Name'][i]) #modify name
    data_train['Cabin'][i]=Cabin_firstletter([data_train['Cabin'][i]]) #modify Cabin
    data_train['Ticket'][i]=Ticket_group(data_train['Ticket'][i] ) #modify Ticket
    data_train['Age'][i]=Age_group(data_train['Age'][i]) #modify Ticket
    data_train['Fare'][i]=Fare_group(data_train['Fare'][i]) #modify 
data_train = data_train.fillna(-1)
data_train.to_csv( 'titanic_train_modified.csv' , index = False )

plt.figure(1)
plot_correlation_map( data_train )
plt.show()

#modify testing data
data_test = pd.read_csv("D:/Kaggle/Titanic/test.csv")
data_test = data_test.fillna(-1)    
data_test.info()

data_test['Sex'][data_test['Sex']=='female']=1 #change gender to number
data_test['Sex'][data_test['Sex']=='male']=0
data_test['Sex']=data_test['Sex'].astype(float)

for i in range(0,len(data_test)):
    print i
    data_test['Name'][i]=Name_surffix(data_test['Name'][i]) #modify name
    data_test['Cabin'][i]=Cabin_firstletter([data_test['Cabin'][i]]) #modify Cabin
    data_test['Ticket'][i]=Ticket_group(data_test['Ticket'][i] ) #modify Ticket
    data_test['Age'][i]=Age_group(data_test['Age'][i]) #modify Ticket
    data_test['Fare'][i]=Fare_group(data_test['Fare'][i]) #modify 
data_test = data_test.fillna(-1)
data_test.to_csv( 'titanic_test_modified.csv' , index = False )


#relative survive rate for each category
#survive_and_feature('Pclass')
#survive_and_feature('Name')
#survive_and_feature('Sex')
#survive_and_feature('Age')
#survive_and_feature('SibSp')
#survive_and_feature('Parch')
#survive_and_feature('Ticket')
#survive_and_feature('Fare')
#survive_and_feature('Cabin')
#survive_and_feature('Embarked')









