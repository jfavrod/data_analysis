import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Bug fix for display formats to ovoid run time errors.
pd.set_option('display.float_format', lambda x: '%f'%x)


# Read in data.
data = pd.read_csv('addhealth_pds.csv', low_memory=False)


data['H1TO30'] = pd.to_numeric(data['H1TO30'])
data['H1TO37'] = pd.to_numeric(data['H1TO37'], errors='coerce')
data['H1TO40'] = pd.to_numeric(data['H1TO40'], errors='coerce')


"""
Some Data Management
Creating new variables for questions where 0 means no, and 1 - 18
is an age in which the activity occured.
"""

tried_index = {0:'Not Tried', 1:'Have Tried'}

data.loc[:,'H1ED7'] = data.loc[:,'H1ED7'].replace([' ', '6','8'], np.nan)
data['H1ED7'] = data['H1ED7'].dropna()
data['H1ED7'] = data['H1ED7'].map({'0':'Not Suspended', '1':'Suspended'})

data.loc[:,'H1ED9'] = data.loc[:,'H1ED9'].replace([' ', '6','8'], np.nan)
data['H1ED9'] = data['H1ED9'].dropna()
data['H1ED9'] = data['H1ED9'].map({'0':'Not Expelled', '1':'Expelled'})

data.loc[:,'H1TO1'] = data.loc[:,'H1TO1'].replace([6,8,9], np.nan)
data['H1TO1'] = data['H1TO1'].dropna()
data['H1TO1'] = data['H1TO1'].map(tried_index)

data.loc[:,'H1TO12'] = data.loc[:,'H1TO12'].replace([6,8,9], np.nan)
data['H1TO12'] = data['H1TO12'].dropna()
data['H1TO12'] = data['H1TO12'].map(tried_index)

data.loc[:,'TRYPOT'] = pd.cut(data.H1TO30, [-1,0,18])
data.loc[:,'TRYPOT'] = data.loc[:,'TRYPOT'].replace([96,98,99], np.nan)
data['TRYPOT'] = data['TRYPOT'].dropna()

data.loc[:,'TRYCOKE'] = pd.cut(data.H1TO34, [-1,0,18])
data.loc[:,'TRYCOKE'] = data.loc[:,'TRYCOKE'].replace([96,98,99], np.nan)
data['TRYCOKE'] = data['TRYCOKE'].dropna()

data.loc[:,'TRYINH'] = pd.cut(data.H1TO37, [-1,0,18])
data.loc[:,'TRYINH'] = data.loc[:,'TRYINH'].replace([96,98,99], np.nan)
data['TRYINH'] = data['TRYINH'].dropna()

data.loc[:,'TRYHARD'] = pd.cut(data.H1TO40, [-1,0,18])
data.loc[:,'TRYHARD'] = data.loc[:,'TRYHARD'].replace([96,98,99], np.nan)
data['TRYHARD'] = data['TRYHARD'].dropna()

data.loc[:,'PROPD'] = pd.cut(data.H1DS2, [-1,0,5])
data.loc[:,'PROPD'] = data.loc[:,'PROPD'].replace([6,8,9], np.nan)
data['PROPD'] = data['PROPD'].dropna()

data.loc[:,'SHOPL'] = pd.cut(data.H1DS4, [-1,0,5])
data.loc[:,'SHOPL'] = data.loc[:,'SHOPL'].replace([6,8,9], np.nan)
data['SHOPL'] = data['SHOPL'].dropna()

data.loc[:,'FIGHT'] = pd.cut(data.H1DS5, [-1,0,5])
data.loc[:,'FIGHT'] = data.loc[:,'FIGHT'].replace([6,8,9], np.nan)
data['FIGHT'] = data['FIGHT'].dropna()

data.loc[:,'SELLD'] = pd.cut(data.H1DS12, [-1,0,5])
data.loc[:,'SELLD'] = data.loc[:,'SELLD'].replace([6,8,9], np.nan)
data['SELLD'] = data['SELLD'].dropna()

data.loc[:,'WEPVI'] = pd.cut(data.H1FV8, [-1,0,2])
data.loc[:,'WEPVI'] = data.loc[:,'WEPVI'].replace([6,8,9], np.nan)
data['WEPVI'] = data['WEPVI'].dropna()

data.loc[:,'SUATM'] = pd.cut(data.H1SU2, [-1,0,5])
data.loc[:,'SUATM'] = data.loc[:,'SUATM'].replace([6,7,8,9], np.nan)
data['SUATM'] = data['SUATM'].dropna()


question = {
'H1ED7' : 'Have you ever received an out-of-school suspension from school?',
'H1ED9' : 'Have you ever been expelled from school?',

'H1TO1' : 'Have you ever tried cigarette smoking, even just 1 or 2 puffs?',
'H1TO12' : 'Have you had a alcohol more than 2 or 3 times in your life?',
'TRYPOT' : 'Have you tried marijuana?',
'TRYCOKE' : 'Have you tried cocaine?',
'TRYINH' : 'Have you tried inhalants?',
'TRYHARD' : 'Have you ever tried hard drugs like LSD, PCP, ecstasy, mushrooms\n\
speed, ice, heroin, or pills without a doctor’s prescription?',

'PROPD' : 'In the past 12 months, how often did you deliberately damage\n\
property that didn’t belong to you?',
'SHOPL' : 'Have you ever shoplifted?',
'FIGHT' : 'Do you often get into serious physical fights?',
'SELLD' : 'Do you sell drugs?',

'WEPTH' : 'Have you threatened someone with a gun or knife in the past 12 months?',
'WEPVI' : 'Have you shot or stabbed someone in the past 12 months?',

'H1SU1' : 'During the past 12 months, did you seriously think about suicide?',
'SUATM' : 'During the past 12 months, did you attempt suicide?'
}


NO_RELIGION = 0
UNIMPORTANT_REL = 4
REFUSE_IMPORTANT_REL = 6
DONT_KNOW_IMPORTANT_REL = 8
REFUSE_REL = 96
DONT_KNOW_REL = 98
NA_REL = 99


"""
Religious and non-religious subsets.
"""

non_religious = data[
    (data['H1RE1'] == NO_RELIGION) |
    (data['H1RE1'] == DONT_KNOW_REL)
]

religious = data[
    (data['H1RE1'] != NO_RELIGION) &
    (data['H1RE1'] != DONT_KNOW_REL) &
    (data['H1RE1'] != REFUSE_REL) &
    (data['H1RE1'] != UNIMPORTANT_REL) &
    (data['H1RE4'] != REFUSE_IMPORTANT_REL)
]


def dist(var):
    dist = data.loc[:,var].value_counts(sort=True, normalize=True)
    dist.index = ['pos', 'neg:']
    print(dist * 100)


def desc(var):
    data[var] = data[var].astype('category')
    print(data[var].describe())


def univarGraph(var):
    counts = data[var].value_counts(normalize=True)
    counts.index = ['pos', 'neg']
    counts.plot(kind='bar', title=question[var])
    plt.show()


def bivarGraph(var):
    nrel_counts = non_religious[var].value_counts(normalize=True)
    rel_counts = religious[var].value_counts(normalize=True)
    
    nrel_counts.index = ['pos', 'neg']
    rel_counts.index = ['pos', 'neg']

    nrel_neg = nrel_counts.to_dict()['neg']*100
    rel_neg = rel_counts.to_dict()['neg']*100
                                              
    suspend = np.array([nrel_neg, rel_neg])
    df = pd.DataFrame(suspend, index=['Non Religious', 'Religious'])
    df.plot(kind='bar', legend=False, title=question[var])
    plt.show()


vars = ['H1ED7', 'H1ED9', 'H1TO1', 'H1TO12', 'TRYPOT', 'TRYCOKE', 'TRYINH',
'TRYHARD', 'PROPD', 'SHOPL', 'FIGHT', 'SELLD', 'WEPVI', 'SUATM']


"""
Total number of participants.
"""
print('-----------------------------------------')
print('# of participants: %d' % (data.shape)[0])
print('-----------------------------------------')
print('')


"""
Total number in non-religious group.
"""
num_non_religious = (non_religious.shape)[0]
print('-----------------------------------------')
print('# of non-religious individuals: %d' % num_non_religious)
print('-----------------------------------------')
print('')


"""
Total number in non-religious group.
"""
num_religious = (religious.shape)[0]
print('-----------------------------------------')
print('# of religious individuals: %d' % num_religious)
print('-----------------------------------------')
print('')



for var in vars:
    print('Analysis for "' + question[var] + '"')
    print('Distribution:')
    dist(var)
    print()
    
    print('Discription:')
    desc(var)
    print()


for var in vars:
    univarGraph(var)


for var in vars:
    bivarGraph(var)

