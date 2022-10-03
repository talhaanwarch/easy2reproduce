# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:02:41 2022

@author: chtal
"""

import pandas as pd

file='export_legalannotationprojectowner_6f5344738f1fb24d3b465aa5.manifest'
with open(file,'r') as f:
  txt=f.readlines()

def create_df(idx,exact_conll=False):  
    sample_1=eval(txt[idx])['annotations']
    tags=sample_1[0]['content']['tags']#NER
    text=sample_1[0]['content']['plainText']['1']#plan text

    #let get all ranges and tags
    ranges=[i['range'] for i in tags]
    types=[i['type'].replace(' ','-') for i in tags]
    
    li2=[i[1] for i in ranges]
    li1=[i[0] for i in ranges]
    
    #get all ranges with no ner type
    l2=[[li2[i],li1[1+i]] for i in range(len(li1)-1)]
    #range did not start from zero nor end at the text end, so i have to add 'other' at these two edges
    if ranges[0][0]!=0:
      l2.insert(0,[0,ranges[0][0]])
    if ranges[-1][1]!=len(text):
      l2.append([ranges[-1][1],len(text)])
      
    #add ner type with the range
    for i,j in zip(ranges,types):
      i.append(j)
    other=[i.append('Other') or i for i in l2]
      
    
    df=pd.DataFrame(ranges+other,columns=['start','end','cat'])
    df.sort_values('start',inplace=True)
    df.reset_index(inplace=True,drop=True)
    
    #convert ranges to text    
    text_list=[]
    for _,i in df.iterrows():
      text_list.append(text[i.start:i.end])
    
    df=pd.DataFrame(zip(text_list,df.cat.to_list()),columns=['text','ner'])

    #there are some strings, which are v lengthy and for bert it will be difficult to read such lengthy text, so i will split them
    n=10#no of words in each row
    df=df.set_index(['ner'])['text'].str.split().apply(
                   lambda x: pd.Series([' '.join(x[i:i+n]) for i in range(0, len(x), n)])
                ).stack().reset_index().drop('level_1',axis=1)
    df.columns=['ner','text']
    if exact_conll:
        return df
    else:
        return [df['text'].to_list(),df['ner'].to_list()]

df_list=[]
exact_conll=False #should be false for easy incorporation to huggingface

for i in range(len(txt)):
    try:
        df_list.append(create_df(i,exact_conll))
    except Exception as  e:
        print(i,e)


if exact_conll:
    df=pd.concat(df_list)
    #optional if want to add text id
    # idx=[[i]*len(j) for i,j in enumerate(df_list)]
    # idx = [item for sublist in idx for item in sublist]
    # df['idx']=idx

else:
    
    df=pd.DataFrame(df_list)
    df.columns=['tokens','ner_tags']




df.to_csv('dataframe.csv',index=False)
