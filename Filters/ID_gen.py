import pandas as pd
import string
import random
import warnings
import os
import pickle

class GenID (object):
  
  def __init__(self, creator):

    self.creator = creator
    self.ID_dict = 'pickle/' + creator + '_ID_dict.p'

    if not os.path.exists(self.ID_dict):
      self.dataset_master = {} # Stores all datasetID references
    else:
      self.dataset_master = pickle.load(open(self.ID_dict, "rb"))
    self.code = ""
  
  def randomID(self, name = string):
    name = name.replace(" ", "")
    return ''.join(random.sample(name, 3)).upper()
  
  
  def generate(self, dataset_name, df, header_exists=True, header=None): 
    # df is input data in pd dataframe -- ie. ("Andrea", "IMDB Movie Review", df01)
    # header for the dataframe if it doesn't exist in your df yet -- input example: ["Sentences", "Tok Sentences", .....] (depending on your data)
        
    if not header_exists: # add header if current df doesn't have one
      df.columns = header
      
    if set(["Data ID"]).issubset(df.columns): # remove Data ID column if there's one already
      warnings.warn("Data ID column already exists. Replacing the column.")
      df = df[df.columns[df.columns!="Data ID"]]
      
    
    creatorID = self.creator[:2].upper() # Andrea -> AN
    
    datasetID = dataset_name.replace(" ", "")[:3].upper() # IMDB Movie Review -> IMD
    
    while(datasetID in self.dataset_master): # if duplicating ID in dataset_master list, generate random ID code
      warnings.warn("DatasetID already exists. Generating a new ID at random.")
      datasetID = self.randomID(dataset_name)
    
    idx = ["-".join([creatorID, datasetID, str(i).zfill(8)]) for i in range(len(df))] # AND-IMD-00000001
      
    df.insert(0, "Data ID", idx , True)
    
    self.dataset_master[datasetID] = dataset_name
    self.code = "-".join([creatorID, datasetID])

    if not os.path.exists('pickle'):
      os.mkdir('pickle')

    file = open(self.ID_dict, 'wb')
    pickle.dump(self.dataset_master, file)
    file.close()

    print("ID for this dataset ({}): {}".format(dataset_name, self.code))
    
    return df
