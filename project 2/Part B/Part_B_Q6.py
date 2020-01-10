from Part_B_text_classifier import textClassifier, get_title
import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import time
import datetime
from tqdm import tqdm

# Question 6
def qns6a():
  celltypeList = ['RNN', 'LSTM']
  idtypeList = ['char', 'word']
  with_dropout = False
  networktype = "RNN"
  num_layer = 1
  gradient_clipping = False
  for celltype in celltypeList:
    for idtype in idtypeList:
      title = get_title(idtype,networktype, with_dropout, cell_type = celltype, num_layers=num_layer, gradient_clipping=gradient_clipping)
      print(title)
      tf.reset_default_graph() 
      textClassifier(idtype,networktype, with_dropout, cell_type = celltype, num_layers=num_layer, gradient_clipping=gradient_clipping)
      downloadfile(title)
  print("Question 6a done")

def qns6b_6c():
  num_layerList = [1,2]
  gradient_clippingList = [False,True]
  idtypeList = ['char', 'word']
  num=0
  with_dropout = False
  networktype = "RNN"
  celltype = 'GRU'
  for num_layer in num_layerList:
    for gradient_clipping in gradient_clippingList:
      for idtype in idtypeList:
        title = get_title(idtype,networktype, with_dropout, cell_type = celltype, num_layers=num_layer, gradient_clipping=gradient_clipping)
        print(title)
        num+= 1
        if num <= 2:
          print("Done in question 1 to 5") # done in question 3 and 4 already
          continue
        elif num >= 6:# the rest not required by assigment
          print("Question 6b and 6c done")
          return
        tf.reset_default_graph() 
        textClassifier(idtype,networktype, with_dropout, cell_type = celltype, num_layers=num_layer, gradient_clipping=gradient_clipping)
        downloadfile(title)
		
qns6a()
qns6b_6c()