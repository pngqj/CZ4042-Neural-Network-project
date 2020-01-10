from Part_B_text_classifier import textClassifier, get_title
import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import time
import datetime
from tqdm import tqdm

# Question 1 to 5
def qns5():
  idtypeList = ['char', 'word']
  networktypeList = ['CNN', 'RNN']
  with_dropoutList = [False,True]
  cell_type = "GRU"
  for idtype in idtypeList:
    for networktype in networktypeList:
      for with_dropout in with_dropoutList:
        tf.reset_default_graph() 
        title = get_title(idtype,networktype, with_dropout, cell_type = cell_type)
        print(title)
        textClassifier(idtype,networktype, with_dropout, cell_type = cell_type)
		
qns5()