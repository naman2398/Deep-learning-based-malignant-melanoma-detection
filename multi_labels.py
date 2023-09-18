import pandas as pd
import numpy as np

data = pd.read_csv("../trainLabels.csv")

labels = np.zeros([data.shape[0], 5], dtype=np.uint8)

for i in range(data.shape[0]):
  level = data.iloc[i].level
  for j in range(5):
    if (j<level):
      labels[i][j] = 1

np.save("labels", labels)
