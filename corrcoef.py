"""
import numpy as np
x = [2, 1]
y = [1, 2]
matrix = np.corrcoef(x, y)
print(matrix)

if np.std(x) == 0 or np.std(y) == 0:
    print("One of the inputs is constant; correlation is undefined.")
else:
    matrix = np.corrcoef(x, y)
    print(matrix)
"""

"""
import numpy as np
import pandas as pd
data = {
    'x': [45, 37, 42, 35, 39],
    'y': [38, 31, 26, 28, 33],
    'z': [10, 15, 17, 21, 12]
}
df = pd.DataFrame(data, columns = ['x', 'y', 'z'])
print("DataFrame is: \n")
print(df)
matrix = df.corr()
print(matrix)

"""


import numpy as np
from sklearn import datasets
import pandas as pd
data = datasets.load_iris()
df = pd.DataFrame(data = data.data, columns = data.feature_names)
df['target'] = data.target
corr_matrix = df.drop(columns='target').corr()
print("Correlation Matrix:\n", corr_matrix)
corr_unstacked = corr_matrix.abs().unstack()
corr_unstacked = corr_unstacked[corr_unstacked < 1]  # remove diagonal/self correlations
max_corr_pair = corr_unstacked.idxmax()
max_corr_value = corr_unstacked.max()
print(f"\nMost highly correlated features: {max_corr_pair} with correlation {max_corr_value:.3f}")
