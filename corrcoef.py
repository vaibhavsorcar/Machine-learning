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