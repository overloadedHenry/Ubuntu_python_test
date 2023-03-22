import pandas as pd
import numpy as np

ser = pd.Series(["zero", "one", "two"])
# print(ser)
#
# print(ser.values)
# print(ser.index)
# print(list(ser.index))

Ser = pd.Series(np.arange(10), index = list("0123456789"))
# print(Ser)
#
# print(Ser[::-1])



