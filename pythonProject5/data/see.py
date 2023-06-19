import pandas as pd
import scipy.io as sio
import numpy as np

data = sio.loadmat('20newsgroups.mat')
print(data['data'][0][0])
x1 = data['data'][0][0]
x2 = data['data'][0][1]
x3 = data['data'][0][2]
# x1 = data['bbc'].toarray()
# x2 = data['guardian'].toarray()
# x3 = data['reuters'].toarray()
# # y = data['truth']
#
print(x1.shape)
print(x2.shape)
print(x3.shape)
#
view1 = []
view2 = []
view3 = []
for v1 in range(0, 2000):
    view1.append('a' + str(v1))
for v2 in range(0, 2000):
    view2.append('b' + str(v2))
for v3 in range(0, 2000):
    view3.append('c' + str(v3))
#
x1 = pd.DataFrame(x1.T,columns=view1)
x2 = pd.DataFrame(x2.T,columns=view2)
x3 = pd.DataFrame(x3.T,columns=view3)
#
print(x1)
# y = pd.DataFrame(y)
x = pd.concat([x1,x2,x3],axis=1)
print(x)
x.to_csv('newsgroups.csv', index = False)

