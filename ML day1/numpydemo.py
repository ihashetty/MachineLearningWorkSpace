import numpy as np

vector=np.array([1,2,3,4]) # 1 D numeric numpy array
print("type of vector:", type(vector))
print("shape of vector:", vector.shape)

vector1=np.array([[1,2,3],[10,203,11]]) # 2 D numeric numpy array
print("type of vector1:", type(vector1))
print("shape of vector1:", vector1.shape)

vector3=np.array([[[1,2,3],[10,203,11]],[[11,21,31],[10,22,111]],[[11,21,31],[10,22,111]]]) # 3 D numeric numpy array
print("type of vector3:", type(vector3))
print("shape of vector3:", vector3.shape)

vector33=np.array([[1,2,3,4],[1,2,3,4]])
print("axis 0  sum:",vector33.sum(axis=0))

print("axis 1  sum:",vector33.sum(axis=1))

np.random.seed(32)
x=np.random.rand(3)
print("random numbers:",x)