import numpy as np


print("np:",np.__version__)
list1=np.array([1, 2, 3, 4, 5])
list2=np.array([10, 20, 30, 40, 50])

list3=list1+list2

list4=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]

list4=np.reshape(list4,(4,3))

print(list4)
print("List3:", list3)
print(np.sum(list2))

print(20 in list2)
print(np.where(list2 == 20))
print(list1[1:3])

