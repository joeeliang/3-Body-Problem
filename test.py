import numpy as np

arr1 = np.empty((0,2,2))
arr2 = np.random.rand(2, 2, 2)

print(arr1)
print(arr2)

print("add")

combined_arr = np.concatenate((arr1, arr2), axis=0)
print(combined_arr)  # (6, 4, 5)