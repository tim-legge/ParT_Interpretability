import numpy as np

testing_list = [0,1,2,3,4]
print(sum(testing_list))

testing_np_list = [np.array([[10,9,8],[4,5,1]]), np.array([[10,9,8],[1,7,5]])]
print(sum(testing_np_list))
assert isinstance(sum(testing_np_list), np.ndarray), "Sum of numpy arrays is not a numpy array"