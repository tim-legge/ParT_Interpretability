import numpy as np

testing_list = [0,1,2,3,4]
print(sum(testing_list))

testing_np_list = [np.array([[10,9,8],[4,5,1]]), np.array([[10,9,8],[1,7,5]])]
print(sum(testing_np_list))

hist, bin_edges = np.histogram(np.zeros(1), bins=50, range=(-2.4, 6.3))
print(bin_edges)

lp_bin_edges = np.linspace(-2.4, 6.3, num=49)
print(lp_bin_edges)