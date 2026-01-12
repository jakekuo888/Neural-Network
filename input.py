import main
import numpy as np

nn = main.Neural_net([1,2,3,4])
test_ = np.array([0.1])
print(main.sMax(nn.forward(test_)))