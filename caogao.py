import matplotlib.pyplot as plt
train = [3,5,7,6,8,2]
val = [2,4,7,9,5,1]
plt.plot(train, label='training_error')
plt.plot(val, label='validation_error') 
plt.xlabel("Epoch")
plt.ylabel("MSE Error")
plt.title("Error vs Epoch")
plt.legend()
plt.show()
