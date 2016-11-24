predicted =[]
actual = []

with open("err_validate.txt","r") as f1:
    for line in f1:
        line = str(line)
        line.split()
        actual.append(line[:8])
        predicted.append(line[9:-1])


print predicted
print actual


import matplotlib.pyplot as plt
import numpy as np

l1, = plt.plot(actual)
l2, = plt.plot(predicted)

plt.xlabel('Iterations')
plt.legend((l1, l2), ('Train', 'Validation'), loc='upper right', )
plt.grid(True)
plt.savefig("./x.png")
plt.show()
