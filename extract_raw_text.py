import pandas as pd
import os
import matplotlib.pyplot as plt

file = open('/Users/macintoshhd/thesis_recommendation/raw_text.txt', 'r')
data = file.readlines()
loss = []

for d in data:
    try:
        loss.append(d.split('=')[1].replace('\n', '').replace(';', ''))
    except IndexError:
        pass

iteration = [a for a in range(1, len(loss)+1)]

# dict = {'loss' : loss, "iteration" : iteration}
# df = pd.DataFrame(dict)
# df.to_csv('/Users/macintoshhd/thesis_recommendation/cf.csv')
#

x = [float(float(x)/100)+0.7 for x in loss]

# plt.xlabel("Iterations")
# plt.ylabel("Mean Square Error")
# plt.plot(x)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x)
ax.set_ylabel('MSE loss')
ax.set_xlabel('Iteration')
# ax.grid()
ax.set_title('Collaborative Filtering loss over time')
plt.show()
fig.savefig('Collaborative.png')

