import numpy as np
import os
import matplotlib.pyplot as plt

# add arrays element wise
# element wise division by num_arrays

STL = np.zeros(15000)
MTL12 = np.zeros(15000)
MTL13 = np.zeros(15000)
MTL14 = np.zeros(15000)


for idx, filename in enumerate(os.listdir('./')):
    if filename.startswith('STL'):
        STL1 = np.load(filename, allow_pickle = True)
        STL = np.add(STL, STL1)
    if filename.startswith('MTL12'):
        MTL121 = np.load(filename, allow_pickle = True)
        MTL121 = MTL121.item()
        MTL121 = np.array(MTL121[1][1])
        MTL12 = np.add(MTL12, MTL121)
    if filename.startswith('MTL13'):
        MTL131 = np.load(filename, allow_pickle = True)
        MTL131 = MTL131.item()
        MTL131 = np.array(MTL131[1][1])
        MTL13 = np.add(MTL13, MTL131)
    if filename.startswith('MTL14'):
        MTL141 = np.load(filename, allow_pickle = True)
        MTL141 = MTL141.item()
        MTL141 = np.array(MTL141[1][1])
        MTL14 = np.add(MTL14, MTL141)

STL = STL / 2
MTL12 = MTL12 / 5
MTL13 = MTL13 / 5
MTL14 = MTL14 / 5

print(STL)
print(MTL12)
print(MTL13)
print(MTL14)

np.save('./average1', STL)
np.save('./average12', MTL12)
np.save('./average13', MTL13)
np.save('./average14', MTL14)

x = [i for i in range(15000)]
x = x[::10]

plt.figure(figsize=(8, 8), dpi=80)
plt.title("compare accuracy")
plt.plot(x, STL[::10], label="STL", color='purple', linewidth=1.5)
plt.plot(x, MTL12[::10], label="MTL12", color="orange", linewidth=1.5)
plt.plot(x, MTL13[::10], label="task_1_and_3", color='dodgerblue', linewidth=1.5)
plt.plot(x, MTL14[::10], label="task_1-4", color='red', linewidth=1.5)
plt.legend(frameon=False, prop={'size': 22})
plt.xlabel('Epochs')
plt.ylabel('% correct') 
plt.legend(frameon=False)
plt.tick_params(axis='both', which='major', labelsize=17)
plt.show()

