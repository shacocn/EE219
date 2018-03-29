from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt

categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
train_data = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)

length = []
data = []
index = []

for m in range(8):
	temp_index = []
	temp_index.append(list (np.where(train_data.target==m))[0])
	index.append(temp_index)
	temp_data = []
	for n in index[m][0]:
		temp_data.append(train_data.data[n])
	data.append(temp_data)
	length.append(len(temp_data)) 


plt.figure()
plt_index = range(8)
width = 1
color = ['b','b','b','b','g','g','g','g']
categ = plt.bar(plt_index, length, width, color=color)
label_index = np.arange(0,8,1).tolist() 

plt.xticks (label_index, ('gra','win','pc','mac','autos','motor','base','hock'))
plt.ylim(300,700)
plt.ylabel('Training Documents')
plt.title('Training Documents v.s. Class')
plt.legend((categ[1], categ[5]), ('Computer Technology', 'Recreational Activity'), loc = 'upper right')
plt.show()
