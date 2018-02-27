import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding,Flatten,Conv1D,MaxPool1D
from keras.preprocessing.text import Tokenizer
from  keras.preprocessing.sequence import pad_sequences
#from keras import optimizers


from os import listdir
import json

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# set parameters:
max_features = 5000
embedding_dims = 50
maxlen = 1000

max_words = 1000
batch_size = 32
epochs = 5
 

def readFile(fname,typeclass):

    
    docLabels = []
    docLabels = [f for f in sorted(listdir(fname)) if  f.endswith('.txt')]
    
    #create a list data that stores the content of all text files in order of their names in docLabels
    
    data = []
    Yn=[]
    for i,doc in enumerate(docLabels):
      data.append(open(fname + '/' + doc).read())
      #Yn.append(typeclass)
      Yn = np.append(Yn, [typeclass], axis=0)
    return data,Yn      




fname='/home/marta/marta/projInter/database1a/outlier'
data1,Yn1=readFile(fname,0)
Yn1=np.array(Yn1, dtype=np.int64)

fname='/home/marta/marta/projInter/database1a/normal'
data2,Yn2=readFile(fname,1)
Yn2=np.array(Yn2, dtype=np.int64)
data = data1+ data2

#tokenizer = nltk.RegexpTokenizer(r'\w+')
#stopword_set = set(stopwords.words('english'))



Yn=np.concatenate((Yn1, Yn2), axis=0)

from sklearn.model_selection import train_test_split
print('Spliting training and test set')


tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))



    


X_train1, X_test1, y_train, y_test = train_test_split(data, Yn, test_size=0.20, random_state=7)

X_train_index=tokenizer.texts_to_sequences(X_train1)
X_test_index=tokenizer.texts_to_sequences(X_test1)


print('Loading data...')



print(len(X_train_index), 'train sequences')
print(len(X_test_index), 'test sequences')


num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True)
	


x_train = pad_sequences(X_train_index, maxlen=maxlen, padding='post')
x_test = pad_sequences(X_test_index, maxlen=maxlen, padding='post')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)



print('Building model...')

model = Sequential()
model.add(Embedding(20000, 50, input_length=maxlen))



model.add(Conv1D(filters=100,kernel_size=2,  activation = 'relu')) 
model.add(Conv1D(filters=50,kernel_size=2)) 
model.add( MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
#sgd = optimizers.SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#tensorboard = TensorBoard(log_dir='logs/',  write_images=False,write_graph=True, write_grads=True, histogram_freq=1)


#history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs, verbose=1,validation_split=0.1,callbacks=[tensorboard])
history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs, verbose=1,validation_split=0.1)

score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])




# calculate predictions
predictions = model.predict(x_test)
proba = model.predict_proba(x_test)
Y_predict = model.predict_classes(x_test)
#Y_predict = [1 if (x==0) else 1 for x in Y_predict]
#y_test  = [0 if (x==1) else 1 for x in yn_test]


print('writing result file')



with open('txt_pred_percent.txt', 'w') as f:
    f.write(json.dumps(proba.tolist()))
    

with open('txt_pred_class.txt', 'w') as f1:
    f1.write(json.dumps(np.reshape(Y_predict, (Y_predict.shape[0],)).tolist()))


with open('txt_ground_truth.txt', 'w') as f2:
    f2.write(json.dumps(y_test.tolist()))
"""

    
#GRAPH------------------

from keras import backend as K



get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[1].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([x_test, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([x_test, 1])[0]


from sklearn.decomposition import PCA

#http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
#key: view PCA python
pca = PCA(n_components=3)
pca.fit(layer_output)
layer_output3 = pca.transform(layer_output)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



centers = [[1, 1], [-1, -1], [1, -1]]

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)


for name, label in [('normal', 0), ('outlier', 1)]:
    ax.text3D(layer_output3[y_test == label, 0].mean(),
              layer_output3[y_test == label,  1].mean() + 1.5,
              layer_output3[y_test == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y_test = np.choose(y_test, [2, 1]).astype(np.float)



ax.scatter(layer_output3[:, 0], layer_output3[:, 1], layer_output3[:, 2], c=y_test, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()



pca = PCA(n_components=2)
pca.fit(layer_output)
layer_output2 = pca.transform(layer_output)


# Percentage of variance explained for each components
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

plt.figure()
for c, i, target_name in zip("rgb", [1, 2],  ['normal','outlier']):
   plt.scatter(layer_output2[y_test==i,0], layer_output2[y_test==i,1], c=c, label= target_name)
   
   

#for i, txt in enumerate(layer_output2):
    #plt.annotate(i, (layer_output2[i,0],layer_output2[i,1]))
    
plt.legend()
plt.title('PCA 2D')

plt.show()

"""
  