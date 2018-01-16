
# RNN for Human Activity Recognition - 2D Pose Input

This experiment aims to determine the level of accuracy attainable in activity recognition using a 2D body pose dataset and an LSTM RNN. This involves classifying the following six types of human movement:

- JUMPING,
- JUMPING_JACKS,
- BOXING,
- WAVING_2HANDS,
- WAVING_1HAND,
- CLAPPING_HANDS.

The motivations behind this experiment are:

-  To determine if 2D pose has comparable accuracy to 3D pose for use in activity recognition. This would allow the use of RGB only cameras for human and animal pose estimation, as opposed to RGBD or a large motion capture dataset.


- To determine if  2D pose has comparable accuracy to using raw RGB images for use in activity recognition. This is based on the idea that limiting the input feature vector can help to deal with a limited dataset, as is likely to occur in animal activity recognition, by allowing for a smaller model to be used (citation required).


- To verify the concept for use in future works involving behaviour prediction from motion in 2D images.

The network used in this experiment is based on that of Guillaume Chevalier, 'LSTMs for Human Activity Recognition, 2016'  https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition, available under the MIT License.


## Dataset overview

The dataset consists of pose estimations, made using the software OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose's) on a subset of the Berkeley Multimodal Human Action Database (MHAD) dataset http://tele-immersion.citris-uc.org/berkeley_mhad.

This dataset is comprised of 12 subjects doing the 6 listed actions for 5 repetitions, filmed from 4 angles, repeated 5 times each.  
In total, there are 1438 videos (2 were missing) made up of 211200 individual frames.

The below image is an example of the 4 camera views during the 'boxing' action for subject 1

![boxing gif](images/boxing_all_views.gif.png "Title")

The input for the LSTM is the 2D position of 18 joints across a timeseries of frames numbering n_steps (window-width), with an associated class label for the frame series.  
A single frame's input (where j refers to a joint) is stored as:

[  j0_x,  j0_y, j1_x, j1_y , j2_x, j2_y, j3_x, j3_y, j4_x, j4_y, j5_x, j5_y, j6_x, j6_y, j7_x, j7_y, j8_x, j8_y, j9_x, j9_y, j10_x, j10_y, j11_x, j11_y, j12_x, j12_y, j13_x, j13_y, j14_x, j14_y, j15_x, j15_y, j16_x, j16_y, j17_x, j17_y ]

For the following experiment, very little preprocessing has been done to the dataset.  
The following steps were taken:
1. openpose run on individual frames, for each subject, action and view, outputting JSON of 18 joint x and y position keypoints and accuracies per frame
2. JSONs converted into txt format, keeping only x and y positions of each frame, action being performed during frame, and order of frames. This is used to create a database of associated activity class number and corresponding series of joint 2D positions
3. No further prepossessing was performed.  

In some cases, multiple people were detected in each frame, in which only the first detection was used.

The data has not been normalised with regards to subject position in the frame, motion across frame (if any), size of the subject, speed of action etc. It is essentially the raw 2D position of each joint viewed from a stationary camera.  
In many cases, individual joints were not located and a position of [0.0,0.0] was given for that joint

A summary of the dataset used for input is:

 - 211200 individual images 
 - n_steps = 33 frames (~=1.5s at 22Hz)
 - Images with noisy pose detection (detection of >=2 people) = 5132  
 - Training_split = 0.8
   - Length X_train = 4519
   - Length X_test = 1197




## Training and Results below: 
Training took approximately 2 mins running on a single GTX1080Ti, and was run for 2,800,000 iterations with a batch size of 1500  (600 epochs)



```python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
from random import randint

import os
```


```python
# Useful Constants

# Output classes to learn how to classify
LABELS = [    
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"

] 
DATASET_PATH = "data/HAR_pose_activities/database/"

X_train_path = DATASET_PATH + "x_train.txt"
X_test_path = DATASET_PATH + "x_test.txt"

y_train_path = DATASET_PATH + "y_train.txt"
y_test_path = DATASET_PATH + "y_test.txt"

n_steps = 33 # 33 timesteps per series
```

## Preparing dataset:


```python

# Load "X" (the neural network's training and testing inputs)

def load_X(X_path):
    file = open(X_path, 'r')
    # Read dataset from disk, dealing with text files' syntax
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]], 
        dtype=np.float32
    )
    file.close()
    blocks = int(len(X_) / n_steps)
    
    X_ = np.array(np.split(X_,blocks))

    return X_ #np.transpose(np.array(X_), (1, 2, 0))

# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    
    # for 0-based indexing 
    return y_ - 1

X_train = load_X(X_train_path)
X_test = load_X(X_test_path)
#print X_test

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)
# proof that it actually works for the skeptical: replace labelled classes with random classes to train on
#for i in range(len(y_train)):
#    y_train[i] = randint(0, 5)

```

## Set Parameters:



```python
# Input Data 

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # num testing series
#n_steps = len(X_train[0])  # num timesteps per series
n_input = len(X_train[0][0])  # num input parameters per timestep

n_hidden = 32 # Hidden layer num of features
n_classes = 6 

learning_rate = 0.0025
lambda_loss_amount = 0.0015

training_iters = training_data_count *600  # Loop 600 times on the dataset
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training

print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_train.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("\nThe dataset has not been preprocessed, is not normalised etc")



```

    (X shape, y shape, every X's mean, every X's standard deviation)
    ((4519, 33, 36), (1197, 1), 250.95729, 125.17004)
    
    The dataset has not been preprocessed, is not normalised etc


## Utility functions for training:


```python
def LSTM_RNN(_X, _weights, _biases):
    # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])   
    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # A single output is produced, in style of "many to one" classifier, 
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


```

## Build the network:


```python

# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


```

## Train the network:


```python
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs =         extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))
    if len(batch_ys[0]) < n_classes:
        temp_ys = np.zeros((batch_size, n_classes))
        temp_ys[:batch_ys.shape[0],:batch_ys.shape[1]] = batch_ys
        batch_ys = temp_ys
        #print temp_ys
        
    

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")

# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))

```

    Training iter #1500:   Batch Loss = 3.263521, Accuracy = 0.469999998808
    PERFORMANCE ON TEST SET: Batch Loss = 3.5757329464, Accuracy = 0.157059311867
    Training iter #30000:   Batch Loss = 2.705399, Accuracy = 0.401333332062
    PERFORMANCE ON TEST SET: Batch Loss = 2.86222410202, Accuracy = 0.326649963856
    Training iter #60000:   Batch Loss = 2.817847, Accuracy = 0.27133333683
    PERFORMANCE ON TEST SET: Batch Loss = 2.57150363922, Accuracy = 0.455304920673
    Training iter #90000:   Batch Loss = 2.239222, Accuracy = 0.595333337784
    PERFORMANCE ON TEST SET: Batch Loss = 2.32777118683, Accuracy = 0.581453621387
    Training iter #120000:   Batch Loss = 2.213794, Accuracy = 0.65066665411
    PERFORMANCE ON TEST SET: Batch Loss = 2.18517518044, Accuracy = 0.61236423254
    Training iter #150000:   Batch Loss = 2.095799, Accuracy = 0.595333337784
    PERFORMANCE ON TEST SET: Batch Loss = 2.10119485855, Accuracy = 0.639097750187
    Training iter #180000:   Batch Loss = 1.701719, Accuracy = 0.84933334589
    PERFORMANCE ON TEST SET: Batch Loss = 1.92504358292, Accuracy = 0.690058469772
    Training iter #210000:   Batch Loss = 1.950691, Accuracy = 0.678666651249
    PERFORMANCE ON TEST SET: Batch Loss = 1.81697237492, Accuracy = 0.751879692078
    Training iter #240000:   Batch Loss = 1.787254, Accuracy = 0.709999978542
    PERFORMANCE ON TEST SET: Batch Loss = 1.74033951759, Accuracy = 0.749373435974
    Training iter #270000:   Batch Loss = 1.661022, Accuracy = 0.850000023842
    PERFORMANCE ON TEST SET: Batch Loss = 1.65689218044, Accuracy = 0.802004992962
    Training iter #300000:   Batch Loss = 2.047323, Accuracy = 0.555999994278
    PERFORMANCE ON TEST SET: Batch Loss = 1.86204314232, Accuracy = 0.705096065998
    Training iter #330000:   Batch Loss = 1.679251, Accuracy = 0.791999995708
    PERFORMANCE ON TEST SET: Batch Loss = 1.75121617317, Accuracy = 0.749373435974
    Training iter #360000:   Batch Loss = 2.338127, Accuracy = 0.450666666031
    PERFORMANCE ON TEST SET: Batch Loss = 1.95276904106, Accuracy = 0.672514617443
    Training iter #390000:   Batch Loss = 2.124724, Accuracy = 0.532666683197
    PERFORMANCE ON TEST SET: Batch Loss = 1.78574895859, Accuracy = 0.726817071438
    Training iter #420000:   Batch Loss = 1.456425, Accuracy = 0.882666647434
    PERFORMANCE ON TEST SET: Batch Loss = 1.64866518974, Accuracy = 0.765246450901
    Training iter #450000:   Batch Loss = 1.699594, Accuracy = 0.78733330965
    PERFORMANCE ON TEST SET: Batch Loss = 1.58122229576, Accuracy = 0.819548845291
    Training iter #480000:   Batch Loss = 1.763726, Accuracy = 0.67199999094
    PERFORMANCE ON TEST SET: Batch Loss = 1.55079030991, Accuracy = 0.825396835804
    Training iter #510000:   Batch Loss = 1.367215, Accuracy = 0.902000010014
    PERFORMANCE ON TEST SET: Batch Loss = 1.52583169937, Accuracy = 0.8228905797
    Training iter #540000:   Batch Loss = 1.546330, Accuracy = 0.839333355427
    PERFORMANCE ON TEST SET: Batch Loss = 1.49364805222, Accuracy = 0.842105269432
    Training iter #570000:   Batch Loss = 1.716455, Accuracy = 0.686666667461
    PERFORMANCE ON TEST SET: Batch Loss = 1.51919567585, Accuracy = 0.816207170486
    Training iter #600000:   Batch Loss = 1.414168, Accuracy = 0.889999985695
    PERFORMANCE ON TEST SET: Batch Loss = 1.46393382549, Accuracy = 0.847117781639
    Training iter #630000:   Batch Loss = 1.468379, Accuracy = 0.84399998188
    PERFORMANCE ON TEST SET: Batch Loss = 1.4708520174, Accuracy = 0.846282362938
    Training iter #660000:   Batch Loss = 1.445780, Accuracy = 0.819333314896
    PERFORMANCE ON TEST SET: Batch Loss = 1.42821788788, Accuracy = 0.860484540462
    Training iter #690000:   Batch Loss = 1.490381, Accuracy = 0.841333329678
    PERFORMANCE ON TEST SET: Batch Loss = 1.41866397858, Accuracy = 0.860484540462
    Training iter #720000:   Batch Loss = 1.429964, Accuracy = 0.858666658401
    PERFORMANCE ON TEST SET: Batch Loss = 1.40660262108, Accuracy = 0.863826215267
    Training iter #750000:   Batch Loss = 1.302027, Accuracy = 0.899999976158
    PERFORMANCE ON TEST SET: Batch Loss = 1.3661904335, Accuracy = 0.871345043182
    Training iter #780000:   Batch Loss = 1.441992, Accuracy = 0.86533331871
    PERFORMANCE ON TEST SET: Batch Loss = 1.34452557564, Accuracy = 0.880534648895
    Training iter #810000:   Batch Loss = 1.583977, Accuracy = 0.758666694164
    PERFORMANCE ON TEST SET: Batch Loss = 1.33716642857, Accuracy = 0.878863811493
    Training iter #840000:   Batch Loss = 1.160152, Accuracy = 0.96266669035
    PERFORMANCE ON TEST SET: Batch Loss = 1.35014379025, Accuracy = 0.873851299286
    Training iter #870000:   Batch Loss = 1.433864, Accuracy = 0.856666684151
    PERFORMANCE ON TEST SET: Batch Loss = 1.31117510796, Accuracy = 0.883040964603
    Training iter #900000:   Batch Loss = 1.328209, Accuracy = 0.84933334589
    PERFORMANCE ON TEST SET: Batch Loss = 1.31462466717, Accuracy = 0.887218058109
    Training iter #930000:   Batch Loss = 1.164049, Accuracy = 0.958666682243
    PERFORMANCE ON TEST SET: Batch Loss = 1.27637970448, Accuracy = 0.899749398232
    Training iter #960000:   Batch Loss = 1.223539, Accuracy = 0.92733335495
    PERFORMANCE ON TEST SET: Batch Loss = 1.26833677292, Accuracy = 0.897243082523
    Training iter #990000:   Batch Loss = 1.233070, Accuracy = 0.882666647434
    PERFORMANCE ON TEST SET: Batch Loss = 1.25876665115, Accuracy = 0.902255654335
    Training iter #1020000:   Batch Loss = 1.356969, Accuracy = 0.874666690826
    PERFORMANCE ON TEST SET: Batch Loss = 1.32628536224, Accuracy = 0.873851299286
    Training iter #1050000:   Batch Loss = 1.283907, Accuracy = 0.899999976158
    PERFORMANCE ON TEST SET: Batch Loss = 1.2961139679, Accuracy = 0.88805347681
    Training iter #1080000:   Batch Loss = 1.330929, Accuracy = 0.844666659832
    PERFORMANCE ON TEST SET: Batch Loss = 1.329611063, Accuracy = 0.879699230194
    Training iter #1110000:   Batch Loss = 1.355566, Accuracy = 0.862666666508
    PERFORMANCE ON TEST SET: Batch Loss = 1.23856925964, Accuracy = 0.908939003944



```python
#one_hot_predictions1, accuracy1, final_loss1 = sess.run(
#    [pred, accuracy, cost],
#    feed_dict={
#        x: X_test,
#        y: one_hot(y_test)
#    }
#)
#
#print("NEW RESULT: " + \
 #     "Batch Loss = {}".format(final_loss1) + \
  #    ", Accuracy = {}".format(accuracy1))

```

## Results:




```python
# (Inline plots: )
%matplotlib inline

font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

width = 12
height = 12
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
#plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
#plt.plot(indep_test_axis, np.array(test_losses), "b-", linewidth=2.0, label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "b-", linewidth=2.0, label="Test accuracies")

plt.title("Training session's Accuracy over Iterations")
plt.legend(loc='lower right', shadow=True)
plt.ylabel('Training Accuracy')
plt.xlabel('Training Iteration')

plt.show()



# Results

predictions = one_hot_predictions.argmax(1)
print(predictions)

print("Testing Accuracy: {}%".format(100*accuracy))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
print("Created using test set of {} datapoints, normalised to % of each class in the test dataset".format(len(y_test)))
confusion_matrix = metrics.confusion_matrix(y_test, predictions)


#print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100


# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.Blues
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

```
![accuracy](images/accuracy.png "Title")
![accuracy](images/confusion_matrix.png "Title")

```python
#sess.close()
```

## Conclusion

Overall Accuracy of 92.65% is fantastic, considering that training took <2mins.
There are significant spikes in decreased accuracy even far into training, which suggests the need for more data.

Noticeable confusion between activities of Clapping hands and boxing, which is understandable.


In terms of the applicability of this to a wider dataset, I would imagine that it would be able to work for any activities in which the training included a views from all angles to be tested on. It would be interesting to see it's applicability to other, in-between views.

This experiment confirms the idea that 2D pose can be used for activity recognition, and provides verification to continue onto use of 2D pose for behaviour estimation.


## Future Works

Possibility to extend into a bidirectional-LSTM   (https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs)

Use on subtler activity classes and perhaps `normal vs abnormal` activity


## References

The dataset can be found at http://tele-immersion.citris-uc.org/berkeley_mhad released under the BSD-2 license
>Copyright (c) 2013, Regents of the University of California All rights reserved.

The network used in this experiment is based on the following, available under the [MIT License](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/LICENSE). :
> Guillaume Chevalier, LSTMs for Human Activity Recognition, 2016
> https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition




```python
# Let's convert this notebook to a README for the GitHub project's title page:
!jupyter nbconvert --to markdown LSTM.ipynb
!mv LSTM.md README.md
```
