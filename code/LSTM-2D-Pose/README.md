
# RNN for Human Activity Recognition - 2D Pose Input

This experiment aims to determine the level of accuracy attainable in Human activity recognition using a 2D body pose dataset and an LSTM RNN. This involves classifying the type of movement amongst six categories:
- JUMPING,
- JUMPING_JACKS,
- BOXING,
- WAVING_2HANDS,
- WAVING_1HAND,
- CLAPPING_HANDS.

The motivation behind this experiment is to determine if:

- 2D pose has comparable accuracy to 3D pose for activity estimation using an LSTM RNN, allowing the use of RGB only cameras for pose estimation, as opposed to RGBD or a large motion capture dataset.


- 2D pose has comparable accuracy to using raw RGB images for activity estimation using an LSTM RNN. This is based on the assumption that limiting the input feature vector may help to deal with a limited dataset (citation required)


- Verify the concept for use in future works involving behaviour prediction from motion in 2D images


## Dataset overview

The dataset is comprised of pose estimations, made using the software OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose's) on a subset of the Berkeley Multimodal Human Action Database (MHAD) dataset http://tele-immersion.citris-uc.org/berkeley_mhad.

This dataset is composed of 12 subjects doing the 6 listed actions for 5 repetitions, filmed from 4 angles, repeated 5 times each.  
In total, there are 1438 videos (2 were missing) made up of 211200 individual frames.

The below image is an example of the 4 camera views during the 'boxing' action for subject 1

![alt text](boxing_all_views .gif.png "Title")

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
Training took approximately 2 mins running on a single GTX1080Ti, and was run for 2800000 iterations with a batch size of 1500  (600 epochs)



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

## Additional Parameters:

Here are some core parameter definitions for the training. 

The whole neural network's structure could be summarised by enumerating those parameters and the fact an LSTM is used. 


```python
# Input Data 

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # num testing series
#n_steps = len(X_train[0])  # num timesteps per series
n_input = len(X_train[0][0])  # num input parameters per timestep

# LSTM Neural Network's internal structure

n_hidden = 32 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)


# Training 

learning_rate = 0.0025
lambda_loss_amount = 0.0015
print training_data_count
training_iters = training_data_count * 600  # Loop 600 times on the dataset
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training


# Some debugging info

print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_train.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("\nThe dataset has not been preprocessed, is not normalised etc")



```

    4519
    (X shape, y shape, every X's mean, every X's standard deviation)
    ((4519, 33, 36), (1197, 1), 250.95729, 125.17004)
    
    The dataset has not been preprocessed, is not normalised etc


## Utility functions for training:


```python
def LSTM_RNN(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
    # Note, some code of this notebook is inspired from an slightly different 
    # RNN architecture used on another dataset, some of the credits goes to 
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    #print(_X.shape)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier, 
    # as in the image describing RNNs at the top of this page
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
# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
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

    Training iter #1500:   Batch Loss = 2.858431, Accuracy = 0.393999993801
    PERFORMANCE ON TEST SET: Batch Loss = 3.4218416214, Accuracy = 0.173767745495
    Training iter #30000:   Batch Loss = 2.474910, Accuracy = 0.679333329201
    PERFORMANCE ON TEST SET: Batch Loss = 2.69645643234, Accuracy = 0.389306604862
    Training iter #60000:   Batch Loss = 2.766936, Accuracy = 0.172666668892
    PERFORMANCE ON TEST SET: Batch Loss = 2.44221806526, Accuracy = 0.479532152414
    Training iter #90000:   Batch Loss = 2.058732, Accuracy = 0.727999985218
    PERFORMANCE ON TEST SET: Batch Loss = 2.21119689941, Accuracy = 0.534669995308
    Training iter #120000:   Batch Loss = 1.926992, Accuracy = 0.635999977589
    PERFORMANCE ON TEST SET: Batch Loss = 1.91425514221, Accuracy = 0.636591494083
    Training iter #150000:   Batch Loss = 2.150640, Accuracy = 0.523999989033
    PERFORMANCE ON TEST SET: Batch Loss = 1.90793943405, Accuracy = 0.61236423254
    Training iter #180000:   Batch Loss = 1.562822, Accuracy = 0.794000029564
    PERFORMANCE ON TEST SET: Batch Loss = 1.79039335251, Accuracy = 0.675856292248
    Training iter #210000:   Batch Loss = 1.940795, Accuracy = 0.576666653156
    PERFORMANCE ON TEST SET: Batch Loss = 1.69785773754, Accuracy = 0.714285731316
    Training iter #240000:   Batch Loss = 1.559604, Accuracy = 0.721333324909
    PERFORMANCE ON TEST SET: Batch Loss = 1.53751063347, Accuracy = 0.755221366882
    Training iter #270000:   Batch Loss = 1.454836, Accuracy = 0.882666647434
    PERFORMANCE ON TEST SET: Batch Loss = 1.48143863678, Accuracy = 0.786131978035
    Training iter #300000:   Batch Loss = 1.453875, Accuracy = 0.811333358288
    PERFORMANCE ON TEST SET: Batch Loss = 1.46064841747, Accuracy = 0.795321643353
    Training iter #330000:   Batch Loss = 1.487346, Accuracy = 0.793333351612
    PERFORMANCE ON TEST SET: Batch Loss = 1.43448424339, Accuracy = 0.821219742298
    Training iter #360000:   Batch Loss = 1.472841, Accuracy = 0.796000003815
    PERFORMANCE ON TEST SET: Batch Loss = 1.36664795876, Accuracy = 0.849624037743
    Training iter #390000:   Batch Loss = 1.549598, Accuracy = 0.742666661739
    PERFORMANCE ON TEST SET: Batch Loss = 1.55220019817, Accuracy = 0.752715110779
    Training iter #420000:   Batch Loss = 1.231814, Accuracy = 0.874666690826
    PERFORMANCE ON TEST SET: Batch Loss = 1.32333791256, Accuracy = 0.835421860218
    Training iter #450000:   Batch Loss = 1.326605, Accuracy = 0.867333352566
    PERFORMANCE ON TEST SET: Batch Loss = 1.27999770641, Accuracy = 0.859649121761
    Training iter #480000:   Batch Loss = 1.356358, Accuracy = 0.811999976635
    PERFORMANCE ON TEST SET: Batch Loss = 1.22538757324, Accuracy = 0.886382639408
    Training iter #510000:   Batch Loss = 1.136359, Accuracy = 0.922666668892
    PERFORMANCE ON TEST SET: Batch Loss = 1.22425293922, Accuracy = 0.88805347681
    Training iter #540000:   Batch Loss = 1.457735, Accuracy = 0.790000021458
    PERFORMANCE ON TEST SET: Batch Loss = 1.44615828991, Accuracy = 0.809523820877
    Training iter #570000:   Batch Loss = 1.476400, Accuracy = 0.783999979496
    PERFORMANCE ON TEST SET: Batch Loss = 1.34481608868, Accuracy = 0.832915604115
    Training iter #600000:   Batch Loss = 1.196212, Accuracy = 0.889333307743
    PERFORMANCE ON TEST SET: Batch Loss = 1.24111294746, Accuracy = 0.862990796566
    Training iter #630000:   Batch Loss = 1.212855, Accuracy = 0.889333307743
    PERFORMANCE ON TEST SET: Batch Loss = 1.21287846565, Accuracy = 0.881370067596
    Training iter #660000:   Batch Loss = 1.276989, Accuracy = 0.842666685581
    PERFORMANCE ON TEST SET: Batch Loss = 1.21749556065, Accuracy = 0.872180461884
    Training iter #690000:   Batch Loss = 1.310523, Accuracy = 0.835333347321
    PERFORMANCE ON TEST SET: Batch Loss = 1.1776175499, Accuracy = 0.885547220707
    Training iter #720000:   Batch Loss = 1.191452, Accuracy = 0.899333357811
    PERFORMANCE ON TEST SET: Batch Loss = 1.16556513309, Accuracy = 0.895572245121
    Training iter #750000:   Batch Loss = 1.120094, Accuracy = 0.901333332062
    PERFORMANCE ON TEST SET: Batch Loss = 1.1469233036, Accuracy = 0.906432747841
    Training iter #780000:   Batch Loss = 1.174471, Accuracy = 0.888000011444
    PERFORMANCE ON TEST SET: Batch Loss = 1.1368560791, Accuracy = 0.907268166542
    Training iter #810000:   Batch Loss = 1.197682, Accuracy = 0.861333310604
    PERFORMANCE ON TEST SET: Batch Loss = 1.11667203903, Accuracy = 0.90559732914
    Training iter #840000:   Batch Loss = 0.947227, Accuracy = 0.973333358765
    PERFORMANCE ON TEST SET: Batch Loss = 1.11638569832, Accuracy = 0.908939003944
    Training iter #870000:   Batch Loss = 1.120424, Accuracy = 0.907999992371
    PERFORMANCE ON TEST SET: Batch Loss = 1.10826396942, Accuracy = 0.913951516151
    Training iter #900000:   Batch Loss = 1.124660, Accuracy = 0.89200001955
    PERFORMANCE ON TEST SET: Batch Loss = 1.16280066967, Accuracy = 0.882205486298
    Training iter #930000:   Batch Loss = 1.004702, Accuracy = 0.941333353519
    PERFORMANCE ON TEST SET: Batch Loss = 1.12939500809, Accuracy = 0.88805347681
    Training iter #960000:   Batch Loss = 1.200920, Accuracy = 0.839999973774
    PERFORMANCE ON TEST SET: Batch Loss = 1.14680039883, Accuracy = 0.888888895512
    Training iter #990000:   Batch Loss = 1.321847, Accuracy = 0.844666659832
    PERFORMANCE ON TEST SET: Batch Loss = 1.28664565086, Accuracy = 0.837092757225
    Training iter #1020000:   Batch Loss = 1.102982, Accuracy = 0.920666694641
    PERFORMANCE ON TEST SET: Batch Loss = 1.12982702255, Accuracy = 0.893901407719
    Training iter #1050000:   Batch Loss = 1.488249, Accuracy = 0.790666639805
    PERFORMANCE ON TEST SET: Batch Loss = 1.1061822176, Accuracy = 0.895572245121
    Training iter #1080000:   Batch Loss = 1.089628, Accuracy = 0.882000029087
    PERFORMANCE ON TEST SET: Batch Loss = 1.15002560616, Accuracy = 0.884711802006
    Training iter #1110000:   Batch Loss = 1.051222, Accuracy = 0.924666643143
    PERFORMANCE ON TEST SET: Batch Loss = 1.08740139008, Accuracy = 0.898078501225
    Training iter #1140000:   Batch Loss = 1.092026, Accuracy = 0.891333341599
    PERFORMANCE ON TEST SET: Batch Loss = 1.03141903877, Accuracy = 0.91311609745
    Training iter #1170000:   Batch Loss = 0.915130, Accuracy = 0.963999986649
    PERFORMANCE ON TEST SET: Batch Loss = 1.00868606567, Accuracy = 0.926482856274
    Training iter #1200000:   Batch Loss = 1.035496, Accuracy = 0.917999982834
    PERFORMANCE ON TEST SET: Batch Loss = 0.995334565639, Accuracy = 0.929824590683
    Training iter #1230000:   Batch Loss = 1.043092, Accuracy = 0.91333335638
    PERFORMANCE ON TEST SET: Batch Loss = 1.00767338276, Accuracy = 0.925647437572
    Training iter #1260000:   Batch Loss = 0.882345, Accuracy = 0.977333307266
    PERFORMANCE ON TEST SET: Batch Loss = 0.991328239441, Accuracy = 0.930660009384
    Training iter #1290000:   Batch Loss = 1.385255, Accuracy = 0.710666656494
    PERFORMANCE ON TEST SET: Batch Loss = 1.36463439465, Accuracy = 0.761069357395
    Training iter #1320000:   Batch Loss = 1.190219, Accuracy = 0.795333325863
    PERFORMANCE ON TEST SET: Batch Loss = 1.19317436218, Accuracy = 0.84795320034
    Training iter #1350000:   Batch Loss = 1.424231, Accuracy = 0.739333331585
    PERFORMANCE ON TEST SET: Batch Loss = 1.25660252571, Accuracy = 0.802004992962
    Training iter #1380000:   Batch Loss = 1.088668, Accuracy = 0.894666671753
    PERFORMANCE ON TEST SET: Batch Loss = 1.10594177246, Accuracy = 0.878863811493
    Training iter #1410000:   Batch Loss = 1.081797, Accuracy = 0.832666695118
    PERFORMANCE ON TEST SET: Batch Loss = 1.06329703331, Accuracy = 0.896407663822
    Training iter #1440000:   Batch Loss = 1.109054, Accuracy = 0.906666696072
    PERFORMANCE ON TEST SET: Batch Loss = 1.12083053589, Accuracy = 0.870509624481
    Training iter #1470000:   Batch Loss = 1.161330, Accuracy = 0.846666693687
    PERFORMANCE ON TEST SET: Batch Loss = 1.05593252182, Accuracy = 0.897243082523
    Training iter #1500000:   Batch Loss = 1.001712, Accuracy = 0.909333348274
    PERFORMANCE ON TEST SET: Batch Loss = 1.06217455864, Accuracy = 0.890559732914
    Training iter #1530000:   Batch Loss = 1.038150, Accuracy = 0.902666687965
    PERFORMANCE ON TEST SET: Batch Loss = 1.01684617996, Accuracy = 0.907268166542
    Training iter #1560000:   Batch Loss = 1.020987, Accuracy = 0.899999976158
    PERFORMANCE ON TEST SET: Batch Loss = 1.00129270554, Accuracy = 0.914786994457
    Training iter #1590000:   Batch Loss = 0.854409, Accuracy = 0.977999985218
    PERFORMANCE ON TEST SET: Batch Loss = 0.986159026623, Accuracy = 0.919799506664
    Training iter #1620000:   Batch Loss = 0.975534, Accuracy = 0.924000024796
    PERFORMANCE ON TEST SET: Batch Loss = 0.986457705498, Accuracy = 0.918964087963
    Training iter #1650000:   Batch Loss = 0.998456, Accuracy = 0.913999974728
    PERFORMANCE ON TEST SET: Batch Loss = 0.999859452248, Accuracy = 0.913951516151
    Training iter #1680000:   Batch Loss = 0.932903, Accuracy = 0.937333345413
    PERFORMANCE ON TEST SET: Batch Loss = 1.02358818054, Accuracy = 0.912280678749
    Training iter #1710000:   Batch Loss = 0.992034, Accuracy = 0.925999999046
    PERFORMANCE ON TEST SET: Batch Loss = 1.04915475845, Accuracy = 0.893901407719
    Training iter #1740000:   Batch Loss = 0.976052, Accuracy = 0.915333330631
    PERFORMANCE ON TEST SET: Batch Loss = 0.984206974506, Accuracy = 0.919799506664
    Training iter #1770000:   Batch Loss = 1.057021, Accuracy = 0.901333332062
    PERFORMANCE ON TEST SET: Batch Loss = 1.02015125751, Accuracy = 0.907268166542
    Training iter #1800000:   Batch Loss = 1.689890, Accuracy = 0.729333341122
    PERFORMANCE ON TEST SET: Batch Loss = 1.17303204536, Accuracy = 0.862155377865
    Training iter #1830000:   Batch Loss = 0.959456, Accuracy = 0.945333361626
    PERFORMANCE ON TEST SET: Batch Loss = 1.12682652473, Accuracy = 0.851294875145
    Training iter #1860000:   Batch Loss = 1.146214, Accuracy = 0.874666690826
    PERFORMANCE ON TEST SET: Batch Loss = 1.40079915524, Accuracy = 0.746867179871
    Training iter #1890000:   Batch Loss = 1.170020, Accuracy = 0.828666687012
    PERFORMANCE ON TEST SET: Batch Loss = 1.24955153465, Accuracy = 0.792815387249
    Training iter #1920000:   Batch Loss = 0.936818, Accuracy = 0.941333353519
    PERFORMANCE ON TEST SET: Batch Loss = 1.15138697624, Accuracy = 0.84043443203
    Training iter #1950000:   Batch Loss = 1.101296, Accuracy = 0.829333305359
    PERFORMANCE ON TEST SET: Batch Loss = 1.11628997326, Accuracy = 0.853801190853
    Training iter #1980000:   Batch Loss = 1.318386, Accuracy = 0.800000011921
    PERFORMANCE ON TEST SET: Batch Loss = 1.23742032051, Accuracy = 0.815371751785
    Training iter #2010000:   Batch Loss = 0.916270, Accuracy = 0.952666640282
    PERFORMANCE ON TEST SET: Batch Loss = 1.10201179981, Accuracy = 0.848788619041
    Training iter #2040000:   Batch Loss = 1.094458, Accuracy = 0.834666669369
    PERFORMANCE ON TEST SET: Batch Loss = 1.08372020721, Accuracy = 0.866332471371
    Training iter #2070000:   Batch Loss = 1.080838, Accuracy = 0.874666690826
    PERFORMANCE ON TEST SET: Batch Loss = 1.04466867447, Accuracy = 0.883876383305
    Training iter #2100000:   Batch Loss = 0.954394, Accuracy = 0.929333329201
    PERFORMANCE ON TEST SET: Batch Loss = 1.01631319523, Accuracy = 0.893065989017
    Training iter #2130000:   Batch Loss = 1.002459, Accuracy = 0.878000020981
    PERFORMANCE ON TEST SET: Batch Loss = 1.00755012035, Accuracy = 0.900584816933
    Training iter #2160000:   Batch Loss = 1.106710, Accuracy = 0.828666687012
    PERFORMANCE ON TEST SET: Batch Loss = 1.05919265747, Accuracy = 0.880534648895
    Training iter #2190000:   Batch Loss = 0.940905, Accuracy = 0.92199999094
    PERFORMANCE ON TEST SET: Batch Loss = 1.01107513905, Accuracy = 0.903091073036
    Training iter #2220000:   Batch Loss = 1.560668, Accuracy = 0.757333338261
    PERFORMANCE ON TEST SET: Batch Loss = 1.13032889366, Accuracy = 0.862990796566
    Training iter #2250000:   Batch Loss = 1.018481, Accuracy = 0.87933331728
    PERFORMANCE ON TEST SET: Batch Loss = 1.15920805931, Accuracy = 0.842105269432
    Training iter #2280000:   Batch Loss = 1.145407, Accuracy = 0.828666687012
    PERFORMANCE ON TEST SET: Batch Loss = 1.12602710724, Accuracy = 0.86549705267
    Training iter #2310000:   Batch Loss = 1.035072, Accuracy = 0.911333322525
    PERFORMANCE ON TEST SET: Batch Loss = 0.98381125927, Accuracy = 0.922305762768
    Training iter #2340000:   Batch Loss = 0.835576, Accuracy = 0.95733332634
    PERFORMANCE ON TEST SET: Batch Loss = 1.0050753355, Accuracy = 0.903091073036
    Training iter #2370000:   Batch Loss = 0.882022, Accuracy = 0.948000013828
    PERFORMANCE ON TEST SET: Batch Loss = 0.933361768723, Accuracy = 0.929824590683
    Training iter #2400000:   Batch Loss = 0.876405, Accuracy = 0.951333343983
    PERFORMANCE ON TEST SET: Batch Loss = 0.91218495369, Accuracy = 0.936507940292
    Training iter #2430000:   Batch Loss = 0.943895, Accuracy = 0.925333321095
    PERFORMANCE ON TEST SET: Batch Loss = 1.05424368382, Accuracy = 0.875522136688
    Training iter #2460000:   Batch Loss = 1.155895, Accuracy = 0.775333344936
    PERFORMANCE ON TEST SET: Batch Loss = 1.02826154232, Accuracy = 0.883876383305
    Training iter #2490000:   Batch Loss = 0.885080, Accuracy = 0.949333310127
    PERFORMANCE ON TEST SET: Batch Loss = 0.952493369579, Accuracy = 0.921470344067
    Training iter #2520000:   Batch Loss = 0.880505, Accuracy = 0.936666667461
    PERFORMANCE ON TEST SET: Batch Loss = 0.914833545685, Accuracy = 0.940685033798


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
plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
#plt.plot(indep_test_axis, np.array(test_losses), "b-", linewidth=2.0, label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "g-", linewidth=2.0, label="Test accuracies")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')

plt.show()

# Results

predictions = one_hot_predictions.argmax(1)

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


```python
sess.close()
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
