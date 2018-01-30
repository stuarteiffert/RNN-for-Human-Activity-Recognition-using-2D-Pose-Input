
## RNN for Human Activity Recognition - 2D Pose Input

This experiment is the classification of human activities using a 2D pose time series dataset and an LSTM RNN.
The idea is to prove the concept that using a series of 2D poses, rather than 3D poses or a raw 2D images, can produce an accurate estimation of the behaviour of a person or animal.
This is a step towards creating a method of classifying an animal's current behaviour state and predicting it's likely next state, allowing for better interaction with an autonomous mobile robot.

## Objectives

The aims of this experiment are:

-  To determine if 2D pose has comparable accuracy to 3D pose for use in activity recognition. This would allow the use of RGB only cameras for human and animal pose estimation, as opposed to RGBD or a large motion capture dataset.


- To determine if  2D pose has comparable accuracy to using raw RGB images for use in activity recognition. This is based on the idea that limiting the input feature vector can help to deal with a limited dataset, as is likely to occur in animal activity recognition, by allowing for a smaller model to be used (citation required).


- To verify the concept for use in future works involving behaviour prediction from motion in 2D images.

The network used in this experiment is based on that of Guillaume Chevalier, 'LSTMs for Human Activity Recognition, 2016'  https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition, available under the MIT License.
Notable changes that have been made (other than accounting for dataset sizes) are:
 - Adapting for use with a large dataset ordered by class, using random sampling without replacement for mini-batch.  
 This allows for use of smaller batch sizes when using a dataset ordered by class. "It has been observed in practice that when using a larger batch there is a significant degradation in the quality of the model, as measured by its ability to generalize"  
      _N.S Keskar, D. Mudigere, et al, 'On Large-Batch Training for Deep Learning: Generalization Gap and Sharp 
      Minima', ICLR 2017_ https://arxiv.org/abs/1609.04836
      
 - Exponentially decaying learning rate implemented



## Dataset overview

The dataset consists of pose estimations, made using the software OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose's) on a subset of the Berkeley Multimodal Human Action Database (MHAD) dataset http://tele-immersion.citris-uc.org/berkeley_mhad.

This dataset is comprised of 12 subjects doing the following 6 actions for 5 repetitions, filmed from 4 angles, repeated 5 times each.  

- JUMPING,
- JUMPING_JACKS,
- BOXING,
- WAVING_2HANDS,
- WAVING_1HAND,
- CLAPPING_HANDS.

In total, there are 1438 videos (2 were missing) made up of 211200 individual frames.

The below image is an example of the 4 camera views during the 'boxing' action for subject 1

![alt text](images/boxing_all_views.gif.png "Title")


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
 - n_steps = 32 frames (~=1.5s at 22Hz)
 - Images with noisy pose detection (detection of >=2 people) = 5132  
 - Training_split = 0.8
 - Overlap = 0.8125 (26 / 32) ie 26 frame overlap
   - Length X_train = 22625 * 32 frames
   - Length X_test = 5751 * 32 frames
   
Note that their is no overlap between test and train sets, which were seperated by activity repetition entirely, before creating the 26 of 32 frame overlap.




## Training and Results below: 
Training took approximately 4 mins running on a single GTX1080Ti, and was run for 22,000,000ish iterations with a batch size of 5000  (600 epochs)



```python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import random
from random import randint
import time
import os
```

## Preparing dataset:


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

X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH + "X_test.txt"

y_train_path = DATASET_PATH + "Y_train.txt"
y_test_path = DATASET_PATH + "Y_test.txt"

n_steps = 32 # 32 timesteps per series
```


```python

# Load the networks inputs

def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]], 
        dtype=np.float32
    )
    file.close()
    blocks = int(len(X_) / n_steps)
    
    X_ = np.array(np.split(X_,blocks))

    return X_ 

# Load the networks outputs

def load_y(y_path):
    file = open(y_path, 'r')
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

training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 1197 test series
n_input = len(X_train[0][0])  # num input parameters per timestep

n_hidden = 34 # Hidden layer num of features
n_classes = 6 

#updated for learning-rate decay
# calculated as: decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
decaying_learning_rate = True
learning_rate = 0.0025 #used if decaying_learning_rate set to False
init_learning_rate = 0.005
decay_rate = 0.96 #the base of the exponential in the decay
decay_steps = 100000 #used in decay every 60000 steps with a base of 0.96

global_step = tf.Variable(0, trainable=False)
lambda_loss_amount = 0.0015

training_iters = training_data_count *300  # Loop 300 times on the dataset, ie 300 epochs
batch_size = 512
display_iter = batch_size*8  # To show test set accuracy during training

print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_train.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("\nThe dataset has not been preprocessed, is not normalised etc")



```

    (X shape, y shape, every X's mean, every X's standard deviation)
    ((22625, 32, 36), (5751, 1), 251.01117, 126.12204)
    
    The dataset has not been preprocessed, is not normalised etc


## Utility functions for training:


```python
def LSTM_RNN(_X, _weights, _biases):
    # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])   
    # Rectifies Linear Unit activation function used
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, _labels, _unsampled, batch_size):
    # Fetch a "batch_size" amount of data and labels from "(X|y)_train" data. 
    # Elements of each batch are chosen randomly, without replacement, from X_train with corresponding label from Y_train
    # unsampled_indices keeps track of sampled data ensuring non-replacement. Resets when remaining datapoints < batch_size    
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    batch_labels = np.empty((batch_size,1)) 

    for i in range(batch_size):
        # Loop index
        # index = random sample from _unsampled (indices)
        index = random.choice(_unsampled)
        batch_s[i] = _train[index] 
        batch_labels[i] = _labels[index]
        _unsampled.remove(index)


    return batch_s, batch_labels, _unsampled


def one_hot(y_):
    # One hot encoding of the network outputs
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
if decaying_learning_rate:
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)


#decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps) #exponentially decayed learning rate
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step) # Adam Optimizer

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

# Perform Training steps with "batch_size" amount of data at each loop. 
# Elements of each batch are chosen randomly, without replacement, from X_train, 
# restarting when remaining datapoints < batch_size
step = 1
time_start = time.time()
unsampled_indices = range(0,len(X_train))

while step * batch_size <= training_iters:
    #print (sess.run(learning_rate)) #decaying learning rate
    #print (sess.run(global_step)) # global number of iterations
    if len(unsampled_indices) < batch_size:
        unsampled_indices = range(0,len(X_train)) 
    batch_xs, raw_labels, unsampled_indicies = extract_batch_size(X_train, y_train, unsampled_indices, batch_size)
    batch_ys = one_hot(raw_labels)
    # check that encoded output is same length as num_classes, if not, pad it 
    if len(batch_ys[0]) < n_classes:
        temp_ys = np.zeros((batch_size, n_classes))
        temp_ys[:batch_ys.shape[0],:batch_ys.shape[1]] = batch_ys
        batch_ys = temp_ys
       
    

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
        print("Iter #" + str(step*batch_size) + \
              ":  Learning rate = " + "{:.6f}".format(sess.run(learning_rate)) + \
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
        print("PERFORMANCE ON TEST SET:             " + \
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
time_stop = time.time()
print("TOTAL TIME:  {}".format(time_stop - time_start))
```

    Iter #512:  Learning rate = 0.005000:   Batch Loss = 4.315274, Accuracy = 0.15234375
    PERFORMANCE ON TEST SET:             Batch Loss = 3.68809938431, Accuracy = 0.205529466271
    Iter #4096:  Learning rate = 0.005000:   Batch Loss = 3.057283, Accuracy = 0.263671875
    PERFORMANCE ON TEST SET:             Batch Loss = 3.04467487335, Accuracy = 0.262215256691
    Iter #8192:  Learning rate = 0.005000:   Batch Loss = 2.814128, Accuracy = 0.345703125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.78404188156, Accuracy = 0.383933216333
    Iter #12288:  Learning rate = 0.005000:   Batch Loss = 2.682125, Accuracy = 0.44921875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.71795201302, Accuracy = 0.407233536243
    Iter #16384:  Learning rate = 0.005000:   Batch Loss = 2.361879, Accuracy = 0.5
    PERFORMANCE ON TEST SET:             Batch Loss = 2.44809341431, Accuracy = 0.497652590275
    Iter #20480:  Learning rate = 0.005000:   Batch Loss = 2.457385, Accuracy = 0.482421875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.40264844894, Accuracy = 0.491218924522
    Iter #24576:  Learning rate = 0.005000:   Batch Loss = 2.324637, Accuracy = 0.53515625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.33724546432, Accuracy = 0.5679012537
    Iter #28672:  Learning rate = 0.005000:   Batch Loss = 2.195830, Accuracy = 0.580078125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.18553352356, Accuracy = 0.560772061348
    Iter #32768:  Learning rate = 0.005000:   Batch Loss = 2.254563, Accuracy = 0.552734375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.43282079697, Accuracy = 0.494001030922
    Iter #36864:  Learning rate = 0.005000:   Batch Loss = 2.121828, Accuracy = 0.591796875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.07830786705, Accuracy = 0.613284647465
    Iter #40960:  Learning rate = 0.005000:   Batch Loss = 2.028024, Accuracy = 0.6484375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.98002696037, Accuracy = 0.670492112637
    Iter #45056:  Learning rate = 0.005000:   Batch Loss = 1.830399, Accuracy = 0.708984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.96158254147, Accuracy = 0.612936854362
    Iter #49152:  Learning rate = 0.005000:   Batch Loss = 1.906033, Accuracy = 0.671875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.18290996552, Accuracy = 0.551730155945
    Iter #53248:  Learning rate = 0.005000:   Batch Loss = 1.980418, Accuracy = 0.62890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.88367033005, Accuracy = 0.681794464588
    Iter #57344:  Learning rate = 0.005000:   Batch Loss = 1.784389, Accuracy = 0.71875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.77447557449, Accuracy = 0.728742837906
    Iter #61440:  Learning rate = 0.005000:   Batch Loss = 1.774224, Accuracy = 0.69140625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.82376849651, Accuracy = 0.65988522768
    Iter #65536:  Learning rate = 0.005000:   Batch Loss = 1.828996, Accuracy = 0.7109375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.86022770405, Accuracy = 0.663536787033
    Iter #69632:  Learning rate = 0.005000:   Batch Loss = 1.709383, Accuracy = 0.736328125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.68174505234, Accuracy = 0.733089923859
    Iter #73728:  Learning rate = 0.005000:   Batch Loss = 2.291216, Accuracy = 0.494140625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.58328270912, Accuracy = 0.406885772943
    Iter #77824:  Learning rate = 0.005000:   Batch Loss = 2.208295, Accuracy = 0.494140625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.1197783947, Accuracy = 0.525647699833
    Iter #81920:  Learning rate = 0.005000:   Batch Loss = 1.933757, Accuracy = 0.640625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.96123898029, Accuracy = 0.60511213541
    Iter #86016:  Learning rate = 0.005000:   Batch Loss = 1.801550, Accuracy = 0.666015625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.72301244736, Accuracy = 0.719527065754
    Iter #90112:  Learning rate = 0.005000:   Batch Loss = 1.621892, Accuracy = 0.74609375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6734855175, Accuracy = 0.725439071655
    Iter #94208:  Learning rate = 0.005000:   Batch Loss = 1.639348, Accuracy = 0.736328125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.76429510117, Accuracy = 0.706311941147
    Iter #98304:  Learning rate = 0.005000:   Batch Loss = 1.577094, Accuracy = 0.763671875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.63778710365, Accuracy = 0.754303574562
    Iter #102400:  Learning rate = 0.004800:   Batch Loss = 1.569097, Accuracy = 0.787109375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.54719305038, Accuracy = 0.773604571819
    Iter #106496:  Learning rate = 0.004800:   Batch Loss = 1.543472, Accuracy = 0.775390625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.56548810005, Accuracy = 0.751695334911
    Iter #110592:  Learning rate = 0.004800:   Batch Loss = 1.622594, Accuracy = 0.744140625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.62523841858, Accuracy = 0.717266559601
    Iter #114688:  Learning rate = 0.004800:   Batch Loss = 1.581402, Accuracy = 0.73828125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4758143425, Accuracy = 0.792905569077
    Iter #118784:  Learning rate = 0.004800:   Batch Loss = 1.481808, Accuracy = 0.78515625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.70568919182, Accuracy = 0.706138074398
    Iter #122880:  Learning rate = 0.004800:   Batch Loss = 1.553600, Accuracy = 0.75
    PERFORMANCE ON TEST SET:             Batch Loss = 1.66563844681, Accuracy = 0.721439778805
    Iter #126976:  Learning rate = 0.004800:   Batch Loss = 1.975961, Accuracy = 0.591796875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.86973595619, Accuracy = 0.608589828014
    Iter #131072:  Learning rate = 0.004800:   Batch Loss = 1.678564, Accuracy = 0.703125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.09086513519, Accuracy = 0.511910974979
    Iter #135168:  Learning rate = 0.004800:   Batch Loss = 1.644627, Accuracy = 0.7109375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.64355492592, Accuracy = 0.734133183956
    Iter #139264:  Learning rate = 0.004800:   Batch Loss = 1.566514, Accuracy = 0.76953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.54966616631, Accuracy = 0.754477500916
    Iter #143360:  Learning rate = 0.004800:   Batch Loss = 1.510630, Accuracy = 0.744140625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.47167301178, Accuracy = 0.791166782379
    Iter #147456:  Learning rate = 0.004800:   Batch Loss = 1.384963, Accuracy = 0.806640625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.46597898006, Accuracy = 0.76473659277
    Iter #151552:  Learning rate = 0.004800:   Batch Loss = 1.398920, Accuracy = 0.78515625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.40417599678, Accuracy = 0.790471196175
    Iter #155648:  Learning rate = 0.004800:   Batch Loss = 1.405812, Accuracy = 0.8046875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4478969574, Accuracy = 0.782298743725
    Iter #159744:  Learning rate = 0.004800:   Batch Loss = 1.364672, Accuracy = 0.806640625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3612511158, Accuracy = 0.798817574978
    Iter #163840:  Learning rate = 0.004800:   Batch Loss = 1.324863, Accuracy = 0.841796875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.33567452431, Accuracy = 0.812554359436
    Iter #167936:  Learning rate = 0.004800:   Batch Loss = 1.278751, Accuracy = 0.8671875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.31781840324, Accuracy = 0.806816220284
    Iter #172032:  Learning rate = 0.004800:   Batch Loss = 1.405587, Accuracy = 0.787109375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.32240736485, Accuracy = 0.805772900581
    Iter #176128:  Learning rate = 0.004800:   Batch Loss = 1.353744, Accuracy = 0.80078125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.37743389606, Accuracy = 0.770996332169
    Iter #180224:  Learning rate = 0.004800:   Batch Loss = 1.261146, Accuracy = 0.841796875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.24856877327, Accuracy = 0.836028516293
    Iter #184320:  Learning rate = 0.004800:   Batch Loss = 1.259667, Accuracy = 0.849609375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.26456689835, Accuracy = 0.818988025188
    Iter #188416:  Learning rate = 0.004800:   Batch Loss = 1.255547, Accuracy = 0.84765625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.24595379829, Accuracy = 0.841071128845
    Iter #192512:  Learning rate = 0.004800:   Batch Loss = 1.225653, Accuracy = 0.86328125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.24891602993, Accuracy = 0.837245702744
    Iter #196608:  Learning rate = 0.004800:   Batch Loss = 1.140967, Accuracy = 0.912109375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.26040267944, Accuracy = 0.833246409893
    Iter #200704:  Learning rate = 0.004608:   Batch Loss = 1.196584, Accuracy = 0.865234375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.26048076153, Accuracy = 0.827682137489
    Iter #204800:  Learning rate = 0.004608:   Batch Loss = 1.271167, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.22448432446, Accuracy = 0.839853942394
    Iter #208896:  Learning rate = 0.004608:   Batch Loss = 1.182554, Accuracy = 0.861328125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.24009859562, Accuracy = 0.828377664089
    Iter #212992:  Learning rate = 0.004608:   Batch Loss = 1.229189, Accuracy = 0.849609375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.20654940605, Accuracy = 0.844896554947
    Iter #217088:  Learning rate = 0.004608:   Batch Loss = 1.165783, Accuracy = 0.876953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.19247591496, Accuracy = 0.841940522194
    Iter #221184:  Learning rate = 0.004608:   Batch Loss = 1.153975, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.19130706787, Accuracy = 0.850113034248
    Iter #225280:  Learning rate = 0.004608:   Batch Loss = 1.134334, Accuracy = 0.87890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.21968829632, Accuracy = 0.840549468994
    Iter #229376:  Learning rate = 0.004608:   Batch Loss = 1.113050, Accuracy = 0.896484375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.17810988426, Accuracy = 0.846113741398
    Iter #233472:  Learning rate = 0.004608:   Batch Loss = 1.124773, Accuracy = 0.876953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.12641298771, Accuracy = 0.864023625851
    Iter #237568:  Learning rate = 0.004608:   Batch Loss = 1.191074, Accuracy = 0.853515625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.18647623062, Accuracy = 0.85532951355
    Iter #241664:  Learning rate = 0.004608:   Batch Loss = 1.094016, Accuracy = 0.896484375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.17532229424, Accuracy = 0.845070421696
    Iter #245760:  Learning rate = 0.004608:   Batch Loss = 1.160288, Accuracy = 0.869140625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.15901100636, Accuracy = 0.858285486698
    Iter #249856:  Learning rate = 0.004608:   Batch Loss = 1.197851, Accuracy = 0.8515625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.11959958076, Accuracy = 0.862284839153
    Iter #253952:  Learning rate = 0.004608:   Batch Loss = 1.078244, Accuracy = 0.88671875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.09663808346, Accuracy = 0.872717797756
    Iter #258048:  Learning rate = 0.004608:   Batch Loss = 1.106227, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.10878241062, Accuracy = 0.870805084705
    Iter #262144:  Learning rate = 0.004608:   Batch Loss = 1.090835, Accuracy = 0.8984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.08694088459, Accuracy = 0.860545992851
    Iter #266240:  Learning rate = 0.004608:   Batch Loss = 1.164523, Accuracy = 0.857421875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.12304532528, Accuracy = 0.852025747299
    Iter #270336:  Learning rate = 0.004608:   Batch Loss = 1.078694, Accuracy = 0.876953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.09252691269, Accuracy = 0.862632572651
    Iter #274432:  Learning rate = 0.004608:   Batch Loss = 1.059673, Accuracy = 0.87890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0776604414, Accuracy = 0.886802315712
    Iter #278528:  Learning rate = 0.004608:   Batch Loss = 1.126772, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.15313673019, Accuracy = 0.844548761845
    Iter #282624:  Learning rate = 0.004608:   Batch Loss = 1.097116, Accuracy = 0.861328125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.07134866714, Accuracy = 0.873587191105
    Iter #286720:  Learning rate = 0.004608:   Batch Loss = 1.086557, Accuracy = 0.8671875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.05864846706, Accuracy = 0.876717090607
    Iter #290816:  Learning rate = 0.004608:   Batch Loss = 1.184409, Accuracy = 0.861328125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.11632204056, Accuracy = 0.867675185204
    Iter #294912:  Learning rate = 0.004608:   Batch Loss = 1.116552, Accuracy = 0.857421875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.12175059319, Accuracy = 0.857416093349
    Iter #299008:  Learning rate = 0.004608:   Batch Loss = 1.091724, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.10400354862, Accuracy = 0.858807146549
    Iter #303104:  Learning rate = 0.004424:   Batch Loss = 0.986324, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.02296221256, Accuracy = 0.894279241562
    Iter #307200:  Learning rate = 0.004424:   Batch Loss = 0.998030, Accuracy = 0.89453125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.02142214775, Accuracy = 0.883150756359
    Iter #311296:  Learning rate = 0.004424:   Batch Loss = 0.980945, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.01557290554, Accuracy = 0.889932215214
    Iter #315392:  Learning rate = 0.004424:   Batch Loss = 1.009427, Accuracy = 0.88671875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.10311126709, Accuracy = 0.849243581295
    Iter #319488:  Learning rate = 0.004424:   Batch Loss = 1.058067, Accuracy = 0.8671875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.00473856926, Accuracy = 0.901408433914
    Iter #323584:  Learning rate = 0.004424:   Batch Loss = 0.994620, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.01256406307, Accuracy = 0.883324623108
    Iter #327680:  Learning rate = 0.004424:   Batch Loss = 1.095577, Accuracy = 0.853515625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.03729939461, Accuracy = 0.865762472153
    Iter #331776:  Learning rate = 0.004424:   Batch Loss = 1.038876, Accuracy = 0.873046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.989422976971, Accuracy = 0.896713614464
    Iter #335872:  Learning rate = 0.004424:   Batch Loss = 1.045440, Accuracy = 0.87890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.99707865715, Accuracy = 0.888193368912
    Iter #339968:  Learning rate = 0.004424:   Batch Loss = 1.273818, Accuracy = 0.83203125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.65911006927, Accuracy = 0.718483746052
    Iter #344064:  Learning rate = 0.004424:   Batch Loss = 1.150636, Accuracy = 0.830078125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1963326931, Accuracy = 0.808902800083
    Iter #348160:  Learning rate = 0.004424:   Batch Loss = 1.067706, Accuracy = 0.85546875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.04460120201, Accuracy = 0.869761765003
    Iter #352256:  Learning rate = 0.004424:   Batch Loss = 1.048522, Accuracy = 0.865234375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.03802132607, Accuracy = 0.876717090607
    Iter #356352:  Learning rate = 0.004424:   Batch Loss = 1.012132, Accuracy = 0.904296875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0310716629, Accuracy = 0.864893078804
    Iter #360448:  Learning rate = 0.004424:   Batch Loss = 0.989047, Accuracy = 0.888671875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.996744275093, Accuracy = 0.885585129261
    Iter #364544:  Learning rate = 0.004424:   Batch Loss = 0.977884, Accuracy = 0.912109375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.993276000023, Accuracy = 0.893409848213
    Iter #368640:  Learning rate = 0.004424:   Batch Loss = 0.963871, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.00227975845, Accuracy = 0.885932862759
    Iter #372736:  Learning rate = 0.004424:   Batch Loss = 0.930576, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.02242136002, Accuracy = 0.875152170658
    Iter #376832:  Learning rate = 0.004424:   Batch Loss = 0.947550, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.970808267593, Accuracy = 0.905755519867
    Iter #380928:  Learning rate = 0.004424:   Batch Loss = 0.952235, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.952122449875, Accuracy = 0.900712907314
    Iter #385024:  Learning rate = 0.004424:   Batch Loss = 0.912896, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.00123631954, Accuracy = 0.877586483955
    Iter #389120:  Learning rate = 0.004424:   Batch Loss = 0.947593, Accuracy = 0.89453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.96993792057, Accuracy = 0.891497135162
    Iter #393216:  Learning rate = 0.004424:   Batch Loss = 0.968514, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.956546664238, Accuracy = 0.896539747715
    Iter #397312:  Learning rate = 0.004424:   Batch Loss = 0.961573, Accuracy = 0.896484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.978764533997, Accuracy = 0.888019502163
    Iter #401408:  Learning rate = 0.004247:   Batch Loss = 0.991264, Accuracy = 0.880859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.02045118809, Accuracy = 0.868370711803
    Iter #405504:  Learning rate = 0.004247:   Batch Loss = 0.994957, Accuracy = 0.89453125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.02534663677, Accuracy = 0.866631865501
    Iter #409600:  Learning rate = 0.004247:   Batch Loss = 1.031549, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.995945394039, Accuracy = 0.875152170658
    Iter #413696:  Learning rate = 0.004247:   Batch Loss = 0.914717, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.969509243965, Accuracy = 0.898626327515
    Iter #417792:  Learning rate = 0.004247:   Batch Loss = 0.957765, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.973476171494, Accuracy = 0.874804377556
    Iter #421888:  Learning rate = 0.004247:   Batch Loss = 0.931691, Accuracy = 0.91015625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0062816143, Accuracy = 0.876021564007
    Iter #425984:  Learning rate = 0.004247:   Batch Loss = 0.894961, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.924855530262, Accuracy = 0.905059993267
    Iter #430080:  Learning rate = 0.004247:   Batch Loss = 0.954015, Accuracy = 0.87890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.992160022259, Accuracy = 0.876195430756
    Iter #434176:  Learning rate = 0.004247:   Batch Loss = 0.965562, Accuracy = 0.8984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.00343847275, Accuracy = 0.874456644058
    Iter #438272:  Learning rate = 0.004247:   Batch Loss = 0.998388, Accuracy = 0.865234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.989467918873, Accuracy = 0.880020856857
    Iter #442368:  Learning rate = 0.004247:   Batch Loss = 1.025146, Accuracy = 0.83984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.01342511177, Accuracy = 0.863675892353
    Iter #446464:  Learning rate = 0.004247:   Batch Loss = 0.858868, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.92470240593, Accuracy = 0.896191954613
    Iter #450560:  Learning rate = 0.004247:   Batch Loss = 0.871211, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.934871554375, Accuracy = 0.895148694515
    Iter #454656:  Learning rate = 0.004247:   Batch Loss = 0.878051, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9347448349, Accuracy = 0.896191954613
    Iter #458752:  Learning rate = 0.004247:   Batch Loss = 0.876325, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.922950863838, Accuracy = 0.895844221115
    Iter #462848:  Learning rate = 0.004247:   Batch Loss = 0.878252, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.907755672932, Accuracy = 0.901234567165
    Iter #466944:  Learning rate = 0.004247:   Batch Loss = 0.909490, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.894319534302, Accuracy = 0.91271084547
    Iter #471040:  Learning rate = 0.004247:   Batch Loss = 0.898837, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.926019668579, Accuracy = 0.897235274315
    Iter #475136:  Learning rate = 0.004247:   Batch Loss = 0.956898, Accuracy = 0.896484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.946250200272, Accuracy = 0.894800901413
    Iter #479232:  Learning rate = 0.004247:   Batch Loss = 0.884311, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.891948878765, Accuracy = 0.905407726765
    Iter #483328:  Learning rate = 0.004247:   Batch Loss = 0.884706, Accuracy = 0.9140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.920020639896, Accuracy = 0.894627034664
    Iter #487424:  Learning rate = 0.004247:   Batch Loss = 0.834675, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.864518523216, Accuracy = 0.920361697674
    Iter #491520:  Learning rate = 0.004247:   Batch Loss = 0.836728, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.853603303432, Accuracy = 0.924708724022
    Iter #495616:  Learning rate = 0.004247:   Batch Loss = 0.820159, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.85696709156, Accuracy = 0.922274410725
    Iter #499712:  Learning rate = 0.004247:   Batch Loss = 0.838289, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.841126024723, Accuracy = 0.923143804073
    Iter #503808:  Learning rate = 0.004077:   Batch Loss = 0.880463, Accuracy = 0.908203125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.851659417152, Accuracy = 0.920709431171
    Iter #507904:  Learning rate = 0.004077:   Batch Loss = 0.829502, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.86813378334, Accuracy = 0.910450339317
    Iter #512000:  Learning rate = 0.004077:   Batch Loss = 0.885287, Accuracy = 0.900390625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.850416243076, Accuracy = 0.919318377972
    Iter #516096:  Learning rate = 0.004077:   Batch Loss = 0.833354, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.841178774834, Accuracy = 0.926795363426
    Iter #520192:  Learning rate = 0.004077:   Batch Loss = 0.825035, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.856097221375, Accuracy = 0.913058578968
    Iter #524288:  Learning rate = 0.004077:   Batch Loss = 0.866789, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.920736849308, Accuracy = 0.892540454865
    Iter #528384:  Learning rate = 0.004077:   Batch Loss = 1.575331, Accuracy = 0.65625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.35844159126, Accuracy = 0.724048018456
    Iter #532480:  Learning rate = 0.004077:   Batch Loss = 1.152835, Accuracy = 0.779296875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.12607014179, Accuracy = 0.79777431488
    Iter #536576:  Learning rate = 0.004077:   Batch Loss = 1.070047, Accuracy = 0.837890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.01280498505, Accuracy = 0.851156294346
    Iter #540672:  Learning rate = 0.004077:   Batch Loss = 0.956712, Accuracy = 0.876953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.944431066513, Accuracy = 0.87967312336
    Iter #544768:  Learning rate = 0.004077:   Batch Loss = 0.909610, Accuracy = 0.8984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.947910785675, Accuracy = 0.877064883709
    Iter #548864:  Learning rate = 0.004077:   Batch Loss = 0.888367, Accuracy = 0.90234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.937379777431, Accuracy = 0.876369297504
    Iter #552960:  Learning rate = 0.004077:   Batch Loss = 0.925037, Accuracy = 0.884765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.895252943039, Accuracy = 0.898974120617
    Iter #557056:  Learning rate = 0.004077:   Batch Loss = 0.866913, Accuracy = 0.916015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.879227817059, Accuracy = 0.904364466667
    Iter #561152:  Learning rate = 0.004077:   Batch Loss = 0.884819, Accuracy = 0.9140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.872875273228, Accuracy = 0.908537626266
    Iter #565248:  Learning rate = 0.004077:   Batch Loss = 0.876426, Accuracy = 0.900390625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.890360534191, Accuracy = 0.898104667664
    Iter #569344:  Learning rate = 0.004077:   Batch Loss = 0.991116, Accuracy = 0.8828125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.09497725964, Accuracy = 0.833420276642
    Iter #573440:  Learning rate = 0.004077:   Batch Loss = 0.894690, Accuracy = 0.892578125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.909497261047, Accuracy = 0.886976182461
    Iter #577536:  Learning rate = 0.004077:   Batch Loss = 0.892936, Accuracy = 0.892578125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.896346211433, Accuracy = 0.895496428013
    Iter #581632:  Learning rate = 0.004077:   Batch Loss = 0.945477, Accuracy = 0.880859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.909026503563, Accuracy = 0.88436794281
    Iter #585728:  Learning rate = 0.004077:   Batch Loss = 0.863998, Accuracy = 0.900390625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.860609829426, Accuracy = 0.899321854115
    Iter #589824:  Learning rate = 0.004077:   Batch Loss = 0.894184, Accuracy = 0.90234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.887295603752, Accuracy = 0.894974768162
    Iter #593920:  Learning rate = 0.004077:   Batch Loss = 0.866985, Accuracy = 0.904296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.888579964638, Accuracy = 0.886106789112
    Iter #598016:  Learning rate = 0.004077:   Batch Loss = 0.911512, Accuracy = 0.8828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.845924556255, Accuracy = 0.907146573067
    Iter #602112:  Learning rate = 0.003914:   Batch Loss = 0.839528, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.876196742058, Accuracy = 0.893931508064
    Iter #606208:  Learning rate = 0.003914:   Batch Loss = 0.828426, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.857228636742, Accuracy = 0.91079813242
    Iter #610304:  Learning rate = 0.003914:   Batch Loss = 0.814400, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.856397867203, Accuracy = 0.899147987366
    Iter #614400:  Learning rate = 0.003914:   Batch Loss = 0.809834, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.847281873226, Accuracy = 0.903147280216
    Iter #618496:  Learning rate = 0.003914:   Batch Loss = 0.823595, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.835157036781, Accuracy = 0.903321146965
    Iter #622592:  Learning rate = 0.003914:   Batch Loss = 0.920957, Accuracy = 0.880859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.831562995911, Accuracy = 0.906972706318
    Iter #626688:  Learning rate = 0.003914:   Batch Loss = 0.824243, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.848239541054, Accuracy = 0.891671001911
    Iter #630784:  Learning rate = 0.003914:   Batch Loss = 0.815887, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.818643808365, Accuracy = 0.915145218372
    Iter #634880:  Learning rate = 0.003914:   Batch Loss = 0.827680, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.811751186848, Accuracy = 0.91410189867
    Iter #638976:  Learning rate = 0.003914:   Batch Loss = 0.812697, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.809740304947, Accuracy = 0.91618847847
    Iter #643072:  Learning rate = 0.003914:   Batch Loss = 0.805460, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.876259684563, Accuracy = 0.893931508064
    Iter #647168:  Learning rate = 0.003914:   Batch Loss = 0.800154, Accuracy = 0.9296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.843537807465, Accuracy = 0.913232505322
    Iter #651264:  Learning rate = 0.003914:   Batch Loss = 0.837086, Accuracy = 0.908203125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.85007750988, Accuracy = 0.89184486866
    Iter #655360:  Learning rate = 0.003914:   Batch Loss = 0.796072, Accuracy = 0.92578125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.841115355492, Accuracy = 0.907320439816
    Iter #659456:  Learning rate = 0.003914:   Batch Loss = 0.851598, Accuracy = 0.9140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.839597702026, Accuracy = 0.911667525768
    Iter #663552:  Learning rate = 0.003914:   Batch Loss = 0.776145, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.800803482533, Accuracy = 0.916536271572
    Iter #667648:  Learning rate = 0.003914:   Batch Loss = 0.806620, Accuracy = 0.91015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.813037097454, Accuracy = 0.917057931423
    Iter #671744:  Learning rate = 0.003914:   Batch Loss = 0.804169, Accuracy = 0.908203125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.810933768749, Accuracy = 0.912363052368
    Iter #675840:  Learning rate = 0.003914:   Batch Loss = 0.817089, Accuracy = 0.9140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.819456040859, Accuracy = 0.903668940067
    Iter #679936:  Learning rate = 0.003914:   Batch Loss = 0.796127, Accuracy = 0.904296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.789317786694, Accuracy = 0.922448277473
    Iter #684032:  Learning rate = 0.003914:   Batch Loss = 0.812355, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.791621088982, Accuracy = 0.913580238819
    Iter #688128:  Learning rate = 0.003914:   Batch Loss = 0.787167, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.821958780289, Accuracy = 0.903842806816
    Iter #692224:  Learning rate = 0.003914:   Batch Loss = 0.804830, Accuracy = 0.9296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.85359621048, Accuracy = 0.894627034664
    Iter #696320:  Learning rate = 0.003914:   Batch Loss = 0.758409, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.797425627708, Accuracy = 0.911319792271
    Iter #700416:  Learning rate = 0.003757:   Batch Loss = 0.758824, Accuracy = 0.9296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.777521014214, Accuracy = 0.921926617622
    Iter #704512:  Learning rate = 0.003757:   Batch Loss = 0.783976, Accuracy = 0.912109375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.788235664368, Accuracy = 0.917927324772
    Iter #708608:  Learning rate = 0.003757:   Batch Loss = 0.752631, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.775636553764, Accuracy = 0.918622851372
    Iter #712704:  Learning rate = 0.003757:   Batch Loss = 0.790168, Accuracy = 0.9140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.785072565079, Accuracy = 0.920013904572
    Iter #716800:  Learning rate = 0.003757:   Batch Loss = 0.743440, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.789997041225, Accuracy = 0.916710138321
    Iter #720896:  Learning rate = 0.003757:   Batch Loss = 0.802323, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.789044260979, Accuracy = 0.915145218372
    Iter #724992:  Learning rate = 0.003757:   Batch Loss = 0.755107, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.799500465393, Accuracy = 0.912884712219
    Iter #729088:  Learning rate = 0.003757:   Batch Loss = 0.861987, Accuracy = 0.884765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.798428654671, Accuracy = 0.908363759518
    Iter #733184:  Learning rate = 0.003757:   Batch Loss = 0.803933, Accuracy = 0.908203125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.814080476761, Accuracy = 0.907494366169
    Iter #737280:  Learning rate = 0.003757:   Batch Loss = 0.799435, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.774695754051, Accuracy = 0.925752043724
    Iter #741376:  Learning rate = 0.003757:   Batch Loss = 0.771688, Accuracy = 0.9140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.794607162476, Accuracy = 0.902103960514
    Iter #745472:  Learning rate = 0.003757:   Batch Loss = 0.754050, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.807493567467, Accuracy = 0.90940707922
    Iter #749568:  Learning rate = 0.003757:   Batch Loss = 0.776635, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.788853526115, Accuracy = 0.91271084547
    Iter #753664:  Learning rate = 0.003757:   Batch Loss = 0.780404, Accuracy = 0.9296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.797102868557, Accuracy = 0.90801602602
    Iter #757760:  Learning rate = 0.003757:   Batch Loss = 0.825089, Accuracy = 0.91015625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.04161834717, Accuracy = 0.845592081547
    Iter #761856:  Learning rate = 0.003757:   Batch Loss = 0.950108, Accuracy = 0.857421875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.06441175938, Accuracy = 0.804729640484
    Iter #765952:  Learning rate = 0.003757:   Batch Loss = 0.832553, Accuracy = 0.884765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.85548722744, Accuracy = 0.88036864996
    Iter #770048:  Learning rate = 0.003757:   Batch Loss = 0.810594, Accuracy = 0.91015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.807729482651, Accuracy = 0.895670294762
    Iter #774144:  Learning rate = 0.003757:   Batch Loss = 0.757618, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.793379187584, Accuracy = 0.912015318871
    Iter #778240:  Learning rate = 0.003757:   Batch Loss = 0.780720, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.795503079891, Accuracy = 0.905755519867
    Iter #782336:  Learning rate = 0.003757:   Batch Loss = 0.831041, Accuracy = 0.900390625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.858609557152, Accuracy = 0.884889602661
    Iter #786432:  Learning rate = 0.003757:   Batch Loss = 0.847450, Accuracy = 0.8828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.875633239746, Accuracy = 0.892192661762
    Iter #790528:  Learning rate = 0.003757:   Batch Loss = 0.800875, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.90530872345, Accuracy = 0.8614153862
    Iter #794624:  Learning rate = 0.003757:   Batch Loss = 0.823337, Accuracy = 0.896484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.786111950874, Accuracy = 0.912015318871
    Iter #798720:  Learning rate = 0.003757:   Batch Loss = 0.787568, Accuracy = 0.91015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.777332723141, Accuracy = 0.91010260582
    Iter #802816:  Learning rate = 0.003607:   Batch Loss = 0.782602, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.805973172188, Accuracy = 0.902277886868
    Iter #806912:  Learning rate = 0.003607:   Batch Loss = 0.779580, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.76156270504, Accuracy = 0.914971292019
    Iter #811008:  Learning rate = 0.003607:   Batch Loss = 0.751130, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.745925962925, Accuracy = 0.923839330673
    Iter #815104:  Learning rate = 0.003607:   Batch Loss = 0.766130, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.774173498154, Accuracy = 0.907842099667
    Iter #819200:  Learning rate = 0.003607:   Batch Loss = 0.803578, Accuracy = 0.88671875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.75128787756, Accuracy = 0.926099836826
    Iter #823296:  Learning rate = 0.003607:   Batch Loss = 0.727057, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.749192595482, Accuracy = 0.920709431171
    Iter #827392:  Learning rate = 0.003607:   Batch Loss = 0.774633, Accuracy = 0.91015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.734124898911, Accuracy = 0.928881943226
    Iter #831488:  Learning rate = 0.003607:   Batch Loss = 0.778091, Accuracy = 0.91015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.751571238041, Accuracy = 0.920535564423
    Iter #835584:  Learning rate = 0.003607:   Batch Loss = 0.737282, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.754064619541, Accuracy = 0.91410189867
    Iter #839680:  Learning rate = 0.003607:   Batch Loss = 1.044268, Accuracy = 0.818359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.879606485367, Accuracy = 0.869761765003
    Iter #843776:  Learning rate = 0.003607:   Batch Loss = 0.865865, Accuracy = 0.88671875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.847263991833, Accuracy = 0.886280655861
    Iter #847872:  Learning rate = 0.003607:   Batch Loss = 0.793591, Accuracy = 0.9140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.881714940071, Accuracy = 0.880020856857
    Iter #851968:  Learning rate = 0.003607:   Batch Loss = 0.722958, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.777040779591, Accuracy = 0.913928031921
    Iter #856064:  Learning rate = 0.003607:   Batch Loss = 0.770231, Accuracy = 0.912109375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.761781394482, Accuracy = 0.91688400507
    Iter #860160:  Learning rate = 0.003607:   Batch Loss = 0.761634, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.763178288937, Accuracy = 0.914971292019
    Iter #864256:  Learning rate = 0.003607:   Batch Loss = 0.881197, Accuracy = 0.888671875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.911143660545, Accuracy = 0.874108850956
    Iter #868352:  Learning rate = 0.003607:   Batch Loss = 0.759176, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.765720486641, Accuracy = 0.91340637207
    Iter #872448:  Learning rate = 0.003607:   Batch Loss = 0.791288, Accuracy = 0.91015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.774572372437, Accuracy = 0.908363759518
    Iter #876544:  Learning rate = 0.003607:   Batch Loss = 0.710074, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.732242465019, Accuracy = 0.925230383873
    Iter #880640:  Learning rate = 0.003607:   Batch Loss = 0.721644, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.752512991428, Accuracy = 0.918101191521
    Iter #884736:  Learning rate = 0.003607:   Batch Loss = 0.763135, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.805955171585, Accuracy = 0.892540454865
    Iter #888832:  Learning rate = 0.003607:   Batch Loss = 0.758742, Accuracy = 0.9140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.801658987999, Accuracy = 0.903495073318
    Iter #892928:  Learning rate = 0.003607:   Batch Loss = 0.800088, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.793779015541, Accuracy = 0.901060700417
    Iter #897024:  Learning rate = 0.003607:   Batch Loss = 0.708966, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.760018110275, Accuracy = 0.909580945969
    Iter #901120:  Learning rate = 0.003463:   Batch Loss = 0.850657, Accuracy = 0.8828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.795417368412, Accuracy = 0.902451753616
    Iter #905216:  Learning rate = 0.003463:   Batch Loss = 0.750533, Accuracy = 0.91015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.753048419952, Accuracy = 0.92088329792
    Iter #909312:  Learning rate = 0.003463:   Batch Loss = 0.703903, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.776239156723, Accuracy = 0.896365821362
    Iter #913408:  Learning rate = 0.003463:   Batch Loss = 0.742773, Accuracy = 0.912109375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.736223578453, Accuracy = 0.921231091022
    Iter #917504:  Learning rate = 0.003463:   Batch Loss = 0.757043, Accuracy = 0.92578125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.767543673515, Accuracy = 0.906798839569
    Iter #921600:  Learning rate = 0.003463:   Batch Loss = 0.727461, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.731266856194, Accuracy = 0.925925910473
    Iter #925696:  Learning rate = 0.003463:   Batch Loss = 0.694188, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.719542086124, Accuracy = 0.927664756775
    Iter #929792:  Learning rate = 0.003463:   Batch Loss = 0.714977, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.725893199444, Accuracy = 0.920535564423
    Iter #933888:  Learning rate = 0.003463:   Batch Loss = 0.737835, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.748510241508, Accuracy = 0.921231091022
    Iter #937984:  Learning rate = 0.003463:   Batch Loss = 0.747734, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.807438015938, Accuracy = 0.888541102409
    Iter #942080:  Learning rate = 0.003463:   Batch Loss = 0.697164, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.724835932255, Accuracy = 0.923665463924
    Iter #946176:  Learning rate = 0.003463:   Batch Loss = 0.695094, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.720028996468, Accuracy = 0.935315608978
    Iter #950272:  Learning rate = 0.003463:   Batch Loss = 0.674766, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.738963782787, Accuracy = 0.91479742527
    Iter #954368:  Learning rate = 0.003463:   Batch Loss = 0.699598, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.76727950573, Accuracy = 0.903321146965
    Iter #958464:  Learning rate = 0.003463:   Batch Loss = 0.694827, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.746849179268, Accuracy = 0.91410189867
    Iter #962560:  Learning rate = 0.003463:   Batch Loss = 0.965101, Accuracy = 0.837890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.943105757236, Accuracy = 0.843331575394
    Iter #966656:  Learning rate = 0.003463:   Batch Loss = 0.896669, Accuracy = 0.876953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.941865742207, Accuracy = 0.860372126102
    Iter #970752:  Learning rate = 0.003463:   Batch Loss = 0.802726, Accuracy = 0.908203125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.866376638412, Accuracy = 0.870109558105
    Iter #974848:  Learning rate = 0.003463:   Batch Loss = 0.790059, Accuracy = 0.90234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.823079824448, Accuracy = 0.880716383457
    Iter #978944:  Learning rate = 0.003463:   Batch Loss = 0.747170, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.778214633465, Accuracy = 0.898104667664
    Iter #983040:  Learning rate = 0.003463:   Batch Loss = 0.723561, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.761344671249, Accuracy = 0.909754812717
    Iter #987136:  Learning rate = 0.003463:   Batch Loss = 0.765129, Accuracy = 0.912109375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.800669550896, Accuracy = 0.885411262512
    Iter #991232:  Learning rate = 0.003463:   Batch Loss = 0.758510, Accuracy = 0.9140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.742921888828, Accuracy = 0.920013904572
    Iter #995328:  Learning rate = 0.003463:   Batch Loss = 0.763158, Accuracy = 0.900390625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.724748969078, Accuracy = 0.923839330673
    Iter #999424:  Learning rate = 0.003463:   Batch Loss = 0.681796, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.724787175655, Accuracy = 0.921926617622
    Iter #1003520:  Learning rate = 0.003324:   Batch Loss = 0.690316, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.72168380022, Accuracy = 0.922969937325
    Iter #1007616:  Learning rate = 0.003324:   Batch Loss = 0.688456, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.710611522198, Accuracy = 0.924708724022
    Iter #1011712:  Learning rate = 0.003324:   Batch Loss = 0.683703, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.716839432716, Accuracy = 0.929925203323
    Iter #1015808:  Learning rate = 0.003324:   Batch Loss = 0.663091, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.745573759079, Accuracy = 0.91479742527
    Iter #1019904:  Learning rate = 0.003324:   Batch Loss = 0.666109, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.702327728271, Accuracy = 0.931142389774
    Iter #1024000:  Learning rate = 0.003324:   Batch Loss = 0.688113, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.699773192406, Accuracy = 0.933055102825
    Iter #1028096:  Learning rate = 0.003324:   Batch Loss = 0.803408, Accuracy = 0.888671875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.783278465271, Accuracy = 0.901756227016
    Iter #1032192:  Learning rate = 0.003324:   Batch Loss = 0.716069, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.821259975433, Accuracy = 0.897235274315
    Iter #1036288:  Learning rate = 0.003324:   Batch Loss = 0.751052, Accuracy = 0.91015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.745138943195, Accuracy = 0.921231091022
    Iter #1040384:  Learning rate = 0.003324:   Batch Loss = 0.703932, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.766361474991, Accuracy = 0.905233860016
    Iter #1044480:  Learning rate = 0.003324:   Batch Loss = 0.694189, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.718913674355, Accuracy = 0.923839330673
    Iter #1048576:  Learning rate = 0.003324:   Batch Loss = 0.725809, Accuracy = 0.912109375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.720657289028, Accuracy = 0.922100484371
    Iter #1052672:  Learning rate = 0.003324:   Batch Loss = 0.660377, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.715163826942, Accuracy = 0.927143096924
    Iter #1056768:  Learning rate = 0.003324:   Batch Loss = 0.817615, Accuracy = 0.89453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.988829016685, Accuracy = 0.825421690941
    Iter #1060864:  Learning rate = 0.003324:   Batch Loss = 0.871058, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.852286934853, Accuracy = 0.864545285702
    Iter #1064960:  Learning rate = 0.003324:   Batch Loss = 0.750994, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.792659282684, Accuracy = 0.892714321613
    Iter #1069056:  Learning rate = 0.003324:   Batch Loss = 0.752916, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.788265824318, Accuracy = 0.895496428013
    Iter #1073152:  Learning rate = 0.003324:   Batch Loss = 0.794721, Accuracy = 0.884765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.798053026199, Accuracy = 0.890106081963
    Iter #1077248:  Learning rate = 0.003324:   Batch Loss = 0.683748, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.754388391972, Accuracy = 0.900191247463
    Iter #1081344:  Learning rate = 0.003324:   Batch Loss = 0.783891, Accuracy = 0.892578125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.826525092125, Accuracy = 0.879499197006
    Iter #1085440:  Learning rate = 0.003324:   Batch Loss = 0.708806, Accuracy = 0.9296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.744448065758, Accuracy = 0.91271084547
    Iter #1089536:  Learning rate = 0.003324:   Batch Loss = 0.717720, Accuracy = 0.9140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.714300453663, Accuracy = 0.919318377972
    Iter #1093632:  Learning rate = 0.003324:   Batch Loss = 0.689205, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.770590603352, Accuracy = 0.901408433914
    Iter #1097728:  Learning rate = 0.003324:   Batch Loss = 0.721471, Accuracy = 0.92578125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.759981989861, Accuracy = 0.90871155262
    Iter #1101824:  Learning rate = 0.003191:   Batch Loss = 0.681274, Accuracy = 0.9296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.72883105278, Accuracy = 0.91340637207
    Iter #1105920:  Learning rate = 0.003191:   Batch Loss = 0.658529, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.699233174324, Accuracy = 0.932707369328
    Iter #1110016:  Learning rate = 0.003191:   Batch Loss = 0.662977, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.687258660793, Accuracy = 0.930446863174
    Iter #1114112:  Learning rate = 0.003191:   Batch Loss = 0.672585, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.703345417976, Accuracy = 0.924882650375
    Iter #1118208:  Learning rate = 0.003191:   Batch Loss = 0.649114, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.743745684624, Accuracy = 0.907146573067
    Iter #1122304:  Learning rate = 0.003191:   Batch Loss = 0.688640, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.71540248394, Accuracy = 0.922622144222
    Iter #1126400:  Learning rate = 0.003191:   Batch Loss = 0.669561, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.700877308846, Accuracy = 0.921057224274
    Iter #1130496:  Learning rate = 0.003191:   Batch Loss = 0.700884, Accuracy = 0.916015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.691003680229, Accuracy = 0.930099129677
    Iter #1134592:  Learning rate = 0.003191:   Batch Loss = 0.709217, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.759344220161, Accuracy = 0.904538333416
    Iter #1138688:  Learning rate = 0.003191:   Batch Loss = 0.682504, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.69336616993, Accuracy = 0.923665463924
    Iter #1142784:  Learning rate = 0.003191:   Batch Loss = 0.670215, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.728442549706, Accuracy = 0.911841392517
    Iter #1146880:  Learning rate = 0.003191:   Batch Loss = 0.646213, Accuracy = 0.9453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.734402477741, Accuracy = 0.90801602602
    Iter #1150976:  Learning rate = 0.003191:   Batch Loss = 0.661416, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.62763786316, Accuracy = 0.525126039982
    Iter #1155072:  Learning rate = 0.003191:   Batch Loss = 1.082134, Accuracy = 0.732421875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.00131726265, Accuracy = 0.803512454033
    Iter #1159168:  Learning rate = 0.003191:   Batch Loss = 0.956906, Accuracy = 0.833984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.859838724136, Accuracy = 0.863328099251
    Iter #1163264:  Learning rate = 0.003191:   Batch Loss = 0.813028, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.939559817314, Accuracy = 0.869066238403
    Iter #1167360:  Learning rate = 0.003191:   Batch Loss = 0.749933, Accuracy = 0.916015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.883615791798, Accuracy = 0.865762472153
    Iter #1171456:  Learning rate = 0.003191:   Batch Loss = 0.801750, Accuracy = 0.87890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.80298101902, Accuracy = 0.881064176559
    Iter #1175552:  Learning rate = 0.003191:   Batch Loss = 0.729984, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.779083848, Accuracy = 0.897409141064
    Iter #1179648:  Learning rate = 0.003191:   Batch Loss = 0.731781, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.710050463676, Accuracy = 0.928012490273
    Iter #1183744:  Learning rate = 0.003191:   Batch Loss = 0.682568, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.782557308674, Accuracy = 0.903495073318
    Iter #1187840:  Learning rate = 0.003191:   Batch Loss = 0.676786, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.728367805481, Accuracy = 0.923143804073
    Iter #1191936:  Learning rate = 0.003191:   Batch Loss = 0.715444, Accuracy = 0.916015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.732080519199, Accuracy = 0.911667525768
    Iter #1196032:  Learning rate = 0.003191:   Batch Loss = 0.688803, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.711815297604, Accuracy = 0.921404957771
    Iter #1200128:  Learning rate = 0.003064:   Batch Loss = 0.699238, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.72886121273, Accuracy = 0.917231798172
    Iter #1204224:  Learning rate = 0.003064:   Batch Loss = 0.672276, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.721914410591, Accuracy = 0.917405664921
    Iter #1208320:  Learning rate = 0.003064:   Batch Loss = 0.658381, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.698962807655, Accuracy = 0.925056517124
    Iter #1212416:  Learning rate = 0.003064:   Batch Loss = 0.663931, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.743902683258, Accuracy = 0.912363052368
    Iter #1216512:  Learning rate = 0.003064:   Batch Loss = 0.647647, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.799393594265, Accuracy = 0.888193368912
    Iter #1220608:  Learning rate = 0.003064:   Batch Loss = 0.716091, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.729352355003, Accuracy = 0.914623558521
    Iter #1224704:  Learning rate = 0.003064:   Batch Loss = 0.678453, Accuracy = 0.9296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.703038930893, Accuracy = 0.921231091022
    Iter #1228800:  Learning rate = 0.003064:   Batch Loss = 0.699907, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.734634637833, Accuracy = 0.91079813242
    Iter #1232896:  Learning rate = 0.003064:   Batch Loss = 0.661218, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.686828255653, Accuracy = 0.925578176975
    Iter #1236992:  Learning rate = 0.003064:   Batch Loss = 0.631566, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.682531654835, Accuracy = 0.929577469826
    Iter #1241088:  Learning rate = 0.003064:   Batch Loss = 0.644403, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.669500946999, Accuracy = 0.929577469826
    Iter #1245184:  Learning rate = 0.003064:   Batch Loss = 0.641497, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.708920836449, Accuracy = 0.923839330673
    Iter #1249280:  Learning rate = 0.003064:   Batch Loss = 0.834770, Accuracy = 0.884765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.866336405277, Accuracy = 0.857068359852
    Iter #1253376:  Learning rate = 0.003064:   Batch Loss = 0.724560, Accuracy = 0.912109375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.775895476341, Accuracy = 0.890279948711
    Iter #1257472:  Learning rate = 0.003064:   Batch Loss = 0.732121, Accuracy = 0.92578125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.721507191658, Accuracy = 0.931664049625
    Iter #1261568:  Learning rate = 0.003064:   Batch Loss = 0.700967, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.76562076807, Accuracy = 0.91079813242
    Iter #1265664:  Learning rate = 0.003064:   Batch Loss = 0.701309, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.719761371613, Accuracy = 0.910971999168
    Iter #1269760:  Learning rate = 0.003064:   Batch Loss = 0.668656, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.72073161602, Accuracy = 0.912884712219
    Iter #1273856:  Learning rate = 0.003064:   Batch Loss = 0.706861, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.726877808571, Accuracy = 0.904538333416
    Iter #1277952:  Learning rate = 0.003064:   Batch Loss = 0.698010, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.702741205692, Accuracy = 0.913058578968
    Iter #1282048:  Learning rate = 0.003064:   Batch Loss = 0.700327, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.69996380806, Accuracy = 0.923317670822
    Iter #1286144:  Learning rate = 0.003064:   Batch Loss = 0.669793, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.69402974844, Accuracy = 0.921926617622
    Iter #1290240:  Learning rate = 0.003064:   Batch Loss = 0.678202, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.693642735481, Accuracy = 0.929751336575
    Iter #1294336:  Learning rate = 0.003064:   Batch Loss = 0.722701, Accuracy = 0.908203125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.719504654408, Accuracy = 0.909928679466
    Iter #1298432:  Learning rate = 0.003064:   Batch Loss = 0.678803, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.66567504406, Accuracy = 0.930272996426
    Iter #1302528:  Learning rate = 0.002941:   Batch Loss = 0.602615, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6551861763, Accuracy = 0.936706662178
    Iter #1306624:  Learning rate = 0.002941:   Batch Loss = 0.640220, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.66836977005, Accuracy = 0.931837916374
    Iter #1310720:  Learning rate = 0.002941:   Batch Loss = 0.666273, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.690253257751, Accuracy = 0.918448984623
    Iter #1314816:  Learning rate = 0.002941:   Batch Loss = 0.660718, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.666838347912, Accuracy = 0.932185709476
    Iter #1318912:  Learning rate = 0.002941:   Batch Loss = 0.645685, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.707437336445, Accuracy = 0.91410189867
    Iter #1323008:  Learning rate = 0.002941:   Batch Loss = 0.616287, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.675169110298, Accuracy = 0.934098422527
    Iter #1327104:  Learning rate = 0.002941:   Batch Loss = 0.689656, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.707977592945, Accuracy = 0.91479742527
    Iter #1331200:  Learning rate = 0.002941:   Batch Loss = 0.647947, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.664164304733, Accuracy = 0.930446863174
    Iter #1335296:  Learning rate = 0.002941:   Batch Loss = 0.677020, Accuracy = 0.9296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.699986159801, Accuracy = 0.916710138321
    Iter #1339392:  Learning rate = 0.002941:   Batch Loss = 0.636646, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.783833622932, Accuracy = 0.881585836411
    Iter #1343488:  Learning rate = 0.002941:   Batch Loss = 0.699503, Accuracy = 0.916015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.677273511887, Accuracy = 0.920709431171
    Iter #1347584:  Learning rate = 0.002941:   Batch Loss = 0.680374, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.715898036957, Accuracy = 0.908885419369
    Iter #1351680:  Learning rate = 0.002941:   Batch Loss = 0.693649, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.692663788795, Accuracy = 0.91949224472
    Iter #1355776:  Learning rate = 0.002941:   Batch Loss = 0.657728, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.699551343918, Accuracy = 0.922622144222
    Iter #1359872:  Learning rate = 0.002941:   Batch Loss = 0.688433, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.712745249271, Accuracy = 0.913232505322
    Iter #1363968:  Learning rate = 0.002941:   Batch Loss = 0.625594, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.704366564751, Accuracy = 0.907668232918
    Iter #1368064:  Learning rate = 0.002941:   Batch Loss = 0.650211, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.672032594681, Accuracy = 0.918101191521
    Iter #1372160:  Learning rate = 0.002941:   Batch Loss = 0.665270, Accuracy = 0.92578125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.692314982414, Accuracy = 0.915319085121
    Iter #1376256:  Learning rate = 0.002941:   Batch Loss = 0.664108, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.683425307274, Accuracy = 0.91757953167
    Iter #1380352:  Learning rate = 0.002941:   Batch Loss = 0.635835, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.66362029314, Accuracy = 0.928534150124
    Iter #1384448:  Learning rate = 0.002941:   Batch Loss = 0.662293, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.665395975113, Accuracy = 0.922448277473
    Iter #1388544:  Learning rate = 0.002941:   Batch Loss = 0.628084, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.643435657024, Accuracy = 0.931316316128
    Iter #1392640:  Learning rate = 0.002941:   Batch Loss = 0.604049, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.657763421535, Accuracy = 0.929403603077
    Iter #1396736:  Learning rate = 0.002941:   Batch Loss = 0.677753, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.687112390995, Accuracy = 0.922622144222
    Iter #1400832:  Learning rate = 0.002823:   Batch Loss = 0.625342, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.65788936615, Accuracy = 0.930794656277
    Iter #1404928:  Learning rate = 0.002823:   Batch Loss = 0.630122, Accuracy = 0.9453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.677473187447, Accuracy = 0.924187123775
    Iter #1409024:  Learning rate = 0.002823:   Batch Loss = 0.634090, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.690595269203, Accuracy = 0.91618847847
    Iter #1413120:  Learning rate = 0.002823:   Batch Loss = 0.603632, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.662839531898, Accuracy = 0.931142389774
    Iter #1417216:  Learning rate = 0.002823:   Batch Loss = 0.609831, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.646994829178, Accuracy = 0.934446156025
    Iter #1421312:  Learning rate = 0.002823:   Batch Loss = 0.607376, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.641695857048, Accuracy = 0.931664049625
    Iter #1425408:  Learning rate = 0.002823:   Batch Loss = 0.622849, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.624075174332, Accuracy = 0.940184295177
    Iter #1429504:  Learning rate = 0.002823:   Batch Loss = 0.597163, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.653875470161, Accuracy = 0.926273703575
    Iter #1433600:  Learning rate = 0.002823:   Batch Loss = 0.594033, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.640423536301, Accuracy = 0.933924555779
    Iter #1437696:  Learning rate = 0.002823:   Batch Loss = 0.627318, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.660998940468, Accuracy = 0.925752043724
    Iter #1441792:  Learning rate = 0.002823:   Batch Loss = 0.677080, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.655606985092, Accuracy = 0.927143096924
    Iter #1445888:  Learning rate = 0.002823:   Batch Loss = 0.626950, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.739085376263, Accuracy = 0.903147280216
    Iter #1449984:  Learning rate = 0.002823:   Batch Loss = 0.630460, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.659812510014, Accuracy = 0.925578176975
    Iter #1454080:  Learning rate = 0.002823:   Batch Loss = 0.660828, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.641825199127, Accuracy = 0.931664049625
    Iter #1458176:  Learning rate = 0.002823:   Batch Loss = 0.608296, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.652191519737, Accuracy = 0.926447570324
    Iter #1462272:  Learning rate = 0.002823:   Batch Loss = 0.697049, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.740612387657, Accuracy = 0.903495073318
    Iter #1466368:  Learning rate = 0.002823:   Batch Loss = 0.637429, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.658672451973, Accuracy = 0.932185709476
    Iter #1470464:  Learning rate = 0.002823:   Batch Loss = 0.623241, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.715715706348, Accuracy = 0.905407726765
    Iter #1474560:  Learning rate = 0.002823:   Batch Loss = 0.586810, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.670774519444, Accuracy = 0.929229676723
    Iter #1478656:  Learning rate = 0.002823:   Batch Loss = 0.621061, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.696702361107, Accuracy = 0.91410189867
    Iter #1482752:  Learning rate = 0.002823:   Batch Loss = 0.619216, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.670585691929, Accuracy = 0.925404250622
    Iter #1486848:  Learning rate = 0.002823:   Batch Loss = 0.629931, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.65627849102, Accuracy = 0.926099836826
    Iter #1490944:  Learning rate = 0.002823:   Batch Loss = 0.596044, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.659738898277, Accuracy = 0.926969230175
    Iter #1495040:  Learning rate = 0.002823:   Batch Loss = 0.634664, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.636839270592, Accuracy = 0.934098422527
    Iter #1499136:  Learning rate = 0.002823:   Batch Loss = 0.600852, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.627721965313, Accuracy = 0.935663342476
    Iter #1503232:  Learning rate = 0.002710:   Batch Loss = 0.634106, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.667750835419, Accuracy = 0.922100484371
    Iter #1507328:  Learning rate = 0.002710:   Batch Loss = 0.598516, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.639641880989, Accuracy = 0.933750629425
    Iter #1511424:  Learning rate = 0.002710:   Batch Loss = 0.580128, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.653562903404, Accuracy = 0.937228322029
    Iter #1515520:  Learning rate = 0.002710:   Batch Loss = 0.619445, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.644525885582, Accuracy = 0.928881943226
    Iter #1519616:  Learning rate = 0.002710:   Batch Loss = 0.625973, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.699604988098, Accuracy = 0.917927324772
    Iter #1523712:  Learning rate = 0.002710:   Batch Loss = 0.634093, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.670182347298, Accuracy = 0.92088329792
    Iter #1527808:  Learning rate = 0.002710:   Batch Loss = 0.632183, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.614475011826, Accuracy = 0.948356807232
    Iter #1531904:  Learning rate = 0.002710:   Batch Loss = 0.582478, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.643819987774, Accuracy = 0.932533442974
    Iter #1536000:  Learning rate = 0.002710:   Batch Loss = 0.578969, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.596841394901, Accuracy = 0.951660573483
    Iter #1540096:  Learning rate = 0.002710:   Batch Loss = 0.589900, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.603247761726, Accuracy = 0.94453138113
    Iter #1544192:  Learning rate = 0.002710:   Batch Loss = 0.593655, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.633886694908, Accuracy = 0.930968523026
    Iter #1548288:  Learning rate = 0.002710:   Batch Loss = 0.584145, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.633568108082, Accuracy = 0.934967815876
    Iter #1552384:  Learning rate = 0.002710:   Batch Loss = 0.571741, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.684944272041, Accuracy = 0.923143804073
    Iter #1556480:  Learning rate = 0.002710:   Batch Loss = 0.595781, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.630089342594, Accuracy = 0.931316316128
    Iter #1560576:  Learning rate = 0.002710:   Batch Loss = 0.608940, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.647219479084, Accuracy = 0.927143096924
    Iter #1564672:  Learning rate = 0.002710:   Batch Loss = 0.645999, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.714585542679, Accuracy = 0.908885419369
    Iter #1568768:  Learning rate = 0.002710:   Batch Loss = 0.736678, Accuracy = 0.89453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.728846311569, Accuracy = 0.905581653118
    Iter #1572864:  Learning rate = 0.002710:   Batch Loss = 0.696846, Accuracy = 0.916015625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.72114020586, Accuracy = 0.906451046467
    Iter #1576960:  Learning rate = 0.002710:   Batch Loss = 0.623003, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.689772367477, Accuracy = 0.919840037823
    Iter #1581056:  Learning rate = 0.002710:   Batch Loss = 0.638854, Accuracy = 0.9453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.667924821377, Accuracy = 0.923317670822
    Iter #1585152:  Learning rate = 0.002710:   Batch Loss = 0.653393, Accuracy = 0.92578125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.64829236269, Accuracy = 0.931664049625
    Iter #1589248:  Learning rate = 0.002710:   Batch Loss = 0.663109, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.669057190418, Accuracy = 0.915666818619
    Iter #1593344:  Learning rate = 0.002710:   Batch Loss = 0.586508, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.630834937096, Accuracy = 0.936706662178
    Iter #1597440:  Learning rate = 0.002710:   Batch Loss = 0.642940, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.648498773575, Accuracy = 0.918101191521
    Iter #1601536:  Learning rate = 0.002602:   Batch Loss = 0.589864, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.671922683716, Accuracy = 0.919840037823
    Iter #1605632:  Learning rate = 0.002602:   Batch Loss = 0.636202, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.63825327158, Accuracy = 0.931837916374
    Iter #1609728:  Learning rate = 0.002602:   Batch Loss = 0.630878, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.688216686249, Accuracy = 0.90940707922
    Iter #1613824:  Learning rate = 0.002602:   Batch Loss = 0.605715, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.665169596672, Accuracy = 0.923665463924
    Iter #1617920:  Learning rate = 0.002602:   Batch Loss = 0.540509, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.638888716698, Accuracy = 0.932359576225
    Iter #1622016:  Learning rate = 0.002602:   Batch Loss = 0.618567, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.678462505341, Accuracy = 0.911841392517
    Iter #1626112:  Learning rate = 0.002602:   Batch Loss = 0.704185, Accuracy = 0.908203125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.649471998215, Accuracy = 0.924534857273
    Iter #1630208:  Learning rate = 0.002602:   Batch Loss = 0.599861, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.670994400978, Accuracy = 0.912189185619
    Iter #1634304:  Learning rate = 0.002602:   Batch Loss = 0.600913, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.627527177334, Accuracy = 0.930099129677
    Iter #1638400:  Learning rate = 0.002602:   Batch Loss = 0.604360, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.687577188015, Accuracy = 0.91549295187
    Iter #1642496:  Learning rate = 0.002602:   Batch Loss = 0.582753, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.643537759781, Accuracy = 0.926795363426
    Iter #1646592:  Learning rate = 0.002602:   Batch Loss = 0.600239, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.609202504158, Accuracy = 0.941923141479
    Iter #1650688:  Learning rate = 0.002602:   Batch Loss = 0.605211, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.606323838234, Accuracy = 0.937228322029
    Iter #1654784:  Learning rate = 0.002602:   Batch Loss = 0.588831, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.601177334785, Accuracy = 0.944009721279
    Iter #1658880:  Learning rate = 0.002602:   Batch Loss = 0.562701, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.640897631645, Accuracy = 0.929577469826
    Iter #1662976:  Learning rate = 0.002602:   Batch Loss = 0.567638, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.599541187286, Accuracy = 0.943488061428
    Iter #1667072:  Learning rate = 0.002602:   Batch Loss = 0.583517, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.604779422283, Accuracy = 0.943661987782
    Iter #1671168:  Learning rate = 0.002602:   Batch Loss = 0.589517, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.695907473564, Accuracy = 0.909059286118
    Iter #1675264:  Learning rate = 0.002602:   Batch Loss = 0.629315, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.640697956085, Accuracy = 0.92088329792
    Iter #1679360:  Learning rate = 0.002602:   Batch Loss = 0.570732, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.642328321934, Accuracy = 0.92018777132
    Iter #1683456:  Learning rate = 0.002602:   Batch Loss = 0.678469, Accuracy = 0.919921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.656460881233, Accuracy = 0.920535564423
    Iter #1687552:  Learning rate = 0.002602:   Batch Loss = 0.600802, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.644771397114, Accuracy = 0.925230383873
    Iter #1691648:  Learning rate = 0.002602:   Batch Loss = 0.658497, Accuracy = 0.91796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.644363939762, Accuracy = 0.918970584869
    Iter #1695744:  Learning rate = 0.002602:   Batch Loss = 0.570502, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.618289232254, Accuracy = 0.930968523026
    Iter #1699840:  Learning rate = 0.002602:   Batch Loss = 0.590384, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.617018222809, Accuracy = 0.938793241978
    Iter #1703936:  Learning rate = 0.002498:   Batch Loss = 0.584574, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.618707597256, Accuracy = 0.928708076477
    Iter #1708032:  Learning rate = 0.002498:   Batch Loss = 0.560328, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.635425329208, Accuracy = 0.922622144222
    Iter #1712128:  Learning rate = 0.002498:   Batch Loss = 0.530334, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.629549503326, Accuracy = 0.930620789528
    Iter #1716224:  Learning rate = 0.002498:   Batch Loss = 0.566467, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.620246112347, Accuracy = 0.932707369328
    Iter #1720320:  Learning rate = 0.002498:   Batch Loss = 0.533980, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.580843806267, Accuracy = 0.951312839985
    Iter #1724416:  Learning rate = 0.002498:   Batch Loss = 0.605842, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.639254212379, Accuracy = 0.923665463924
    Iter #1728512:  Learning rate = 0.002498:   Batch Loss = 0.553334, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.617417693138, Accuracy = 0.934793949127
    Iter #1732608:  Learning rate = 0.002498:   Batch Loss = 0.585124, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.648542463779, Accuracy = 0.921057224274
    Iter #1736704:  Learning rate = 0.002498:   Batch Loss = 0.554231, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.589092850685, Accuracy = 0.940705955029
    Iter #1740800:  Learning rate = 0.002498:   Batch Loss = 0.602317, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.594298779964, Accuracy = 0.938271582127
    Iter #1744896:  Learning rate = 0.002498:   Batch Loss = 0.577908, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.634590566158, Accuracy = 0.924013197422
    Iter #1748992:  Learning rate = 0.002498:   Batch Loss = 0.577212, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.601316571236, Accuracy = 0.93844550848
    Iter #1753088:  Learning rate = 0.002498:   Batch Loss = 0.521597, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.593013346195, Accuracy = 0.93914103508
    Iter #1757184:  Learning rate = 0.002498:   Batch Loss = 0.542114, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.622863352299, Accuracy = 0.933750629425
    Iter #1761280:  Learning rate = 0.002498:   Batch Loss = 0.543652, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.574934363365, Accuracy = 0.946965754032
    Iter #1765376:  Learning rate = 0.002498:   Batch Loss = 0.610012, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.639136373997, Accuracy = 0.928708076477
    Iter #1769472:  Learning rate = 0.002498:   Batch Loss = 0.569824, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.628935813904, Accuracy = 0.929055809975
    Iter #1773568:  Learning rate = 0.002498:   Batch Loss = 0.540990, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.599329590797, Accuracy = 0.942270934582
    Iter #1777664:  Learning rate = 0.002498:   Batch Loss = 0.556269, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.603113353252, Accuracy = 0.936706662178
    Iter #1781760:  Learning rate = 0.002498:   Batch Loss = 0.550289, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.606347203255, Accuracy = 0.934272289276
    Iter #1785856:  Learning rate = 0.002498:   Batch Loss = 0.564715, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.63649815321, Accuracy = 0.918448984623
    Iter #1789952:  Learning rate = 0.002498:   Batch Loss = 0.556313, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.654266119003, Accuracy = 0.91827505827
    Iter #1794048:  Learning rate = 0.002498:   Batch Loss = 0.554269, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.619714677334, Accuracy = 0.934272289276
    Iter #1798144:  Learning rate = 0.002498:   Batch Loss = 0.630809, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.602421224117, Accuracy = 0.939314901829
    Iter #1802240:  Learning rate = 0.002398:   Batch Loss = 0.544208, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.609573304653, Accuracy = 0.930968523026
    Iter #1806336:  Learning rate = 0.002398:   Batch Loss = 0.558880, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.601716995239, Accuracy = 0.937054395676
    Iter #1810432:  Learning rate = 0.002398:   Batch Loss = 0.585465, Accuracy = 0.939453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.599101662636, Accuracy = 0.938097715378
    Iter #1814528:  Learning rate = 0.002398:   Batch Loss = 0.561940, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.625413537025, Accuracy = 0.925056517124
    Iter #1818624:  Learning rate = 0.002398:   Batch Loss = 0.503518, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.585852265358, Accuracy = 0.939314901829
    Iter #1822720:  Learning rate = 0.002398:   Batch Loss = 0.540317, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.593084812164, Accuracy = 0.94053208828
    Iter #1826816:  Learning rate = 0.002398:   Batch Loss = 0.572120, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.641852676868, Accuracy = 0.928186416626
    Iter #1830912:  Learning rate = 0.002398:   Batch Loss = 0.574067, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.63907366991, Accuracy = 0.921926617622
    Iter #1835008:  Learning rate = 0.002398:   Batch Loss = 0.523231, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.59480881691, Accuracy = 0.939314901829
    Iter #1839104:  Learning rate = 0.002398:   Batch Loss = 0.547654, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.631554961205, Accuracy = 0.921926617622
    Iter #1843200:  Learning rate = 0.002398:   Batch Loss = 0.589833, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.609549105167, Accuracy = 0.928881943226
    Iter #1847296:  Learning rate = 0.002398:   Batch Loss = 0.550438, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.621236920357, Accuracy = 0.930794656277
    Iter #1851392:  Learning rate = 0.002398:   Batch Loss = 0.563448, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.632429599762, Accuracy = 0.926795363426
    Iter #1855488:  Learning rate = 0.002398:   Batch Loss = 0.665148, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.605780124664, Accuracy = 0.927490890026
    Iter #1859584:  Learning rate = 0.002398:   Batch Loss = 0.594116, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.589929699898, Accuracy = 0.938271582127
    Iter #1863680:  Learning rate = 0.002398:   Batch Loss = 0.563431, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.620802819729, Accuracy = 0.94122761488
    Iter #1867776:  Learning rate = 0.002398:   Batch Loss = 0.559332, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.61670088768, Accuracy = 0.927838623524
    Iter #1871872:  Learning rate = 0.002398:   Batch Loss = 0.584032, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.610561907291, Accuracy = 0.928012490273
    Iter #1875968:  Learning rate = 0.002398:   Batch Loss = 0.559367, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.603057146072, Accuracy = 0.927664756775
    Iter #1880064:  Learning rate = 0.002398:   Batch Loss = 0.569383, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.609865665436, Accuracy = 0.928360283375
    Iter #1884160:  Learning rate = 0.002398:   Batch Loss = 0.579393, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.682248175144, Accuracy = 0.903321146965
    Iter #1888256:  Learning rate = 0.002398:   Batch Loss = 0.610111, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.654805660248, Accuracy = 0.911667525768
    Iter #1892352:  Learning rate = 0.002398:   Batch Loss = 0.565088, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.582720398903, Accuracy = 0.942618668079
    Iter #1896448:  Learning rate = 0.002398:   Batch Loss = 0.569994, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.614137649536, Accuracy = 0.929577469826
    Iter #1900544:  Learning rate = 0.002302:   Batch Loss = 0.616385, Accuracy = 0.9296875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.642225801945, Accuracy = 0.913058578968
    Iter #1904640:  Learning rate = 0.002302:   Batch Loss = 0.560796, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.667030155659, Accuracy = 0.908885419369
    Iter #1908736:  Learning rate = 0.002302:   Batch Loss = 0.614530, Accuracy = 0.923828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.64194136858, Accuracy = 0.923665463924
    Iter #1912832:  Learning rate = 0.002302:   Batch Loss = 0.530101, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.586121201515, Accuracy = 0.935837268829
    Iter #1916928:  Learning rate = 0.002302:   Batch Loss = 0.589809, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.585752189159, Accuracy = 0.938271582127
    Iter #1921024:  Learning rate = 0.002302:   Batch Loss = 0.548962, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.578482151031, Accuracy = 0.940705955029
    Iter #1925120:  Learning rate = 0.002302:   Batch Loss = 0.536413, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.579881846905, Accuracy = 0.940358221531
    Iter #1929216:  Learning rate = 0.002302:   Batch Loss = 0.542393, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.643867790699, Accuracy = 0.924882650375
    Iter #1933312:  Learning rate = 0.002302:   Batch Loss = 0.548659, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.57853782177, Accuracy = 0.940010428429
    Iter #1937408:  Learning rate = 0.002302:   Batch Loss = 0.537242, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.574586033821, Accuracy = 0.942444801331
    Iter #1941504:  Learning rate = 0.002302:   Batch Loss = 0.546451, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6016664505, Accuracy = 0.933750629425
    Iter #1945600:  Learning rate = 0.002302:   Batch Loss = 0.526340, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.668241262436, Accuracy = 0.911667525768
    Iter #1949696:  Learning rate = 0.002302:   Batch Loss = 0.525127, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.67276597023, Accuracy = 0.906972706318
    Iter #1953792:  Learning rate = 0.002302:   Batch Loss = 0.601666, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.769138336182, Accuracy = 0.905929386616
    Iter #1957888:  Learning rate = 0.002302:   Batch Loss = 0.509026, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.591949522495, Accuracy = 0.937923848629
    Iter #1961984:  Learning rate = 0.002302:   Batch Loss = 0.580893, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.597637355328, Accuracy = 0.934446156025
    Iter #1966080:  Learning rate = 0.002302:   Batch Loss = 0.528978, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.63878595829, Accuracy = 0.926099836826
    Iter #1970176:  Learning rate = 0.002302:   Batch Loss = 0.544307, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.652195692062, Accuracy = 0.925056517124
    Iter #1974272:  Learning rate = 0.002302:   Batch Loss = 0.524370, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.58304476738, Accuracy = 0.939662694931
    Iter #1978368:  Learning rate = 0.002302:   Batch Loss = 0.548128, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.600969731808, Accuracy = 0.928186416626
    Iter #1982464:  Learning rate = 0.002302:   Batch Loss = 0.522122, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.576039969921, Accuracy = 0.939662694931
    Iter #1986560:  Learning rate = 0.002302:   Batch Loss = 0.614502, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.594391703606, Accuracy = 0.935663342476
    Iter #1990656:  Learning rate = 0.002302:   Batch Loss = 0.538538, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.574375391006, Accuracy = 0.940879821777
    Iter #1994752:  Learning rate = 0.002302:   Batch Loss = 0.560375, Accuracy = 0.9453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.626274347305, Accuracy = 0.927664756775
    Iter #1998848:  Learning rate = 0.002302:   Batch Loss = 0.557021, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.574815273285, Accuracy = 0.941401481628
    Iter #2002944:  Learning rate = 0.002210:   Batch Loss = 0.500345, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.579725623131, Accuracy = 0.935489475727
    Iter #2007040:  Learning rate = 0.002210:   Batch Loss = 0.512551, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.553451478481, Accuracy = 0.943661987782
    Iter #2011136:  Learning rate = 0.002210:   Batch Loss = 0.502946, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.590188145638, Accuracy = 0.935663342476
    Iter #2015232:  Learning rate = 0.002210:   Batch Loss = 0.525159, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.585133016109, Accuracy = 0.935663342476
    Iter #2019328:  Learning rate = 0.002210:   Batch Loss = 0.587025, Accuracy = 0.9453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.562590479851, Accuracy = 0.943314194679
    Iter #2023424:  Learning rate = 0.002210:   Batch Loss = 0.521685, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.563082098961, Accuracy = 0.93983656168
    Iter #2027520:  Learning rate = 0.002210:   Batch Loss = 0.539724, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.586188137531, Accuracy = 0.941053748131
    Iter #2031616:  Learning rate = 0.002210:   Batch Loss = 0.555823, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.570666432381, Accuracy = 0.94453138113
    Iter #2035712:  Learning rate = 0.002210:   Batch Loss = 0.545889, Accuracy = 0.943359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.657573640347, Accuracy = 0.913928031921
    Iter #2039808:  Learning rate = 0.002210:   Batch Loss = 0.589632, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.593658387661, Accuracy = 0.934967815876
    Iter #2043904:  Learning rate = 0.002210:   Batch Loss = 0.518239, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.581072449684, Accuracy = 0.938967108727
    Iter #2048000:  Learning rate = 0.002210:   Batch Loss = 0.529848, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.61004191637, Accuracy = 0.928360283375
    Iter #2052096:  Learning rate = 0.002210:   Batch Loss = 0.591861, Accuracy = 0.931640625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.597807943821, Accuracy = 0.930968523026
    Iter #2056192:  Learning rate = 0.002210:   Batch Loss = 0.545682, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.571427702904, Accuracy = 0.932011842728
    Iter #2060288:  Learning rate = 0.002210:   Batch Loss = 0.499516, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.561602413654, Accuracy = 0.94383585453
    Iter #2064384:  Learning rate = 0.002210:   Batch Loss = 0.537516, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.585879564285, Accuracy = 0.930794656277
    Iter #2068480:  Learning rate = 0.002210:   Batch Loss = 0.537486, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.580877363682, Accuracy = 0.942270934582
    Iter #2072576:  Learning rate = 0.002210:   Batch Loss = 0.516670, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.594107508659, Accuracy = 0.934272289276
    Iter #2076672:  Learning rate = 0.002210:   Batch Loss = 0.515174, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.559347510338, Accuracy = 0.945400774479
    Iter #2080768:  Learning rate = 0.002210:   Batch Loss = 0.504331, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.582437455654, Accuracy = 0.932359576225
    Iter #2084864:  Learning rate = 0.002210:   Batch Loss = 0.490880, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.570172429085, Accuracy = 0.941575407982
    Iter #2088960:  Learning rate = 0.002210:   Batch Loss = 0.595497, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.548255860806, Accuracy = 0.947835147381
    Iter #2093056:  Learning rate = 0.002210:   Batch Loss = 0.485689, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.558807730675, Accuracy = 0.942097008228
    Iter #2097152:  Learning rate = 0.002210:   Batch Loss = 0.487676, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.554953455925, Accuracy = 0.94453138113
    Iter #2101248:  Learning rate = 0.002122:   Batch Loss = 0.554006, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.585315644741, Accuracy = 0.936011135578
    Iter #2105344:  Learning rate = 0.002122:   Batch Loss = 0.569532, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.570123314857, Accuracy = 0.942097008228
    Iter #2109440:  Learning rate = 0.002122:   Batch Loss = 0.563526, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.685356318951, Accuracy = 0.896539747715
    Iter #2113536:  Learning rate = 0.002122:   Batch Loss = 0.553559, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6671667099, Accuracy = 0.905929386616
    Iter #2117632:  Learning rate = 0.002122:   Batch Loss = 0.585466, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.663801431656, Accuracy = 0.902277886868
    Iter #2121728:  Learning rate = 0.002122:   Batch Loss = 0.566875, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.649036765099, Accuracy = 0.905755519867
    Iter #2125824:  Learning rate = 0.002122:   Batch Loss = 0.552403, Accuracy = 0.9453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.586258530617, Accuracy = 0.936706662178
    Iter #2129920:  Learning rate = 0.002122:   Batch Loss = 0.545174, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.641684412956, Accuracy = 0.91827505827
    Iter #2134016:  Learning rate = 0.002122:   Batch Loss = 0.526993, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.57391166687, Accuracy = 0.938619375229
    Iter #2138112:  Learning rate = 0.002122:   Batch Loss = 0.516626, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.568187892437, Accuracy = 0.942097008228
    Iter #2142208:  Learning rate = 0.002122:   Batch Loss = 0.522306, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.583376169205, Accuracy = 0.936358869076
    Iter #2146304:  Learning rate = 0.002122:   Batch Loss = 0.505769, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.565180897713, Accuracy = 0.94453138113
    Iter #2150400:  Learning rate = 0.002122:   Batch Loss = 0.498606, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.576694250107, Accuracy = 0.93914103508
    Iter #2154496:  Learning rate = 0.002122:   Batch Loss = 0.521796, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.562592208385, Accuracy = 0.940879821777
    Iter #2158592:  Learning rate = 0.002122:   Batch Loss = 0.501406, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.557237148285, Accuracy = 0.94731348753
    Iter #2162688:  Learning rate = 0.002122:   Batch Loss = 0.504227, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.572582364082, Accuracy = 0.941923141479
    Iter #2166784:  Learning rate = 0.002122:   Batch Loss = 0.540284, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.561099410057, Accuracy = 0.943488061428
    Iter #2170880:  Learning rate = 0.002122:   Batch Loss = 0.538146, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.563695311546, Accuracy = 0.945574700832
    Iter #2174976:  Learning rate = 0.002122:   Batch Loss = 0.507844, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.567268073559, Accuracy = 0.93983656168
    Iter #2179072:  Learning rate = 0.002122:   Batch Loss = 0.531524, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.541304051876, Accuracy = 0.948878467083
    Iter #2183168:  Learning rate = 0.002122:   Batch Loss = 0.477827, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.560562551022, Accuracy = 0.936880528927
    Iter #2187264:  Learning rate = 0.002122:   Batch Loss = 0.505966, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.563704252243, Accuracy = 0.938271582127
    Iter #2191360:  Learning rate = 0.002122:   Batch Loss = 0.524675, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.540536344051, Accuracy = 0.942966461182
    Iter #2195456:  Learning rate = 0.002122:   Batch Loss = 0.514280, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.589084565639, Accuracy = 0.937054395676
    Iter #2199552:  Learning rate = 0.002122:   Batch Loss = 0.558962, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.580360352993, Accuracy = 0.933750629425
    Iter #2203648:  Learning rate = 0.002037:   Batch Loss = 0.539153, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.573957920074, Accuracy = 0.931316316128
    Iter #2207744:  Learning rate = 0.002037:   Batch Loss = 0.512345, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.557651162148, Accuracy = 0.94383585453
    Iter #2211840:  Learning rate = 0.002037:   Batch Loss = 0.516524, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.560096025467, Accuracy = 0.939662694931
    Iter #2215936:  Learning rate = 0.002037:   Batch Loss = 0.494531, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.540883779526, Accuracy = 0.945053040981
    Iter #2220032:  Learning rate = 0.002037:   Batch Loss = 0.514008, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.534468233585, Accuracy = 0.94731348753
    Iter #2224128:  Learning rate = 0.002037:   Batch Loss = 0.499434, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.574003338814, Accuracy = 0.932881236076
    Iter #2228224:  Learning rate = 0.002037:   Batch Loss = 0.484709, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.547576963902, Accuracy = 0.949052333832
    Iter #2232320:  Learning rate = 0.002037:   Batch Loss = 0.483566, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.541997909546, Accuracy = 0.946096360683
    Iter #2236416:  Learning rate = 0.002037:   Batch Loss = 0.503918, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.536097824574, Accuracy = 0.945748567581
    Iter #2240512:  Learning rate = 0.002037:   Batch Loss = 0.470373, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.564124107361, Accuracy = 0.941053748131
    Iter #2244608:  Learning rate = 0.002037:   Batch Loss = 0.558792, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.556806385517, Accuracy = 0.940184295177
    Iter #2248704:  Learning rate = 0.002037:   Batch Loss = 0.542916, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.568210661411, Accuracy = 0.932533442974
    Iter #2252800:  Learning rate = 0.002037:   Batch Loss = 0.504780, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.567918479443, Accuracy = 0.934446156025
    Iter #2256896:  Learning rate = 0.002037:   Batch Loss = 0.463772, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.557286381721, Accuracy = 0.941401481628
    Iter #2260992:  Learning rate = 0.002037:   Batch Loss = 0.529583, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.564914524555, Accuracy = 0.938793241978
    Iter #2265088:  Learning rate = 0.002037:   Batch Loss = 0.492740, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.544739305973, Accuracy = 0.943661987782
    Iter #2269184:  Learning rate = 0.002037:   Batch Loss = 0.479516, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.556976556778, Accuracy = 0.940358221531
    Iter #2273280:  Learning rate = 0.002037:   Batch Loss = 0.523295, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.553324222565, Accuracy = 0.939488768578
    Iter #2277376:  Learning rate = 0.002037:   Batch Loss = 0.541209, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.585600197315, Accuracy = 0.934272289276
    Iter #2281472:  Learning rate = 0.002037:   Batch Loss = 0.524233, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.619158744812, Accuracy = 0.921057224274
    Iter #2285568:  Learning rate = 0.002037:   Batch Loss = 0.475449, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.545523762703, Accuracy = 0.942270934582
    Iter #2289664:  Learning rate = 0.002037:   Batch Loss = 0.525737, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.545948982239, Accuracy = 0.943661987782
    Iter #2293760:  Learning rate = 0.002037:   Batch Loss = 0.503999, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.551904499531, Accuracy = 0.94053208828
    Iter #2297856:  Learning rate = 0.002037:   Batch Loss = 0.515294, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.534569680691, Accuracy = 0.942270934582
    Iter #2301952:  Learning rate = 0.001955:   Batch Loss = 0.506163, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.542423784733, Accuracy = 0.944357514381
    Iter #2306048:  Learning rate = 0.001955:   Batch Loss = 0.452664, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.564482152462, Accuracy = 0.942270934582
    Iter #2310144:  Learning rate = 0.001955:   Batch Loss = 0.520066, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.578108727932, Accuracy = 0.934446156025
    Iter #2314240:  Learning rate = 0.001955:   Batch Loss = 0.534366, Accuracy = 0.94140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.600565969944, Accuracy = 0.937402188778
    Iter #2318336:  Learning rate = 0.001955:   Batch Loss = 0.504136, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.537182211876, Accuracy = 0.953921079636
    Iter #2322432:  Learning rate = 0.001955:   Batch Loss = 0.465527, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.567206382751, Accuracy = 0.938619375229
    Iter #2326528:  Learning rate = 0.001955:   Batch Loss = 0.517820, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.536011695862, Accuracy = 0.945053040981
    Iter #2330624:  Learning rate = 0.001955:   Batch Loss = 0.493787, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.529593706131, Accuracy = 0.950269520283
    Iter #2334720:  Learning rate = 0.001955:   Batch Loss = 0.508261, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.529145598412, Accuracy = 0.949400126934
    Iter #2338816:  Learning rate = 0.001955:   Batch Loss = 0.501999, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.538738846779, Accuracy = 0.947661280632
    Iter #2342912:  Learning rate = 0.001955:   Batch Loss = 0.522065, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.725014686584, Accuracy = 0.892540454865
    Iter #2347008:  Learning rate = 0.001955:   Batch Loss = 0.630402, Accuracy = 0.927734375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.665557980537, Accuracy = 0.906798839569
    Iter #2351104:  Learning rate = 0.001955:   Batch Loss = 0.529342, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.575243830681, Accuracy = 0.935315608978
    Iter #2355200:  Learning rate = 0.001955:   Batch Loss = 0.520256, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.574419558048, Accuracy = 0.933750629425
    Iter #2359296:  Learning rate = 0.001955:   Batch Loss = 0.501677, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.565091371536, Accuracy = 0.940010428429
    Iter #2363392:  Learning rate = 0.001955:   Batch Loss = 0.499235, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5352678895, Accuracy = 0.940358221531
    Iter #2367488:  Learning rate = 0.001955:   Batch Loss = 0.486072, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.524138152599, Accuracy = 0.947139620781
    Iter #2371584:  Learning rate = 0.001955:   Batch Loss = 0.485485, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.528716564178, Accuracy = 0.94800901413
    Iter #2375680:  Learning rate = 0.001955:   Batch Loss = 0.534523, Accuracy = 0.9453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.547090947628, Accuracy = 0.936185002327
    Iter #2379776:  Learning rate = 0.001955:   Batch Loss = 0.475882, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.579249620438, Accuracy = 0.928534150124
    Iter #2383872:  Learning rate = 0.001955:   Batch Loss = 0.488666, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.571677803993, Accuracy = 0.930446863174
    Iter #2387968:  Learning rate = 0.001955:   Batch Loss = 0.548667, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.567755818367, Accuracy = 0.934446156025
    Iter #2392064:  Learning rate = 0.001955:   Batch Loss = 0.525165, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.53614372015, Accuracy = 0.947661280632
    Iter #2396160:  Learning rate = 0.001955:   Batch Loss = 0.472452, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.544683456421, Accuracy = 0.947139620781
    Iter #2400256:  Learning rate = 0.001877:   Batch Loss = 0.515062, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.538223803043, Accuracy = 0.948356807232
    Iter #2404352:  Learning rate = 0.001877:   Batch Loss = 0.549833, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.553956985474, Accuracy = 0.941749274731
    Iter #2408448:  Learning rate = 0.001877:   Batch Loss = 0.474232, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.540567517281, Accuracy = 0.947487413883
    Iter #2412544:  Learning rate = 0.001877:   Batch Loss = 0.512531, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.52584028244, Accuracy = 0.951138913631
    Iter #2416640:  Learning rate = 0.001877:   Batch Loss = 0.511138, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.520682156086, Accuracy = 0.952703893185
    Iter #2420736:  Learning rate = 0.001877:   Batch Loss = 0.480288, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.541589558125, Accuracy = 0.949400126934
    Iter #2424832:  Learning rate = 0.001877:   Batch Loss = 0.491814, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.564180850983, Accuracy = 0.927490890026
    Iter #2428928:  Learning rate = 0.001877:   Batch Loss = 0.545809, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.599998116493, Accuracy = 0.919840037823
    Iter #2433024:  Learning rate = 0.001877:   Batch Loss = 0.526372, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.589217841625, Accuracy = 0.926447570324
    Iter #2437120:  Learning rate = 0.001877:   Batch Loss = 0.503507, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.514443099499, Accuracy = 0.954790472984
    Iter #2441216:  Learning rate = 0.001877:   Batch Loss = 0.468752, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.524876892567, Accuracy = 0.94731348753
    Iter #2445312:  Learning rate = 0.001877:   Batch Loss = 0.500652, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.551196932793, Accuracy = 0.942966461182
    Iter #2449408:  Learning rate = 0.001877:   Batch Loss = 0.457047, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5289093256, Accuracy = 0.948530673981
    Iter #2453504:  Learning rate = 0.001877:   Batch Loss = 0.481784, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.57051217556, Accuracy = 0.939662694931
    Iter #2457600:  Learning rate = 0.001877:   Batch Loss = 0.533041, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.560644388199, Accuracy = 0.936532795429
    Iter #2461696:  Learning rate = 0.001877:   Batch Loss = 0.509439, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.553218126297, Accuracy = 0.938619375229
    Iter #2465792:  Learning rate = 0.001877:   Batch Loss = 0.509301, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.551937460899, Accuracy = 0.93774998188
    Iter #2469888:  Learning rate = 0.001877:   Batch Loss = 0.476438, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.532647490501, Accuracy = 0.943488061428
    Iter #2473984:  Learning rate = 0.001877:   Batch Loss = 0.468486, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.529402017593, Accuracy = 0.94453138113
    Iter #2478080:  Learning rate = 0.001877:   Batch Loss = 0.493913, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.525795698166, Accuracy = 0.944879174232
    Iter #2482176:  Learning rate = 0.001877:   Batch Loss = 0.493050, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.537017643452, Accuracy = 0.939662694931
    Iter #2486272:  Learning rate = 0.001877:   Batch Loss = 0.468282, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.547195792198, Accuracy = 0.934793949127
    Iter #2490368:  Learning rate = 0.001877:   Batch Loss = 0.527184, Accuracy = 0.93359375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.515070259571, Accuracy = 0.947661280632
    Iter #2494464:  Learning rate = 0.001877:   Batch Loss = 0.487196, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.544075727463, Accuracy = 0.94314032793
    Iter #2498560:  Learning rate = 0.001877:   Batch Loss = 0.463072, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.52820700407, Accuracy = 0.949052333832
    Iter #2502656:  Learning rate = 0.001802:   Batch Loss = 0.457165, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.515081882477, Accuracy = 0.949747860432
    Iter #2506752:  Learning rate = 0.001802:   Batch Loss = 0.485425, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.51540941, Accuracy = 0.946965754032
    Iter #2510848:  Learning rate = 0.001802:   Batch Loss = 0.482958, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.531985104084, Accuracy = 0.942444801331
    Iter #2514944:  Learning rate = 0.001802:   Batch Loss = 0.446711, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.519189417362, Accuracy = 0.948704600334
    Iter #2519040:  Learning rate = 0.001802:   Batch Loss = 0.470555, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.530745625496, Accuracy = 0.943661987782
    Iter #2523136:  Learning rate = 0.001802:   Batch Loss = 0.475472, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.523129940033, Accuracy = 0.94661796093
    Iter #2527232:  Learning rate = 0.001802:   Batch Loss = 0.480935, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.507289350033, Accuracy = 0.950617313385
    Iter #2531328:  Learning rate = 0.001802:   Batch Loss = 0.443666, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.507266879082, Accuracy = 0.953573286533
    Iter #2535424:  Learning rate = 0.001802:   Batch Loss = 0.461262, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.572423577309, Accuracy = 0.926795363426
    Iter #2539520:  Learning rate = 0.001802:   Batch Loss = 0.471635, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49767100811, Accuracy = 0.951486706734
    Iter #2543616:  Learning rate = 0.001802:   Batch Loss = 0.460681, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.514082014561, Accuracy = 0.947661280632
    Iter #2547712:  Learning rate = 0.001802:   Batch Loss = 0.495591, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.562006890774, Accuracy = 0.943661987782
    Iter #2551808:  Learning rate = 0.001802:   Batch Loss = 0.518148, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.539506316185, Accuracy = 0.941923141479
    Iter #2555904:  Learning rate = 0.001802:   Batch Loss = 0.477097, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.514986336231, Accuracy = 0.948182940483
    Iter #2560000:  Learning rate = 0.001802:   Batch Loss = 0.450361, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.530645549297, Accuracy = 0.944357514381
    Iter #2564096:  Learning rate = 0.001802:   Batch Loss = 0.478102, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.496508538723, Accuracy = 0.958789765835
    Iter #2568192:  Learning rate = 0.001802:   Batch Loss = 0.461406, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.511248648167, Accuracy = 0.951312839985
    Iter #2572288:  Learning rate = 0.001802:   Batch Loss = 0.453228, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.528858661652, Accuracy = 0.947487413883
    Iter #2576384:  Learning rate = 0.001802:   Batch Loss = 0.472887, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.508374214172, Accuracy = 0.950095653534
    Iter #2580480:  Learning rate = 0.001802:   Batch Loss = 0.471550, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.510186493397, Accuracy = 0.949747860432
    Iter #2584576:  Learning rate = 0.001802:   Batch Loss = 0.492911, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.514749109745, Accuracy = 0.94800901413
    Iter #2588672:  Learning rate = 0.001802:   Batch Loss = 0.441560, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.50522339344, Accuracy = 0.954094946384
    Iter #2592768:  Learning rate = 0.001802:   Batch Loss = 0.453383, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.494298189878, Accuracy = 0.957224845886
    Iter #2596864:  Learning rate = 0.001802:   Batch Loss = 0.525743, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.513303995132, Accuracy = 0.952703893185
    Iter #2600960:  Learning rate = 0.001730:   Batch Loss = 0.463830, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.507171034813, Accuracy = 0.958442032337
    Iter #2605056:  Learning rate = 0.001730:   Batch Loss = 0.464145, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.500338673592, Accuracy = 0.957050919533
    Iter #2609152:  Learning rate = 0.001730:   Batch Loss = 0.461201, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4953327775, Accuracy = 0.958789765835
    Iter #2613248:  Learning rate = 0.001730:   Batch Loss = 0.480223, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.557500660419, Accuracy = 0.944705247879
    Iter #2617344:  Learning rate = 0.001730:   Batch Loss = 0.480321, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.517543256283, Accuracy = 0.949226200581
    Iter #2621440:  Learning rate = 0.001730:   Batch Loss = 0.485507, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.501497626305, Accuracy = 0.955485999584
    Iter #2625536:  Learning rate = 0.001730:   Batch Loss = 0.479248, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.523850500584, Accuracy = 0.946965754032
    Iter #2629632:  Learning rate = 0.001730:   Batch Loss = 0.469164, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.515877962112, Accuracy = 0.945400774479
    Iter #2633728:  Learning rate = 0.001730:   Batch Loss = 0.444301, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.495955526829, Accuracy = 0.954268813133
    Iter #2637824:  Learning rate = 0.001730:   Batch Loss = 0.456435, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.485600233078, Accuracy = 0.960180819035
    Iter #2641920:  Learning rate = 0.001730:   Batch Loss = 0.449749, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.491400450468, Accuracy = 0.957746505737
    Iter #2646016:  Learning rate = 0.001730:   Batch Loss = 0.482257, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.499469041824, Accuracy = 0.952008366585
    Iter #2650112:  Learning rate = 0.001730:   Batch Loss = 0.481181, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.528387784958, Accuracy = 0.940184295177
    Iter #2654208:  Learning rate = 0.001730:   Batch Loss = 0.491371, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6969653368, Accuracy = 0.897930800915
    Iter #2658304:  Learning rate = 0.001730:   Batch Loss = 0.574480, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.630158185959, Accuracy = 0.902451753616
    Iter #2662400:  Learning rate = 0.001730:   Batch Loss = 0.494103, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.579413354397, Accuracy = 0.929925203323
    Iter #2666496:  Learning rate = 0.001730:   Batch Loss = 0.494964, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.527734935284, Accuracy = 0.947661280632
    Iter #2670592:  Learning rate = 0.001730:   Batch Loss = 0.498318, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.509145379066, Accuracy = 0.950443387032
    Iter #2674688:  Learning rate = 0.001730:   Batch Loss = 0.476055, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.543831467628, Accuracy = 0.93774998188
    Iter #2678784:  Learning rate = 0.001730:   Batch Loss = 0.467060, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.537402272224, Accuracy = 0.946965754032
    Iter #2682880:  Learning rate = 0.001730:   Batch Loss = 0.490029, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.575011610985, Accuracy = 0.927838623524
    Iter #2686976:  Learning rate = 0.001730:   Batch Loss = 0.482290, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.519793391228, Accuracy = 0.94314032793
    Iter #2691072:  Learning rate = 0.001730:   Batch Loss = 0.477670, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.499949455261, Accuracy = 0.953573286533
    Iter #2695168:  Learning rate = 0.001730:   Batch Loss = 0.460723, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.523096442223, Accuracy = 0.946096360683
    Iter #2699264:  Learning rate = 0.001730:   Batch Loss = 0.517410, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.54250562191, Accuracy = 0.937576055527
    Iter #2703360:  Learning rate = 0.001661:   Batch Loss = 0.477212, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.531854391098, Accuracy = 0.94053208828
    Iter #2707456:  Learning rate = 0.001661:   Batch Loss = 0.500325, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.505989551544, Accuracy = 0.949052333832
    Iter #2711552:  Learning rate = 0.001661:   Batch Loss = 0.447318, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.516589045525, Accuracy = 0.941053748131
    Iter #2715648:  Learning rate = 0.001661:   Batch Loss = 0.487967, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.515110611916, Accuracy = 0.949052333832
    Iter #2719744:  Learning rate = 0.001661:   Batch Loss = 0.462665, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.520423412323, Accuracy = 0.94522690773
    Iter #2723840:  Learning rate = 0.001661:   Batch Loss = 0.455833, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.513803958893, Accuracy = 0.948356807232
    Iter #2727936:  Learning rate = 0.001661:   Batch Loss = 0.460035, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.508012771606, Accuracy = 0.950095653534
    Iter #2732032:  Learning rate = 0.001661:   Batch Loss = 0.495248, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.505453407764, Accuracy = 0.948356807232
    Iter #2736128:  Learning rate = 0.001661:   Batch Loss = 0.463555, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.526397705078, Accuracy = 0.944705247879
    Iter #2740224:  Learning rate = 0.001661:   Batch Loss = 0.468588, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.50989151001, Accuracy = 0.953399419785
    Iter #2744320:  Learning rate = 0.001661:   Batch Loss = 0.444828, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.511084794998, Accuracy = 0.949573993683
    Iter #2748416:  Learning rate = 0.001661:   Batch Loss = 0.447212, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.500794708729, Accuracy = 0.952008366585
    Iter #2752512:  Learning rate = 0.001661:   Batch Loss = 0.446230, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.490704029799, Accuracy = 0.952703893185
    Iter #2756608:  Learning rate = 0.001661:   Batch Loss = 0.468298, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.516977727413, Accuracy = 0.949747860432
    Iter #2760704:  Learning rate = 0.001661:   Batch Loss = 0.508323, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.541007816792, Accuracy = 0.942966461182
    Iter #2764800:  Learning rate = 0.001661:   Batch Loss = 0.481579, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.574685454369, Accuracy = 0.919318377972
    Iter #2768896:  Learning rate = 0.001661:   Batch Loss = 0.471489, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.557268500328, Accuracy = 0.929055809975
    Iter #2772992:  Learning rate = 0.001661:   Batch Loss = 0.466739, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.506975889206, Accuracy = 0.952008366585
    Iter #2777088:  Learning rate = 0.001661:   Batch Loss = 0.454710, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.498102843761, Accuracy = 0.951312839985
    Iter #2781184:  Learning rate = 0.001661:   Batch Loss = 0.495867, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.505194306374, Accuracy = 0.948878467083
    Iter #2785280:  Learning rate = 0.001661:   Batch Loss = 0.434350, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.498650670052, Accuracy = 0.954790472984
    Iter #2789376:  Learning rate = 0.001661:   Batch Loss = 0.466284, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.504788517952, Accuracy = 0.948704600334
    Iter #2793472:  Learning rate = 0.001661:   Batch Loss = 0.428511, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.505931794643, Accuracy = 0.953573286533
    Iter #2797568:  Learning rate = 0.001661:   Batch Loss = 0.493277, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.501238048077, Accuracy = 0.952703893185
    Iter #2801664:  Learning rate = 0.001594:   Batch Loss = 0.533516, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.556006073952, Accuracy = 0.932185709476
    Iter #2805760:  Learning rate = 0.001594:   Batch Loss = 0.457656, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.471756756306, Accuracy = 0.961398005486
    Iter #2809856:  Learning rate = 0.001594:   Batch Loss = 0.449718, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.493979036808, Accuracy = 0.955138266087
    Iter #2813952:  Learning rate = 0.001594:   Batch Loss = 0.419609, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.482079088688, Accuracy = 0.956877052784
    Iter #2818048:  Learning rate = 0.001594:   Batch Loss = 0.442281, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.50616645813, Accuracy = 0.953399419785
    Iter #2822144:  Learning rate = 0.001594:   Batch Loss = 0.437471, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.528924405575, Accuracy = 0.940705955029
    Iter #2826240:  Learning rate = 0.001594:   Batch Loss = 0.451414, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.489310026169, Accuracy = 0.953747153282
    Iter #2830336:  Learning rate = 0.001594:   Batch Loss = 0.460027, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4905397892, Accuracy = 0.951486706734
    Iter #2834432:  Learning rate = 0.001594:   Batch Loss = 0.480928, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.473301380873, Accuracy = 0.958963632584
    Iter #2838528:  Learning rate = 0.001594:   Batch Loss = 0.459908, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.500727891922, Accuracy = 0.948878467083
    Iter #2842624:  Learning rate = 0.001594:   Batch Loss = 0.445302, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.488423466682, Accuracy = 0.952703893185
    Iter #2846720:  Learning rate = 0.001594:   Batch Loss = 0.458719, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.487469255924, Accuracy = 0.956007659435
    Iter #2850816:  Learning rate = 0.001594:   Batch Loss = 0.452304, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.492481768131, Accuracy = 0.955485999584
    Iter #2854912:  Learning rate = 0.001594:   Batch Loss = 0.438619, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.516768813133, Accuracy = 0.941749274731
    Iter #2859008:  Learning rate = 0.001594:   Batch Loss = 0.455123, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.492639243603, Accuracy = 0.954442679882
    Iter #2863104:  Learning rate = 0.001594:   Batch Loss = 0.485803, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.516489863396, Accuracy = 0.944705247879
    Iter #2867200:  Learning rate = 0.001594:   Batch Loss = 0.439790, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.508109807968, Accuracy = 0.947139620781
    Iter #2871296:  Learning rate = 0.001594:   Batch Loss = 0.506083, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.493266522884, Accuracy = 0.951834440231
    Iter #2875392:  Learning rate = 0.001594:   Batch Loss = 0.447521, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.50801897049, Accuracy = 0.954094946384
    Iter #2879488:  Learning rate = 0.001594:   Batch Loss = 0.521284, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.506179451942, Accuracy = 0.946270227432
    Iter #2883584:  Learning rate = 0.001594:   Batch Loss = 0.424421, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.499314308167, Accuracy = 0.944183647633
    Iter #2887680:  Learning rate = 0.001594:   Batch Loss = 0.462308, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.546355068684, Accuracy = 0.942097008228
    Iter #2891776:  Learning rate = 0.001594:   Batch Loss = 0.550788, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.572545289993, Accuracy = 0.933055102825
    Iter #2895872:  Learning rate = 0.001594:   Batch Loss = 0.514820, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.505441784859, Accuracy = 0.943661987782
    Iter #2899968:  Learning rate = 0.001594:   Batch Loss = 0.484875, Accuracy = 0.951171875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.504941046238, Accuracy = 0.945748567581
    Iter #2904064:  Learning rate = 0.001531:   Batch Loss = 0.472148, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.508052766323, Accuracy = 0.946096360683
    Iter #2908160:  Learning rate = 0.001531:   Batch Loss = 0.423385, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.485115349293, Accuracy = 0.953051626682
    Iter #2912256:  Learning rate = 0.001531:   Batch Loss = 0.423595, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.480207920074, Accuracy = 0.953573286533
    Iter #2916352:  Learning rate = 0.001531:   Batch Loss = 0.437081, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.516273260117, Accuracy = 0.94314032793
    Iter #2920448:  Learning rate = 0.001531:   Batch Loss = 0.508516, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.523169457912, Accuracy = 0.942270934582
    Iter #2924544:  Learning rate = 0.001531:   Batch Loss = 0.521420, Accuracy = 0.947265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.499244600534, Accuracy = 0.949052333832
    Iter #2928640:  Learning rate = 0.001531:   Batch Loss = 0.488831, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.482202142477, Accuracy = 0.955138266087
    Iter #2932736:  Learning rate = 0.001531:   Batch Loss = 0.448201, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.475184082985, Accuracy = 0.952182233334
    Iter #2936832:  Learning rate = 0.001531:   Batch Loss = 0.455818, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.497100144625, Accuracy = 0.953225553036
    Iter #2940928:  Learning rate = 0.001531:   Batch Loss = 0.460733, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.517437577248, Accuracy = 0.946791887283
    Iter #2945024:  Learning rate = 0.001531:   Batch Loss = 0.447238, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.522503137589, Accuracy = 0.953399419785
    Iter #2949120:  Learning rate = 0.001531:   Batch Loss = 0.455852, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.517333745956, Accuracy = 0.950269520283
    Iter #2953216:  Learning rate = 0.001531:   Batch Loss = 0.428407, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.483185321093, Accuracy = 0.955833792686
    Iter #2957312:  Learning rate = 0.001531:   Batch Loss = 0.432170, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.479219913483, Accuracy = 0.960006952286
    Iter #2961408:  Learning rate = 0.001531:   Batch Loss = 0.426170, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.472272276878, Accuracy = 0.957746505737
    Iter #2965504:  Learning rate = 0.001531:   Batch Loss = 0.427088, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.534303426743, Accuracy = 0.941053748131
    Iter #2969600:  Learning rate = 0.001531:   Batch Loss = 0.456922, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.482729285955, Accuracy = 0.955312132835
    Iter #2973696:  Learning rate = 0.001531:   Batch Loss = 0.449966, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.474164187908, Accuracy = 0.957398712635
    Iter #2977792:  Learning rate = 0.001531:   Batch Loss = 0.418686, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.544476151466, Accuracy = 0.943488061428
    Iter #2981888:  Learning rate = 0.001531:   Batch Loss = 0.423415, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.488619595766, Accuracy = 0.952703893185
    Iter #2985984:  Learning rate = 0.001531:   Batch Loss = 0.423453, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.477811336517, Accuracy = 0.956877052784
    Iter #2990080:  Learning rate = 0.001531:   Batch Loss = 0.455757, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.511065244675, Accuracy = 0.942966461182
    Iter #2994176:  Learning rate = 0.001531:   Batch Loss = 0.436331, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.509302377701, Accuracy = 0.94122761488
    Iter #2998272:  Learning rate = 0.001531:   Batch Loss = 0.516096, Accuracy = 0.9453125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.497953295708, Accuracy = 0.946965754032
    Iter #3002368:  Learning rate = 0.001469:   Batch Loss = 0.454365, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.507648348808, Accuracy = 0.946965754032
    Iter #3006464:  Learning rate = 0.001469:   Batch Loss = 0.490342, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.517734229565, Accuracy = 0.937054395676
    Iter #3010560:  Learning rate = 0.001469:   Batch Loss = 0.462419, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.560716509819, Accuracy = 0.926969230175
    Iter #3014656:  Learning rate = 0.001469:   Batch Loss = 0.472668, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.521356463432, Accuracy = 0.934793949127
    Iter #3018752:  Learning rate = 0.001469:   Batch Loss = 0.441309, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.500018954277, Accuracy = 0.953399419785
    Iter #3022848:  Learning rate = 0.001469:   Batch Loss = 0.451621, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.52546197176, Accuracy = 0.941575407982
    Iter #3026944:  Learning rate = 0.001469:   Batch Loss = 0.437203, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.496891796589, Accuracy = 0.952703893185
    Iter #3031040:  Learning rate = 0.001469:   Batch Loss = 0.477103, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.520498812199, Accuracy = 0.940184295177
    Iter #3035136:  Learning rate = 0.001469:   Batch Loss = 0.434293, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.508045911789, Accuracy = 0.946791887283
    Iter #3039232:  Learning rate = 0.001469:   Batch Loss = 0.448360, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.489486783743, Accuracy = 0.950791180134
    Iter #3043328:  Learning rate = 0.001469:   Batch Loss = 0.407641, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.498266339302, Accuracy = 0.953921079636
    Iter #3047424:  Learning rate = 0.001469:   Batch Loss = 0.463137, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.53801202774, Accuracy = 0.937228322029
    Iter #3051520:  Learning rate = 0.001469:   Batch Loss = 0.441932, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.511112809181, Accuracy = 0.945053040981
    Iter #3055616:  Learning rate = 0.001469:   Batch Loss = 0.434469, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.509882807732, Accuracy = 0.944879174232
    Iter #3059712:  Learning rate = 0.001469:   Batch Loss = 0.447402, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.496223241091, Accuracy = 0.949052333832
    Iter #3063808:  Learning rate = 0.001469:   Batch Loss = 0.437897, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.498897999525, Accuracy = 0.950095653534
    Iter #3067904:  Learning rate = 0.001469:   Batch Loss = 0.430788, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.492668539286, Accuracy = 0.955485999584
    Iter #3072000:  Learning rate = 0.001469:   Batch Loss = 0.444120, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.505916118622, Accuracy = 0.937228322029
    Iter #3076096:  Learning rate = 0.001469:   Batch Loss = 0.455541, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.512045204639, Accuracy = 0.942966461182
    Iter #3080192:  Learning rate = 0.001469:   Batch Loss = 0.435211, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.478880345821, Accuracy = 0.955659866333
    Iter #3084288:  Learning rate = 0.001469:   Batch Loss = 0.434142, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.482835918665, Accuracy = 0.952703893185
    Iter #3088384:  Learning rate = 0.001469:   Batch Loss = 0.419548, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.488219380379, Accuracy = 0.954094946384
    Iter #3092480:  Learning rate = 0.001469:   Batch Loss = 0.427208, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.472692638636, Accuracy = 0.960354745388
    Iter #3096576:  Learning rate = 0.001469:   Batch Loss = 0.426756, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.467209100723, Accuracy = 0.958615899086
    Iter #3100672:  Learning rate = 0.001411:   Batch Loss = 0.436129, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.468859612942, Accuracy = 0.959659218788
    Iter #3104768:  Learning rate = 0.001411:   Batch Loss = 0.437129, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.525495946407, Accuracy = 0.941053748131
    Iter #3108864:  Learning rate = 0.001411:   Batch Loss = 0.417743, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.478831827641, Accuracy = 0.954268813133
    Iter #3112960:  Learning rate = 0.001411:   Batch Loss = 0.439232, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.471307307482, Accuracy = 0.957050919533
    Iter #3117056:  Learning rate = 0.001411:   Batch Loss = 0.426075, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.465512096882, Accuracy = 0.961224138737
    Iter #3121152:  Learning rate = 0.001411:   Batch Loss = 0.485652, Accuracy = 0.94921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.465232312679, Accuracy = 0.957572579384
    Iter #3125248:  Learning rate = 0.001411:   Batch Loss = 0.401329, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.478838473558, Accuracy = 0.953747153282
    Iter #3129344:  Learning rate = 0.001411:   Batch Loss = 0.424548, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.471224755049, Accuracy = 0.957920372486
    Iter #3133440:  Learning rate = 0.001411:   Batch Loss = 0.411577, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48088824749, Accuracy = 0.955833792686
    Iter #3137536:  Learning rate = 0.001411:   Batch Loss = 0.452065, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.484539926052, Accuracy = 0.952182233334
    Iter #3141632:  Learning rate = 0.001411:   Batch Loss = 0.420075, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.479601740837, Accuracy = 0.957224845886
    Iter #3145728:  Learning rate = 0.001411:   Batch Loss = 0.410344, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.489869773388, Accuracy = 0.948182940483
    Iter #3149824:  Learning rate = 0.001411:   Batch Loss = 0.438796, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.480196237564, Accuracy = 0.94992172718
    Iter #3153920:  Learning rate = 0.001411:   Batch Loss = 0.427431, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.495031297207, Accuracy = 0.94800901413
    Iter #3158016:  Learning rate = 0.001411:   Batch Loss = 0.431033, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.476989448071, Accuracy = 0.956355392933
    Iter #3162112:  Learning rate = 0.001411:   Batch Loss = 0.439025, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.468785583973, Accuracy = 0.954964339733
    Iter #3166208:  Learning rate = 0.001411:   Batch Loss = 0.398163, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.465808212757, Accuracy = 0.956007659435
    Iter #3170304:  Learning rate = 0.001411:   Batch Loss = 0.492407, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.475366234779, Accuracy = 0.954790472984
    Iter #3174400:  Learning rate = 0.001411:   Batch Loss = 0.429932, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.480907618999, Accuracy = 0.950965046883
    Iter #3178496:  Learning rate = 0.001411:   Batch Loss = 0.407252, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.484319329262, Accuracy = 0.954442679882
    Iter #3182592:  Learning rate = 0.001411:   Batch Loss = 0.411754, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.473362445831, Accuracy = 0.956877052784
    Iter #3186688:  Learning rate = 0.001411:   Batch Loss = 0.414686, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.468561112881, Accuracy = 0.958268105984
    Iter #3190784:  Learning rate = 0.001411:   Batch Loss = 0.390042, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.469584107399, Accuracy = 0.955833792686
    Iter #3194880:  Learning rate = 0.001411:   Batch Loss = 0.447723, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.461908310652, Accuracy = 0.961571872234
    Iter #3198976:  Learning rate = 0.001411:   Batch Loss = 0.402013, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.456732273102, Accuracy = 0.960702478886
    Iter #3203072:  Learning rate = 0.001354:   Batch Loss = 0.408050, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.465588390827, Accuracy = 0.959485292435
    Iter #3207168:  Learning rate = 0.001354:   Batch Loss = 0.431710, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.492677360773, Accuracy = 0.954268813133
    Iter #3211264:  Learning rate = 0.001354:   Batch Loss = 0.427776, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.478709936142, Accuracy = 0.950269520283
    Iter #3215360:  Learning rate = 0.001354:   Batch Loss = 0.399400, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.478114843369, Accuracy = 0.954268813133
    Iter #3219456:  Learning rate = 0.001354:   Batch Loss = 0.418675, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.477426290512, Accuracy = 0.956355392933
    Iter #3223552:  Learning rate = 0.001354:   Batch Loss = 0.410471, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.477913290262, Accuracy = 0.953225553036
    Iter #3227648:  Learning rate = 0.001354:   Batch Loss = 0.400962, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.463032156229, Accuracy = 0.959833085537
    Iter #3231744:  Learning rate = 0.001354:   Batch Loss = 0.421690, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.463402301073, Accuracy = 0.958963632584
    Iter #3235840:  Learning rate = 0.001354:   Batch Loss = 0.484960, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.467325031757, Accuracy = 0.958268105984
    Iter #3239936:  Learning rate = 0.001354:   Batch Loss = 0.416907, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.451161175966, Accuracy = 0.962267458439
    Iter #3244032:  Learning rate = 0.001354:   Batch Loss = 0.428010, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.466933310032, Accuracy = 0.957920372486
    Iter #3248128:  Learning rate = 0.001354:   Batch Loss = 0.446929, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.458814948797, Accuracy = 0.959485292435
    Iter #3252224:  Learning rate = 0.001354:   Batch Loss = 0.404374, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.466149657965, Accuracy = 0.956529319286
    Iter #3256320:  Learning rate = 0.001354:   Batch Loss = 0.399419, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.453186094761, Accuracy = 0.960354745388
    Iter #3260416:  Learning rate = 0.001354:   Batch Loss = 0.430823, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.459673374891, Accuracy = 0.956703186035
    Iter #3264512:  Learning rate = 0.001354:   Batch Loss = 0.420263, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47792750597, Accuracy = 0.950443387032
    Iter #3268608:  Learning rate = 0.001354:   Batch Loss = 0.383632, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.468104183674, Accuracy = 0.956877052784
    Iter #3272704:  Learning rate = 0.001354:   Batch Loss = 0.409382, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.502239942551, Accuracy = 0.948878467083
    Iter #3276800:  Learning rate = 0.001354:   Batch Loss = 0.437652, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.478014111519, Accuracy = 0.956181526184
    Iter #3280896:  Learning rate = 0.001354:   Batch Loss = 0.434964, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.485130012035, Accuracy = 0.951486706734
    Iter #3284992:  Learning rate = 0.001354:   Batch Loss = 0.431435, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.470917582512, Accuracy = 0.960354745388
    Iter #3289088:  Learning rate = 0.001354:   Batch Loss = 0.399365, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.470937997103, Accuracy = 0.955659866333
    Iter #3293184:  Learning rate = 0.001354:   Batch Loss = 0.437801, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.464774638414, Accuracy = 0.960354745388
    Iter #3297280:  Learning rate = 0.001354:   Batch Loss = 0.409548, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.469018220901, Accuracy = 0.959659218788
    Iter #3301376:  Learning rate = 0.001300:   Batch Loss = 0.450224, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.466570496559, Accuracy = 0.957920372486
    Iter #3305472:  Learning rate = 0.001300:   Batch Loss = 0.431774, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.483024060726, Accuracy = 0.953051626682
    Iter #3309568:  Learning rate = 0.001300:   Batch Loss = 0.406745, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.454529523849, Accuracy = 0.959137558937
    Iter #3313664:  Learning rate = 0.001300:   Batch Loss = 0.456813, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.529718518257, Accuracy = 0.93844550848
    Iter #3317760:  Learning rate = 0.001300:   Batch Loss = 0.423426, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.454277426004, Accuracy = 0.960180819035
    Iter #3321856:  Learning rate = 0.001300:   Batch Loss = 0.416942, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.478393137455, Accuracy = 0.960180819035
    Iter #3325952:  Learning rate = 0.001300:   Batch Loss = 0.425396, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.464192509651, Accuracy = 0.958094239235
    Iter #3330048:  Learning rate = 0.001300:   Batch Loss = 0.379470, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.449434250593, Accuracy = 0.962615191936
    Iter #3334144:  Learning rate = 0.001300:   Batch Loss = 0.422816, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.450782716274, Accuracy = 0.960354745388
    Iter #3338240:  Learning rate = 0.001300:   Batch Loss = 0.401498, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.448059678078, Accuracy = 0.962615191936
    Iter #3342336:  Learning rate = 0.001300:   Batch Loss = 0.414300, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.46685385704, Accuracy = 0.956877052784
    Iter #3346432:  Learning rate = 0.001300:   Batch Loss = 0.415039, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.476107209921, Accuracy = 0.957572579384
    Iter #3350528:  Learning rate = 0.001300:   Batch Loss = 0.400359, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.450373500586, Accuracy = 0.961050271988
    Iter #3354624:  Learning rate = 0.001300:   Batch Loss = 0.405414, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.451531410217, Accuracy = 0.963484585285
    Iter #3358720:  Learning rate = 0.001300:   Batch Loss = 0.408032, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.45921677351, Accuracy = 0.959137558937
    Iter #3362816:  Learning rate = 0.001300:   Batch Loss = 0.427566, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.455715894699, Accuracy = 0.958268105984
    Iter #3366912:  Learning rate = 0.001300:   Batch Loss = 0.435434, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.450867503881, Accuracy = 0.965745091438
    Iter #3371008:  Learning rate = 0.001300:   Batch Loss = 0.417337, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.487876445055, Accuracy = 0.948356807232
    Iter #3375104:  Learning rate = 0.001300:   Batch Loss = 0.407743, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.460996568203, Accuracy = 0.953573286533
    Iter #3379200:  Learning rate = 0.001300:   Batch Loss = 0.420568, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.449542701244, Accuracy = 0.960354745388
    Iter #3383296:  Learning rate = 0.001300:   Batch Loss = 0.404092, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.459773123264, Accuracy = 0.956703186035
    Iter #3387392:  Learning rate = 0.001300:   Batch Loss = 0.394324, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.467958211899, Accuracy = 0.955833792686
    Iter #3391488:  Learning rate = 0.001300:   Batch Loss = 0.436762, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.454323381186, Accuracy = 0.958789765835
    Iter #3395584:  Learning rate = 0.001300:   Batch Loss = 0.397809, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.450539380312, Accuracy = 0.962789058685
    Iter #3399680:  Learning rate = 0.001300:   Batch Loss = 0.432379, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.488848567009, Accuracy = 0.947139620781
    Iter #3403776:  Learning rate = 0.001248:   Batch Loss = 0.412279, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.439525395632, Accuracy = 0.964527904987
    Iter #3407872:  Learning rate = 0.001248:   Batch Loss = 0.431976, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.445790886879, Accuracy = 0.964701771736
    Iter #3411968:  Learning rate = 0.001248:   Batch Loss = 0.412835, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.494063675404, Accuracy = 0.946444094181
    Iter #3416064:  Learning rate = 0.001248:   Batch Loss = 0.428705, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.454343765974, Accuracy = 0.960876345634
    Iter #3420160:  Learning rate = 0.001248:   Batch Loss = 0.422040, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.453034639359, Accuracy = 0.959485292435
    Iter #3424256:  Learning rate = 0.001248:   Batch Loss = 0.411746, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.465327620506, Accuracy = 0.953921079636
    Iter #3428352:  Learning rate = 0.001248:   Batch Loss = 0.487589, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.525423765182, Accuracy = 0.939662694931
    Iter #3432448:  Learning rate = 0.001248:   Batch Loss = 0.410741, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.466795653105, Accuracy = 0.952703893185
    Iter #3436544:  Learning rate = 0.001248:   Batch Loss = 0.423668, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.456340402365, Accuracy = 0.958615899086
    Iter #3440640:  Learning rate = 0.001248:   Batch Loss = 0.390420, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.443068295717, Accuracy = 0.960876345634
    Iter #3444736:  Learning rate = 0.001248:   Batch Loss = 0.407894, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.444025039673, Accuracy = 0.961224138737
    Iter #3448832:  Learning rate = 0.001248:   Batch Loss = 0.427869, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.483983784914, Accuracy = 0.951312839985
    Iter #3452928:  Learning rate = 0.001248:   Batch Loss = 0.400770, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.471874713898, Accuracy = 0.951834440231
    Iter #3457024:  Learning rate = 0.001248:   Batch Loss = 0.413275, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.441615462303, Accuracy = 0.961571872234
    Iter #3461120:  Learning rate = 0.001248:   Batch Loss = 0.429066, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.472710222006, Accuracy = 0.950095653534
    Iter #3465216:  Learning rate = 0.001248:   Batch Loss = 0.434275, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.437080860138, Accuracy = 0.965223431587
    Iter #3469312:  Learning rate = 0.001248:   Batch Loss = 0.416779, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.461706250906, Accuracy = 0.956877052784
    Iter #3473408:  Learning rate = 0.001248:   Batch Loss = 0.416671, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.445232510567, Accuracy = 0.961050271988
    Iter #3477504:  Learning rate = 0.001248:   Batch Loss = 0.430764, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.46172773838, Accuracy = 0.956355392933
    Iter #3481600:  Learning rate = 0.001248:   Batch Loss = 0.415089, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.46503034234, Accuracy = 0.957224845886
    Iter #3485696:  Learning rate = 0.001248:   Batch Loss = 0.455429, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.466786444187, Accuracy = 0.952356100082
    Iter #3489792:  Learning rate = 0.001248:   Batch Loss = 0.427324, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.465990364552, Accuracy = 0.956007659435
    Iter #3493888:  Learning rate = 0.001248:   Batch Loss = 0.452880, Accuracy = 0.95703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.454936087132, Accuracy = 0.957746505737
    Iter #3497984:  Learning rate = 0.001248:   Batch Loss = 0.394426, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.452684700489, Accuracy = 0.959485292435
    Iter #3502080:  Learning rate = 0.001198:   Batch Loss = 0.379060, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.432284653187, Accuracy = 0.963832378387
    Iter #3506176:  Learning rate = 0.001198:   Batch Loss = 0.408635, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.443799734116, Accuracy = 0.962789058685
    Iter #3510272:  Learning rate = 0.001198:   Batch Loss = 0.406872, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.451291680336, Accuracy = 0.962962985039
    Iter #3514368:  Learning rate = 0.001198:   Batch Loss = 0.409272, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.457384288311, Accuracy = 0.962789058685
    Iter #3518464:  Learning rate = 0.001198:   Batch Loss = 0.399815, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.456718623638, Accuracy = 0.958442032337
    Iter #3522560:  Learning rate = 0.001198:   Batch Loss = 0.412853, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.447014182806, Accuracy = 0.962441325188
    Iter #3526656:  Learning rate = 0.001198:   Batch Loss = 0.401678, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.453399628401, Accuracy = 0.956703186035
    Iter #3530752:  Learning rate = 0.001198:   Batch Loss = 0.416223, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.461005091667, Accuracy = 0.956703186035
    Iter #3534848:  Learning rate = 0.001198:   Batch Loss = 0.413100, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.451637983322, Accuracy = 0.958094239235
    Iter #3538944:  Learning rate = 0.001198:   Batch Loss = 0.400703, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.443166553974, Accuracy = 0.957224845886
    Iter #3543040:  Learning rate = 0.001198:   Batch Loss = 0.415596, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.501101136208, Accuracy = 0.94383585453
    Iter #3547136:  Learning rate = 0.001198:   Batch Loss = 0.413630, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.462069094181, Accuracy = 0.957746505737
    Iter #3551232:  Learning rate = 0.001198:   Batch Loss = 0.419681, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.455313444138, Accuracy = 0.954268813133
    Iter #3555328:  Learning rate = 0.001198:   Batch Loss = 0.406943, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.460124015808, Accuracy = 0.954964339733
    Iter #3559424:  Learning rate = 0.001198:   Batch Loss = 0.426286, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.465258091688, Accuracy = 0.954442679882
    Iter #3563520:  Learning rate = 0.001198:   Batch Loss = 0.412032, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47254806757, Accuracy = 0.950791180134
    Iter #3567616:  Learning rate = 0.001198:   Batch Loss = 0.401519, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.452182978392, Accuracy = 0.956703186035
    Iter #3571712:  Learning rate = 0.001198:   Batch Loss = 0.430800, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.446213364601, Accuracy = 0.960006952286
    Iter #3575808:  Learning rate = 0.001198:   Batch Loss = 0.414326, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.44700679183, Accuracy = 0.960006952286
    Iter #3579904:  Learning rate = 0.001198:   Batch Loss = 0.414716, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.437302976847, Accuracy = 0.964701771736
    Iter #3584000:  Learning rate = 0.001198:   Batch Loss = 0.396957, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.443421810865, Accuracy = 0.963658511639
    Iter #3588096:  Learning rate = 0.001198:   Batch Loss = 0.386240, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.444532692432, Accuracy = 0.962267458439
    Iter #3592192:  Learning rate = 0.001198:   Batch Loss = 0.388991, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.434926390648, Accuracy = 0.963832378387
    Iter #3596288:  Learning rate = 0.001198:   Batch Loss = 0.386139, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.43326702714, Accuracy = 0.964527904987
    Iter #3600384:  Learning rate = 0.001150:   Batch Loss = 0.393312, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.437671124935, Accuracy = 0.964006245136
    Iter #3604480:  Learning rate = 0.001150:   Batch Loss = 0.409384, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.437548577785, Accuracy = 0.960702478886
    Iter #3608576:  Learning rate = 0.001150:   Batch Loss = 0.409769, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.441861629486, Accuracy = 0.959833085537
    Iter #3612672:  Learning rate = 0.001150:   Batch Loss = 0.379100, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.428964346647, Accuracy = 0.964701771736
    Iter #3616768:  Learning rate = 0.001150:   Batch Loss = 0.411531, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.464727282524, Accuracy = 0.960354745388
    Iter #3620864:  Learning rate = 0.001150:   Batch Loss = 0.392365, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.474364995956, Accuracy = 0.951138913631
    Iter #3624960:  Learning rate = 0.001150:   Batch Loss = 0.447467, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.451065778732, Accuracy = 0.957746505737
    Iter #3629056:  Learning rate = 0.001150:   Batch Loss = 0.405254, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.447074055672, Accuracy = 0.960006952286
    Iter #3633152:  Learning rate = 0.001150:   Batch Loss = 0.422385, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.45782867074, Accuracy = 0.957746505737
    Iter #3637248:  Learning rate = 0.001150:   Batch Loss = 0.396159, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.441251039505, Accuracy = 0.958789765835
    Iter #3641344:  Learning rate = 0.001150:   Batch Loss = 0.406409, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.45713108778, Accuracy = 0.954268813133
    Iter #3645440:  Learning rate = 0.001150:   Batch Loss = 0.465105, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.450235784054, Accuracy = 0.957920372486
    Iter #3649536:  Learning rate = 0.001150:   Batch Loss = 0.408192, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.466484248638, Accuracy = 0.954442679882
    Iter #3653632:  Learning rate = 0.001150:   Batch Loss = 0.396042, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.453373372555, Accuracy = 0.956877052784
    Iter #3657728:  Learning rate = 0.001150:   Batch Loss = 0.400229, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.44368276, Accuracy = 0.960702478886
    Iter #3661824:  Learning rate = 0.001150:   Batch Loss = 0.387728, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.449583411217, Accuracy = 0.956529319286
    Iter #3665920:  Learning rate = 0.001150:   Batch Loss = 0.422633, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.450544029474, Accuracy = 0.96418017149
    Iter #3670016:  Learning rate = 0.001150:   Batch Loss = 0.408096, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.421519577503, Accuracy = 0.968527197838
    Iter #3674112:  Learning rate = 0.001150:   Batch Loss = 0.385240, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.451825559139, Accuracy = 0.960006952286
    Iter #3678208:  Learning rate = 0.001150:   Batch Loss = 0.388846, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.449375927448, Accuracy = 0.960006952286
    Iter #3682304:  Learning rate = 0.001150:   Batch Loss = 0.400884, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.450031459332, Accuracy = 0.956181526184
    Iter #3686400:  Learning rate = 0.001150:   Batch Loss = 0.450146, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.426475167274, Accuracy = 0.965397298336
    Iter #3690496:  Learning rate = 0.001150:   Batch Loss = 0.395005, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.431738615036, Accuracy = 0.965223431587
    Iter #3694592:  Learning rate = 0.001150:   Batch Loss = 0.384379, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.427776902914, Accuracy = 0.965223431587
    Iter #3698688:  Learning rate = 0.001150:   Batch Loss = 0.381292, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.422383964062, Accuracy = 0.96678841114
    Iter #3702784:  Learning rate = 0.001104:   Batch Loss = 0.388819, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.438356310129, Accuracy = 0.964006245136
    Iter #3706880:  Learning rate = 0.001104:   Batch Loss = 0.402037, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.424288213253, Accuracy = 0.967657804489
    Iter #3710976:  Learning rate = 0.001104:   Batch Loss = 0.396915, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.45872938633, Accuracy = 0.955659866333
    Iter #3715072:  Learning rate = 0.001104:   Batch Loss = 0.430575, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.465482652187, Accuracy = 0.954790472984
    Iter #3719168:  Learning rate = 0.001104:   Batch Loss = 0.399867, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.449563831091, Accuracy = 0.957398712635
    Iter #3723264:  Learning rate = 0.001104:   Batch Loss = 0.409710, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.442767947912, Accuracy = 0.960702478886
    Iter #3727360:  Learning rate = 0.001104:   Batch Loss = 0.388582, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.445575892925, Accuracy = 0.957224845886
    Iter #3731456:  Learning rate = 0.001104:   Batch Loss = 0.400091, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.438044339418, Accuracy = 0.962093532085
    Iter #3735552:  Learning rate = 0.001104:   Batch Loss = 0.437596, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.467631220818, Accuracy = 0.950965046883
    Iter #3739648:  Learning rate = 0.001104:   Batch Loss = 0.412531, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.450662791729, Accuracy = 0.956877052784
    Iter #3743744:  Learning rate = 0.001104:   Batch Loss = 0.396816, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.43291336298, Accuracy = 0.963484585285
    Iter #3747840:  Learning rate = 0.001104:   Batch Loss = 0.382977, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.432308226824, Accuracy = 0.963484585285
    Iter #3751936:  Learning rate = 0.001104:   Batch Loss = 0.404414, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.439086467028, Accuracy = 0.961398005486
    Iter #3756032:  Learning rate = 0.001104:   Batch Loss = 0.411017, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.437113910913, Accuracy = 0.963136851788
    Iter #3760128:  Learning rate = 0.001104:   Batch Loss = 0.390811, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.425576865673, Accuracy = 0.96678841114
    Iter #3764224:  Learning rate = 0.001104:   Batch Loss = 0.405442, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.42160898447, Accuracy = 0.964527904987
    Iter #3768320:  Learning rate = 0.001104:   Batch Loss = 0.381656, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.418114244938, Accuracy = 0.965918958187
    Iter #3772416:  Learning rate = 0.001104:   Batch Loss = 0.363726, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.432309478521, Accuracy = 0.966440618038
    Iter #3776512:  Learning rate = 0.001104:   Batch Loss = 0.404822, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.424257695675, Accuracy = 0.96487569809
    Iter #3780608:  Learning rate = 0.001104:   Batch Loss = 0.393033, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.440357387066, Accuracy = 0.959833085537
    Iter #3784704:  Learning rate = 0.001104:   Batch Loss = 0.400212, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.455943346024, Accuracy = 0.954268813133
    Iter #3788800:  Learning rate = 0.001104:   Batch Loss = 0.397491, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.44909465313, Accuracy = 0.960006952286
    Iter #3792896:  Learning rate = 0.001104:   Batch Loss = 0.393609, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.430214464664, Accuracy = 0.963310718536
    Iter #3796992:  Learning rate = 0.001104:   Batch Loss = 0.375230, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.443492174149, Accuracy = 0.959137558937
    Iter #3801088:  Learning rate = 0.001060:   Batch Loss = 0.447903, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.449505507946, Accuracy = 0.959485292435
    Iter #3805184:  Learning rate = 0.001060:   Batch Loss = 0.398914, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.449654400349, Accuracy = 0.960528612137
    Iter #3809280:  Learning rate = 0.001060:   Batch Loss = 0.398484, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.429715752602, Accuracy = 0.962441325188
    Iter #3813376:  Learning rate = 0.001060:   Batch Loss = 0.400129, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.456677377224, Accuracy = 0.953051626682
    Iter #3817472:  Learning rate = 0.001060:   Batch Loss = 0.385887, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.436928838491, Accuracy = 0.961050271988
    Iter #3821568:  Learning rate = 0.001060:   Batch Loss = 0.390435, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.422595560551, Accuracy = 0.965571224689
    Iter #3825664:  Learning rate = 0.001060:   Batch Loss = 0.387302, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.429408043623, Accuracy = 0.960180819035
    Iter #3829760:  Learning rate = 0.001060:   Batch Loss = 0.404831, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.432101547718, Accuracy = 0.963658511639
    Iter #3833856:  Learning rate = 0.001060:   Batch Loss = 0.387796, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.416058212519, Accuracy = 0.965397298336
    Iter #3837952:  Learning rate = 0.001060:   Batch Loss = 0.393688, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.460973322392, Accuracy = 0.957572579384
    Iter #3842048:  Learning rate = 0.001060:   Batch Loss = 0.373502, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.429209828377, Accuracy = 0.96487569809
    Iter #3846144:  Learning rate = 0.001060:   Batch Loss = 0.358900, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.420609742403, Accuracy = 0.968353331089
    Iter #3850240:  Learning rate = 0.001060:   Batch Loss = 0.398097, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.415197253227, Accuracy = 0.968005537987
    Iter #3854336:  Learning rate = 0.001060:   Batch Loss = 0.375943, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.41514390707, Accuracy = 0.96817946434
    Iter #3858432:  Learning rate = 0.001060:   Batch Loss = 0.367884, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.422264695168, Accuracy = 0.966092824936
    Iter #3862528:  Learning rate = 0.001060:   Batch Loss = 0.376769, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.421447455883, Accuracy = 0.964701771736
    Iter #3866624:  Learning rate = 0.001060:   Batch Loss = 0.361880, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.42438608408, Accuracy = 0.968353331089
    Iter #3870720:  Learning rate = 0.001060:   Batch Loss = 0.402562, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.426223903894, Accuracy = 0.966266751289
    Iter #3874816:  Learning rate = 0.001060:   Batch Loss = 0.377783, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.426442861557, Accuracy = 0.965397298336
    Iter #3878912:  Learning rate = 0.001060:   Batch Loss = 0.385000, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.410761892796, Accuracy = 0.970613777637
    Iter #3883008:  Learning rate = 0.001060:   Batch Loss = 0.356434, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.437784582376, Accuracy = 0.964006245136
    Iter #3887104:  Learning rate = 0.001060:   Batch Loss = 0.367546, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.41700693965, Accuracy = 0.966962277889
    Iter #3891200:  Learning rate = 0.001060:   Batch Loss = 0.374793, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.411307930946, Accuracy = 0.969048857689
    Iter #3895296:  Learning rate = 0.001060:   Batch Loss = 0.374900, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.420896738768, Accuracy = 0.963310718536
    Iter #3899392:  Learning rate = 0.001060:   Batch Loss = 0.387443, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.421903669834, Accuracy = 0.963484585285
    Iter #3903488:  Learning rate = 0.001018:   Batch Loss = 0.394246, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.426094055176, Accuracy = 0.963310718536
    Iter #3907584:  Learning rate = 0.001018:   Batch Loss = 0.409358, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.41966098547, Accuracy = 0.965397298336
    Iter #3911680:  Learning rate = 0.001018:   Batch Loss = 0.389987, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.415071576834, Accuracy = 0.966614484787
    Iter #3915776:  Learning rate = 0.001018:   Batch Loss = 0.402103, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.439166367054, Accuracy = 0.965397298336
    Iter #3919872:  Learning rate = 0.001018:   Batch Loss = 0.378256, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.416054785252, Accuracy = 0.968005537987
    Iter #3923968:  Learning rate = 0.001018:   Batch Loss = 0.370140, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.411992371082, Accuracy = 0.965918958187
    Iter #3928064:  Learning rate = 0.001018:   Batch Loss = 0.371110, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.430589973927, Accuracy = 0.960006952286
    Iter #3932160:  Learning rate = 0.001018:   Batch Loss = 0.367893, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.426213145256, Accuracy = 0.962962985039
    Iter #3936256:  Learning rate = 0.001018:   Batch Loss = 0.389656, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.44117474556, Accuracy = 0.955485999584
    Iter #3940352:  Learning rate = 0.001018:   Batch Loss = 0.397812, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.433076024055, Accuracy = 0.965571224689
    Iter #3944448:  Learning rate = 0.001018:   Batch Loss = 0.383902, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.41235524416, Accuracy = 0.96817946434
    Iter #3948544:  Learning rate = 0.001018:   Batch Loss = 0.376775, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.440006017685, Accuracy = 0.96418017149
    Iter #3952640:  Learning rate = 0.001018:   Batch Loss = 0.393504, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.423459619284, Accuracy = 0.964354038239
    Iter #3956736:  Learning rate = 0.001018:   Batch Loss = 0.378033, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.418031692505, Accuracy = 0.96418017149
    Iter #3960832:  Learning rate = 0.001018:   Batch Loss = 0.365578, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.418415546417, Accuracy = 0.967136144638
    Iter #3964928:  Learning rate = 0.001018:   Batch Loss = 0.365543, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.402402490377, Accuracy = 0.971309363842
    Iter #3969024:  Learning rate = 0.001018:   Batch Loss = 0.403500, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.472196280956, Accuracy = 0.94592243433
    Iter #3973120:  Learning rate = 0.001018:   Batch Loss = 0.385263, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.434478491545, Accuracy = 0.961224138737
    Iter #3977216:  Learning rate = 0.001018:   Batch Loss = 0.375830, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.424495220184, Accuracy = 0.96487569809
    Iter #3981312:  Learning rate = 0.001018:   Batch Loss = 0.412971, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.43740439415, Accuracy = 0.959659218788
    Iter #3985408:  Learning rate = 0.001018:   Batch Loss = 0.370675, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.435737967491, Accuracy = 0.958094239235
    Iter #3989504:  Learning rate = 0.001018:   Batch Loss = 0.411240, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.471098542213, Accuracy = 0.947139620781
    Iter #3993600:  Learning rate = 0.001018:   Batch Loss = 0.432892, Accuracy = 0.966796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.490505665541, Accuracy = 0.94592243433
    Iter #3997696:  Learning rate = 0.001018:   Batch Loss = 0.431836, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.471243023872, Accuracy = 0.949226200581
    Iter #4001792:  Learning rate = 0.000977:   Batch Loss = 0.429455, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.468992829323, Accuracy = 0.948182940483
    Iter #4005888:  Learning rate = 0.000977:   Batch Loss = 0.395098, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.43634480238, Accuracy = 0.961919665337
    Iter #4009984:  Learning rate = 0.000977:   Batch Loss = 0.393130, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.434394091368, Accuracy = 0.963310718536
    Iter #4014080:  Learning rate = 0.000977:   Batch Loss = 0.390284, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.431989520788, Accuracy = 0.961745798588
    Iter #4018176:  Learning rate = 0.000977:   Batch Loss = 0.418814, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.432697117329, Accuracy = 0.958442032337
    Iter #4022272:  Learning rate = 0.000977:   Batch Loss = 0.386456, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.433618038893, Accuracy = 0.962615191936
    Iter #4026368:  Learning rate = 0.000977:   Batch Loss = 0.379651, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.419236838818, Accuracy = 0.966092824936
    Iter #4030464:  Learning rate = 0.000977:   Batch Loss = 0.399858, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.455546498299, Accuracy = 0.952529966831
    Iter #4034560:  Learning rate = 0.000977:   Batch Loss = 0.424745, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.424670040607, Accuracy = 0.965745091438
    Iter #4038656:  Learning rate = 0.000977:   Batch Loss = 0.368170, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.432592391968, Accuracy = 0.960354745388
    Iter #4042752:  Learning rate = 0.000977:   Batch Loss = 0.393891, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.416328310966, Accuracy = 0.964701771736
    Iter #4046848:  Learning rate = 0.000977:   Batch Loss = 0.401851, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.419160187244, Accuracy = 0.965397298336
    Iter #4050944:  Learning rate = 0.000977:   Batch Loss = 0.383781, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.423865437508, Accuracy = 0.962267458439
    Iter #4055040:  Learning rate = 0.000977:   Batch Loss = 0.360243, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.418378025293, Accuracy = 0.964006245136
    Iter #4059136:  Learning rate = 0.000977:   Batch Loss = 0.377486, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.409683525562, Accuracy = 0.967310011387
    Iter #4063232:  Learning rate = 0.000977:   Batch Loss = 0.379162, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.434152215719, Accuracy = 0.961571872234
    Iter #4067328:  Learning rate = 0.000977:   Batch Loss = 0.390117, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.423316329718, Accuracy = 0.965049564838
    Iter #4071424:  Learning rate = 0.000977:   Batch Loss = 0.366179, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.425166875124, Accuracy = 0.961571872234
    Iter #4075520:  Learning rate = 0.000977:   Batch Loss = 0.372728, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.436886131763, Accuracy = 0.961398005486
    Iter #4079616:  Learning rate = 0.000977:   Batch Loss = 0.401634, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.431874990463, Accuracy = 0.960528612137
    Iter #4083712:  Learning rate = 0.000977:   Batch Loss = 0.399741, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.448357641697, Accuracy = 0.958268105984
    Iter #4087808:  Learning rate = 0.000977:   Batch Loss = 0.362958, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.429533779621, Accuracy = 0.962267458439
    Iter #4091904:  Learning rate = 0.000977:   Batch Loss = 0.403888, Accuracy = 0.96484375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.429616123438, Accuracy = 0.962789058685
    Iter #4096000:  Learning rate = 0.000977:   Batch Loss = 0.378364, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.442798554897, Accuracy = 0.954442679882
    Iter #4100096:  Learning rate = 0.000938:   Batch Loss = 0.376248, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.420575231314, Accuracy = 0.962789058685
    Iter #4104192:  Learning rate = 0.000938:   Batch Loss = 0.391083, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.433288693428, Accuracy = 0.960528612137
    Iter #4108288:  Learning rate = 0.000938:   Batch Loss = 0.390684, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.440288841724, Accuracy = 0.956877052784
    Iter #4112384:  Learning rate = 0.000938:   Batch Loss = 0.371218, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.424109458923, Accuracy = 0.961224138737
    Iter #4116480:  Learning rate = 0.000938:   Batch Loss = 0.391468, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.418218791485, Accuracy = 0.962789058685
    Iter #4120576:  Learning rate = 0.000938:   Batch Loss = 0.396059, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.420795679092, Accuracy = 0.965223431587
    Iter #4124672:  Learning rate = 0.000938:   Batch Loss = 0.372458, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.41367033124, Accuracy = 0.965571224689
    Iter #4128768:  Learning rate = 0.000938:   Batch Loss = 0.400713, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.421510100365, Accuracy = 0.964354038239
    Iter #4132864:  Learning rate = 0.000938:   Batch Loss = 0.374077, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.416084766388, Accuracy = 0.964701771736
    Iter #4136960:  Learning rate = 0.000938:   Batch Loss = 0.348195, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.424359053373, Accuracy = 0.965049564838
    Iter #4141056:  Learning rate = 0.000938:   Batch Loss = 0.416391, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.424696087837, Accuracy = 0.962615191936
    Iter #4145152:  Learning rate = 0.000938:   Batch Loss = 0.423287, Accuracy = 0.9609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.464932739735, Accuracy = 0.946444094181
    Iter #4149248:  Learning rate = 0.000938:   Batch Loss = 0.391943, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.446657419205, Accuracy = 0.957224845886
    Iter #4153344:  Learning rate = 0.000938:   Batch Loss = 0.416779, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.459856510162, Accuracy = 0.955312132835
    Iter #4157440:  Learning rate = 0.000938:   Batch Loss = 0.362991, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4347217381, Accuracy = 0.959833085537
    Iter #4161536:  Learning rate = 0.000938:   Batch Loss = 0.363894, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.415193021297, Accuracy = 0.96418017149
    Iter #4165632:  Learning rate = 0.000938:   Batch Loss = 0.376909, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.453539043665, Accuracy = 0.952182233334
    Iter #4169728:  Learning rate = 0.000938:   Batch Loss = 0.366804, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.411960631609, Accuracy = 0.965571224689
    Iter #4173824:  Learning rate = 0.000938:   Batch Loss = 0.458476, Accuracy = 0.955078125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.506154894829, Accuracy = 0.93914103508
    Iter #4177920:  Learning rate = 0.000938:   Batch Loss = 0.366352, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.412436187267, Accuracy = 0.965745091438
    Iter #4182016:  Learning rate = 0.000938:   Batch Loss = 0.376578, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.438769340515, Accuracy = 0.958789765835
    Iter #4186112:  Learning rate = 0.000938:   Batch Loss = 0.378808, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.425468474627, Accuracy = 0.962615191936
    Iter #4190208:  Learning rate = 0.000938:   Batch Loss = 0.374133, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.418907999992, Accuracy = 0.964354038239
    Iter #4194304:  Learning rate = 0.000938:   Batch Loss = 0.379599, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.425098955631, Accuracy = 0.961745798588
    Iter #4198400:  Learning rate = 0.000938:   Batch Loss = 0.345739, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.405580788851, Accuracy = 0.968005537987
    Iter #4202496:  Learning rate = 0.000900:   Batch Loss = 0.381599, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.40785163641, Accuracy = 0.96817946434
    Iter #4206592:  Learning rate = 0.000900:   Batch Loss = 0.383379, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.443982511759, Accuracy = 0.960876345634
    Iter #4210688:  Learning rate = 0.000900:   Batch Loss = 0.363980, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.41207665205, Accuracy = 0.965571224689
    Iter #4214784:  Learning rate = 0.000900:   Batch Loss = 0.392336, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.404804080725, Accuracy = 0.968353331089
    Iter #4218880:  Learning rate = 0.000900:   Batch Loss = 0.412692, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.431446760893, Accuracy = 0.961571872234
    Iter #4222976:  Learning rate = 0.000900:   Batch Loss = 0.370717, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.414444208145, Accuracy = 0.966440618038
    Iter #4227072:  Learning rate = 0.000900:   Batch Loss = 0.391201, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.402393996716, Accuracy = 0.96957051754
    Iter #4231168:  Learning rate = 0.000900:   Batch Loss = 0.369212, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.406198173761, Accuracy = 0.96678841114
    Iter #4235264:  Learning rate = 0.000900:   Batch Loss = 0.361402, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.410109311342, Accuracy = 0.965571224689
    Iter #4239360:  Learning rate = 0.000900:   Batch Loss = 0.351964, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.407051980495, Accuracy = 0.96817946434
    Iter #4243456:  Learning rate = 0.000900:   Batch Loss = 0.353647, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.412388414145, Accuracy = 0.966266751289
    Iter #4247552:  Learning rate = 0.000900:   Batch Loss = 0.344414, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.405959665775, Accuracy = 0.965745091438
    Iter #4251648:  Learning rate = 0.000900:   Batch Loss = 0.367214, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.409380167723, Accuracy = 0.966440618038
    Iter #4255744:  Learning rate = 0.000900:   Batch Loss = 0.355810, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.410929739475, Accuracy = 0.965918958187
    Iter #4259840:  Learning rate = 0.000900:   Batch Loss = 0.362029, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.420193493366, Accuracy = 0.965223431587
    Iter #4263936:  Learning rate = 0.000900:   Batch Loss = 0.368746, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.40915375948, Accuracy = 0.969048857689
    Iter #4268032:  Learning rate = 0.000900:   Batch Loss = 0.393586, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.415149986744, Accuracy = 0.965397298336
    Iter #4272128:  Learning rate = 0.000900:   Batch Loss = 0.360026, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.409035801888, Accuracy = 0.968353331089
    Iter #4276224:  Learning rate = 0.000900:   Batch Loss = 0.367338, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.433008670807, Accuracy = 0.962093532085
    Iter #4280320:  Learning rate = 0.000900:   Batch Loss = 0.397190, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.405999481678, Accuracy = 0.969744384289
    Iter #4284416:  Learning rate = 0.000900:   Batch Loss = 0.372750, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.414466887712, Accuracy = 0.965745091438
    Iter #4288512:  Learning rate = 0.000900:   Batch Loss = 0.359993, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.40972507, Accuracy = 0.965571224689
    Iter #4292608:  Learning rate = 0.000900:   Batch Loss = 0.372232, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.397840678692, Accuracy = 0.972004890442
    Iter #4296704:  Learning rate = 0.000900:   Batch Loss = 0.383821, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.422474473715, Accuracy = 0.970613777637
    Iter #4300800:  Learning rate = 0.000864:   Batch Loss = 0.367249, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.406313031912, Accuracy = 0.969222724438
    Iter #4304896:  Learning rate = 0.000864:   Batch Loss = 0.367027, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.406243622303, Accuracy = 0.969222724438
    Iter #4308992:  Learning rate = 0.000864:   Batch Loss = 0.394835, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.426200062037, Accuracy = 0.962789058685
    Iter #4313088:  Learning rate = 0.000864:   Batch Loss = 0.367859, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.409400314093, Accuracy = 0.967136144638
    Iter #4317184:  Learning rate = 0.000864:   Batch Loss = 0.369404, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.414095968008, Accuracy = 0.964006245136
    Iter #4321280:  Learning rate = 0.000864:   Batch Loss = 0.352198, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.41194704175, Accuracy = 0.96957051754
    Iter #4325376:  Learning rate = 0.000864:   Batch Loss = 0.351506, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.406764060259, Accuracy = 0.967136144638
    Iter #4329472:  Learning rate = 0.000864:   Batch Loss = 0.376979, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.421779304743, Accuracy = 0.962615191936
    Iter #4333568:  Learning rate = 0.000864:   Batch Loss = 0.363575, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.415273874998, Accuracy = 0.965049564838
    Iter #4337664:  Learning rate = 0.000864:   Batch Loss = 0.358813, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.408319741488, Accuracy = 0.965745091438
    Iter #4341760:  Learning rate = 0.000864:   Batch Loss = 0.352763, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.40688893199, Accuracy = 0.968527197838
    Iter #4345856:  Learning rate = 0.000864:   Batch Loss = 0.403462, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.42007419467, Accuracy = 0.96487569809
    Iter #4349952:  Learning rate = 0.000864:   Batch Loss = 0.395720, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.41314125061, Accuracy = 0.962441325188
    Iter #4354048:  Learning rate = 0.000864:   Batch Loss = 0.358729, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.425068616867, Accuracy = 0.962093532085
    Iter #4358144:  Learning rate = 0.000864:   Batch Loss = 0.354103, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.412996470928, Accuracy = 0.965918958187
    Iter #4362240:  Learning rate = 0.000864:   Batch Loss = 0.389021, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.434592694044, Accuracy = 0.959485292435
    Iter #4366336:  Learning rate = 0.000864:   Batch Loss = 0.356108, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.414656162262, Accuracy = 0.966614484787
    Iter #4370432:  Learning rate = 0.000864:   Batch Loss = 0.364953, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.423166453838, Accuracy = 0.964006245136
    Iter #4374528:  Learning rate = 0.000864:   Batch Loss = 0.363368, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.406352907419, Accuracy = 0.965745091438
    Iter #4378624:  Learning rate = 0.000864:   Batch Loss = 0.354916, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.428495883942, Accuracy = 0.960354745388
    Iter #4382720:  Learning rate = 0.000864:   Batch Loss = 0.382668, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.411046028137, Accuracy = 0.967136144638
    Iter #4386816:  Learning rate = 0.000864:   Batch Loss = 0.384025, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.425131320953, Accuracy = 0.961224138737
    Iter #4390912:  Learning rate = 0.000864:   Batch Loss = 0.377907, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.42081168294, Accuracy = 0.962962985039
    Iter #4395008:  Learning rate = 0.000864:   Batch Loss = 0.379820, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.409230470657, Accuracy = 0.969048857689
    Iter #4399104:  Learning rate = 0.000864:   Batch Loss = 0.356313, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.420056462288, Accuracy = 0.963484585285
    Iter #4403200:  Learning rate = 0.000830:   Batch Loss = 0.355434, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.415637165308, Accuracy = 0.963136851788
    Iter #4407296:  Learning rate = 0.000830:   Batch Loss = 0.358069, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.418036460876, Accuracy = 0.962093532085
    Iter #4411392:  Learning rate = 0.000830:   Batch Loss = 0.372691, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.420405328274, Accuracy = 0.963484585285
    Iter #4415488:  Learning rate = 0.000830:   Batch Loss = 0.366863, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.413297325373, Accuracy = 0.965397298336
    Iter #4419584:  Learning rate = 0.000830:   Batch Loss = 0.365743, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.415819227695, Accuracy = 0.963484585285
    Iter #4423680:  Learning rate = 0.000830:   Batch Loss = 0.371015, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.416886299849, Accuracy = 0.962093532085
    Iter #4427776:  Learning rate = 0.000830:   Batch Loss = 0.366181, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.409236699343, Accuracy = 0.96678841114
    Iter #4431872:  Learning rate = 0.000830:   Batch Loss = 0.386790, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.436039924622, Accuracy = 0.960006952286
    Iter #4435968:  Learning rate = 0.000830:   Batch Loss = 0.375434, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.422995090485, Accuracy = 0.961571872234
    Iter #4440064:  Learning rate = 0.000830:   Batch Loss = 0.350608, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.405881643295, Accuracy = 0.967310011387
    Iter #4444160:  Learning rate = 0.000830:   Batch Loss = 0.345979, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.408935129642, Accuracy = 0.965918958187
    Iter #4448256:  Learning rate = 0.000830:   Batch Loss = 0.360747, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.421093255281, Accuracy = 0.962093532085
    Iter #4452352:  Learning rate = 0.000830:   Batch Loss = 0.414442, Accuracy = 0.958984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.420435965061, Accuracy = 0.961224138737
    Iter #4456448:  Learning rate = 0.000830:   Batch Loss = 0.345910, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.423450082541, Accuracy = 0.962962985039
    Iter #4460544:  Learning rate = 0.000830:   Batch Loss = 0.380995, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.40236517787, Accuracy = 0.96887499094
    Iter #4464640:  Learning rate = 0.000830:   Batch Loss = 0.354026, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.391606152058, Accuracy = 0.970613777637
    Iter #4468736:  Learning rate = 0.000830:   Batch Loss = 0.352680, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.432418137789, Accuracy = 0.963310718536
    Iter #4472832:  Learning rate = 0.000830:   Batch Loss = 0.353770, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.425382614136, Accuracy = 0.967831671238
    Iter #4476928:  Learning rate = 0.000830:   Batch Loss = 0.357077, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.400218993425, Accuracy = 0.96817946434
    Iter #4481024:  Learning rate = 0.000830:   Batch Loss = 0.380226, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.40814819932, Accuracy = 0.967136144638
    Iter #4485120:  Learning rate = 0.000830:   Batch Loss = 0.364831, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.427849918604, Accuracy = 0.962789058685
    Iter #4489216:  Learning rate = 0.000830:   Batch Loss = 0.345158, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.400811553001, Accuracy = 0.969048857689
    Iter #4493312:  Learning rate = 0.000830:   Batch Loss = 0.334206, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.398164123297, Accuracy = 0.967136144638
    Iter #4497408:  Learning rate = 0.000830:   Batch Loss = 0.357699, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.404444426298, Accuracy = 0.965397298336
    Iter #4501504:  Learning rate = 0.000796:   Batch Loss = 0.378881, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.409346938133, Accuracy = 0.965571224689
    Iter #4505600:  Learning rate = 0.000796:   Batch Loss = 0.366140, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.403492331505, Accuracy = 0.966962277889
    Iter #4509696:  Learning rate = 0.000796:   Batch Loss = 0.374958, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39796346426, Accuracy = 0.969396650791
    Iter #4513792:  Learning rate = 0.000796:   Batch Loss = 0.343544, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.402859896421, Accuracy = 0.969918251038
    Iter #4517888:  Learning rate = 0.000796:   Batch Loss = 0.345585, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.415194094181, Accuracy = 0.962615191936
    Iter #4521984:  Learning rate = 0.000796:   Batch Loss = 0.353808, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.424403458834, Accuracy = 0.962962985039
    Iter #4526080:  Learning rate = 0.000796:   Batch Loss = 0.363955, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.412498950958, Accuracy = 0.96678841114
    Iter #4530176:  Learning rate = 0.000796:   Batch Loss = 0.335798, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.420809537172, Accuracy = 0.965223431587
    Iter #4534272:  Learning rate = 0.000796:   Batch Loss = 0.350465, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.410486280918, Accuracy = 0.962789058685
    Iter #4538368:  Learning rate = 0.000796:   Batch Loss = 0.348761, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39503377676, Accuracy = 0.96748393774
    Iter #4542464:  Learning rate = 0.000796:   Batch Loss = 0.363694, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.422112524509, Accuracy = 0.959659218788
    Iter #4546560:  Learning rate = 0.000796:   Batch Loss = 0.345025, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.398208737373, Accuracy = 0.967831671238
    Iter #4550656:  Learning rate = 0.000796:   Batch Loss = 0.364605, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39610093832, Accuracy = 0.968527197838
    Iter #4554752:  Learning rate = 0.000796:   Batch Loss = 0.353391, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.428672581911, Accuracy = 0.961919665337
    Iter #4558848:  Learning rate = 0.000796:   Batch Loss = 0.373671, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.412909895182, Accuracy = 0.96418017149
    Iter #4562944:  Learning rate = 0.000796:   Batch Loss = 0.357977, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.401823103428, Accuracy = 0.968005537987
    Iter #4567040:  Learning rate = 0.000796:   Batch Loss = 0.352226, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.407903552055, Accuracy = 0.965223431587
    Iter #4571136:  Learning rate = 0.000796:   Batch Loss = 0.387451, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.390012025833, Accuracy = 0.97165709734
    Iter #4575232:  Learning rate = 0.000796:   Batch Loss = 0.338984, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.399227082729, Accuracy = 0.969918251038
    Iter #4579328:  Learning rate = 0.000796:   Batch Loss = 0.356392, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.391235619783, Accuracy = 0.972526490688
    Iter #4583424:  Learning rate = 0.000796:   Batch Loss = 0.355489, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.417676091194, Accuracy = 0.962615191936
    Iter #4587520:  Learning rate = 0.000796:   Batch Loss = 0.367297, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.393011599779, Accuracy = 0.970613777637
    Iter #4591616:  Learning rate = 0.000796:   Batch Loss = 0.369139, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.41529238224, Accuracy = 0.965571224689
    Iter #4595712:  Learning rate = 0.000796:   Batch Loss = 0.347243, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.398308396339, Accuracy = 0.969396650791
    Iter #4599808:  Learning rate = 0.000796:   Batch Loss = 0.348117, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.397771418095, Accuracy = 0.968005537987
    Iter #4603904:  Learning rate = 0.000765:   Batch Loss = 0.365683, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39335000515, Accuracy = 0.970439910889
    Iter #4608000:  Learning rate = 0.000765:   Batch Loss = 0.344089, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39796179533, Accuracy = 0.970613777637
    Iter #4612096:  Learning rate = 0.000765:   Batch Loss = 0.354289, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.42101675272, Accuracy = 0.969048857689
    Iter #4616192:  Learning rate = 0.000765:   Batch Loss = 0.355846, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.398806244135, Accuracy = 0.967310011387
    Iter #4620288:  Learning rate = 0.000765:   Batch Loss = 0.350036, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.3981731534, Accuracy = 0.967831671238
    Iter #4624384:  Learning rate = 0.000765:   Batch Loss = 0.360098, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.412255376577, Accuracy = 0.965049564838
    Iter #4628480:  Learning rate = 0.000765:   Batch Loss = 0.365380, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.440603166819, Accuracy = 0.956007659435
    Iter #4632576:  Learning rate = 0.000765:   Batch Loss = 0.361016, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.41297647357, Accuracy = 0.961050271988
    Iter #4636672:  Learning rate = 0.000765:   Batch Loss = 0.335856, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.426118522882, Accuracy = 0.960354745388
    Iter #4640768:  Learning rate = 0.000765:   Batch Loss = 0.341190, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.397452533245, Accuracy = 0.969744384289
    Iter #4644864:  Learning rate = 0.000765:   Batch Loss = 0.373128, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.393399208784, Accuracy = 0.968527197838
    Iter #4648960:  Learning rate = 0.000765:   Batch Loss = 0.369044, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.411000072956, Accuracy = 0.964006245136
    Iter #4653056:  Learning rate = 0.000765:   Batch Loss = 0.372461, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.409606665373, Accuracy = 0.967310011387
    Iter #4657152:  Learning rate = 0.000765:   Batch Loss = 0.384662, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.414841860533, Accuracy = 0.965571224689
    Iter #4661248:  Learning rate = 0.000765:   Batch Loss = 0.363368, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.391125977039, Accuracy = 0.971309363842
    Iter #4665344:  Learning rate = 0.000765:   Batch Loss = 0.356590, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.391750216484, Accuracy = 0.970787703991
    Iter #4669440:  Learning rate = 0.000765:   Batch Loss = 0.378336, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.390749484301, Accuracy = 0.968527197838
    Iter #4673536:  Learning rate = 0.000765:   Batch Loss = 0.361533, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.391392588615, Accuracy = 0.971830964088
    Iter #4677632:  Learning rate = 0.000765:   Batch Loss = 0.352556, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.411709398031, Accuracy = 0.962962985039
    Iter #4681728:  Learning rate = 0.000765:   Batch Loss = 0.348957, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.401605159044, Accuracy = 0.966440618038
    Iter #4685824:  Learning rate = 0.000765:   Batch Loss = 0.384599, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.413554280996, Accuracy = 0.962093532085
    Iter #4689920:  Learning rate = 0.000765:   Batch Loss = 0.365547, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.401436179876, Accuracy = 0.967310011387
    Iter #4694016:  Learning rate = 0.000765:   Batch Loss = 0.350742, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.394661813974, Accuracy = 0.968527197838
    Iter #4698112:  Learning rate = 0.000765:   Batch Loss = 0.356823, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.399377584457, Accuracy = 0.96817946434
    Iter #4702208:  Learning rate = 0.000734:   Batch Loss = 0.356911, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.391284137964, Accuracy = 0.97165709734
    Iter #4706304:  Learning rate = 0.000734:   Batch Loss = 0.357667, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.400249183178, Accuracy = 0.966092824936
    Iter #4710400:  Learning rate = 0.000734:   Batch Loss = 0.353550, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.419644832611, Accuracy = 0.963832378387
    Iter #4714496:  Learning rate = 0.000734:   Batch Loss = 0.351102, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.388881355524, Accuracy = 0.970613777637
    Iter #4718592:  Learning rate = 0.000734:   Batch Loss = 0.365459, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.395069926977, Accuracy = 0.968527197838
    Iter #4722688:  Learning rate = 0.000734:   Batch Loss = 0.354167, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.410922646523, Accuracy = 0.966092824936
    Iter #4726784:  Learning rate = 0.000734:   Batch Loss = 0.369807, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.394776135683, Accuracy = 0.968005537987
    Iter #4730880:  Learning rate = 0.000734:   Batch Loss = 0.335039, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.396281838417, Accuracy = 0.968701124191
    Iter #4734976:  Learning rate = 0.000734:   Batch Loss = 0.337201, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.407324314117, Accuracy = 0.963658511639
    Iter #4739072:  Learning rate = 0.000734:   Batch Loss = 0.340217, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.400988221169, Accuracy = 0.96487569809
    Iter #4743168:  Learning rate = 0.000734:   Batch Loss = 0.353139, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.396533131599, Accuracy = 0.967136144638
    Iter #4747264:  Learning rate = 0.000734:   Batch Loss = 0.338064, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.401565432549, Accuracy = 0.97096157074
    Iter #4751360:  Learning rate = 0.000734:   Batch Loss = 0.361718, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.414275884628, Accuracy = 0.96487569809
    Iter #4755456:  Learning rate = 0.000734:   Batch Loss = 0.346779, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.391309738159, Accuracy = 0.970613777637
    Iter #4759552:  Learning rate = 0.000734:   Batch Loss = 0.341849, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.388540536165, Accuracy = 0.97096157074
    Iter #4763648:  Learning rate = 0.000734:   Batch Loss = 0.337199, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.404559701681, Accuracy = 0.967831671238
    Iter #4767744:  Learning rate = 0.000734:   Batch Loss = 0.346367, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.388189196587, Accuracy = 0.97165709734
    Iter #4771840:  Learning rate = 0.000734:   Batch Loss = 0.342324, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.385699510574, Accuracy = 0.973395943642
    Iter #4775936:  Learning rate = 0.000734:   Batch Loss = 0.361578, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.395240962505, Accuracy = 0.969396650791
    Iter #4780032:  Learning rate = 0.000734:   Batch Loss = 0.348532, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.405320227146, Accuracy = 0.965397298336
    Iter #4784128:  Learning rate = 0.000734:   Batch Loss = 0.384497, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.415870457888, Accuracy = 0.960180819035
    Iter #4788224:  Learning rate = 0.000734:   Batch Loss = 0.350277, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.407819330692, Accuracy = 0.965571224689
    Iter #4792320:  Learning rate = 0.000734:   Batch Loss = 0.367018, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.411473065615, Accuracy = 0.965397298336
    Iter #4796416:  Learning rate = 0.000734:   Batch Loss = 0.396310, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.422610342503, Accuracy = 0.962615191936
    Iter #4800512:  Learning rate = 0.000705:   Batch Loss = 0.357392, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.398293167353, Accuracy = 0.968005537987
    Iter #4804608:  Learning rate = 0.000705:   Batch Loss = 0.351628, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.409670352936, Accuracy = 0.964527904987
    Iter #4808704:  Learning rate = 0.000705:   Batch Loss = 0.352806, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39875292778, Accuracy = 0.968005537987
    Iter #4812800:  Learning rate = 0.000705:   Batch Loss = 0.359271, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.389346450567, Accuracy = 0.969222724438
    Iter #4816896:  Learning rate = 0.000705:   Batch Loss = 0.348277, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.392634063959, Accuracy = 0.969744384289
    Iter #4820992:  Learning rate = 0.000705:   Batch Loss = 0.352284, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.396914333105, Accuracy = 0.966440618038
    Iter #4825088:  Learning rate = 0.000705:   Batch Loss = 0.364118, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.408409655094, Accuracy = 0.966092824936
    Iter #4829184:  Learning rate = 0.000705:   Batch Loss = 0.333258, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.394643902779, Accuracy = 0.96887499094
    Iter #4833280:  Learning rate = 0.000705:   Batch Loss = 0.355847, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.394071996212, Accuracy = 0.96678841114
    Iter #4837376:  Learning rate = 0.000705:   Batch Loss = 0.387129, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.385487854481, Accuracy = 0.971830964088
    Iter #4841472:  Learning rate = 0.000705:   Batch Loss = 0.354178, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.407031714916, Accuracy = 0.965049564838
    Iter #4845568:  Learning rate = 0.000705:   Batch Loss = 0.348676, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.416527152061, Accuracy = 0.962962985039
    Iter #4849664:  Learning rate = 0.000705:   Batch Loss = 0.361484, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.389431506395, Accuracy = 0.970613777637
    Iter #4853760:  Learning rate = 0.000705:   Batch Loss = 0.374874, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.400253683329, Accuracy = 0.964701771736
    Iter #4857856:  Learning rate = 0.000705:   Batch Loss = 0.361736, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.406676054001, Accuracy = 0.964527904987
    Iter #4861952:  Learning rate = 0.000705:   Batch Loss = 0.353254, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.397927969694, Accuracy = 0.965397298336
    Iter #4866048:  Learning rate = 0.000705:   Batch Loss = 0.361508, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.382573753595, Accuracy = 0.969918251038
    Iter #4870144:  Learning rate = 0.000705:   Batch Loss = 0.334763, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.384058326483, Accuracy = 0.970613777637
    Iter #4874240:  Learning rate = 0.000705:   Batch Loss = 0.356046, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.38199159503, Accuracy = 0.971830964088
    Iter #4878336:  Learning rate = 0.000705:   Batch Loss = 0.339842, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.397634834051, Accuracy = 0.970439910889
    Iter #4882432:  Learning rate = 0.000705:   Batch Loss = 0.340163, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.396641165018, Accuracy = 0.970439910889
    Iter #4886528:  Learning rate = 0.000705:   Batch Loss = 0.344949, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.387535274029, Accuracy = 0.96957051754
    Iter #4890624:  Learning rate = 0.000705:   Batch Loss = 0.350899, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.384276032448, Accuracy = 0.970439910889
    Iter #4894720:  Learning rate = 0.000705:   Batch Loss = 0.348310, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.38647800684, Accuracy = 0.97026604414
    Iter #4898816:  Learning rate = 0.000705:   Batch Loss = 0.361770, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.393257528543, Accuracy = 0.967831671238
    Iter #4902912:  Learning rate = 0.000676:   Batch Loss = 0.325324, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.394369184971, Accuracy = 0.96887499094
    Iter #4907008:  Learning rate = 0.000676:   Batch Loss = 0.355340, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.384005665779, Accuracy = 0.970439910889
    Iter #4911104:  Learning rate = 0.000676:   Batch Loss = 0.340802, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.386634528637, Accuracy = 0.971135437489
    Iter #4915200:  Learning rate = 0.000676:   Batch Loss = 0.363966, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.427854955196, Accuracy = 0.954964339733
    Iter #4919296:  Learning rate = 0.000676:   Batch Loss = 0.370543, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.389042139053, Accuracy = 0.970613777637
    Iter #4923392:  Learning rate = 0.000676:   Batch Loss = 0.351435, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.408635437489, Accuracy = 0.962267458439
    Iter #4927488:  Learning rate = 0.000676:   Batch Loss = 0.348128, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.416035592556, Accuracy = 0.960876345634
    Iter #4931584:  Learning rate = 0.000676:   Batch Loss = 0.357216, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.410785317421, Accuracy = 0.963136851788
    Iter #4935680:  Learning rate = 0.000676:   Batch Loss = 0.371542, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.393167674541, Accuracy = 0.968005537987
    Iter #4939776:  Learning rate = 0.000676:   Batch Loss = 0.349106, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.394788444042, Accuracy = 0.965397298336
    Iter #4943872:  Learning rate = 0.000676:   Batch Loss = 0.372700, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.40663421154, Accuracy = 0.965223431587
    Iter #4947968:  Learning rate = 0.000676:   Batch Loss = 0.373899, Accuracy = 0.97265625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.393874108791, Accuracy = 0.968005537987
    Iter #4952064:  Learning rate = 0.000676:   Batch Loss = 0.349622, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.421486973763, Accuracy = 0.960354745388
    Iter #4956160:  Learning rate = 0.000676:   Batch Loss = 0.383652, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.387428879738, Accuracy = 0.970439910889
    Iter #4960256:  Learning rate = 0.000676:   Batch Loss = 0.353879, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.391238749027, Accuracy = 0.96957051754
    Iter #4964352:  Learning rate = 0.000676:   Batch Loss = 0.366261, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.386745154858, Accuracy = 0.97165709734
    Iter #4968448:  Learning rate = 0.000676:   Batch Loss = 0.364623, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.384565800428, Accuracy = 0.96957051754
    Iter #4972544:  Learning rate = 0.000676:   Batch Loss = 0.356671, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.388155460358, Accuracy = 0.969918251038
    Iter #4976640:  Learning rate = 0.000676:   Batch Loss = 0.330530, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.387001156807, Accuracy = 0.96817946434
    Iter #4980736:  Learning rate = 0.000676:   Batch Loss = 0.337780, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.390031099319, Accuracy = 0.968353331089
    Iter #4984832:  Learning rate = 0.000676:   Batch Loss = 0.359193, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.414683520794, Accuracy = 0.961050271988
    Iter #4988928:  Learning rate = 0.000676:   Batch Loss = 0.356248, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.395272552967, Accuracy = 0.96678841114
    Iter #4993024:  Learning rate = 0.000676:   Batch Loss = 0.346431, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.40156313777, Accuracy = 0.965571224689
    Iter #4997120:  Learning rate = 0.000676:   Batch Loss = 0.342668, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.387364894152, Accuracy = 0.96957051754
    Iter #5001216:  Learning rate = 0.000649:   Batch Loss = 0.366133, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.400590986013, Accuracy = 0.969222724438
    Iter #5005312:  Learning rate = 0.000649:   Batch Loss = 0.360529, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.387090802193, Accuracy = 0.969048857689
    Iter #5009408:  Learning rate = 0.000649:   Batch Loss = 0.342933, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.36987093091, Accuracy = 0.974613130093
    Iter #5013504:  Learning rate = 0.000649:   Batch Loss = 0.345010, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.380567193031, Accuracy = 0.972526490688
    Iter #5017600:  Learning rate = 0.000649:   Batch Loss = 0.329396, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.378225803375, Accuracy = 0.973222076893
    Iter #5021696:  Learning rate = 0.000649:   Batch Loss = 0.332632, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383419632912, Accuracy = 0.971483230591
    Iter #5025792:  Learning rate = 0.000649:   Batch Loss = 0.345696, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37768471241, Accuracy = 0.973395943642
    Iter #5029888:  Learning rate = 0.000649:   Batch Loss = 0.344991, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37528860569, Accuracy = 0.97426533699
    Iter #5033984:  Learning rate = 0.000649:   Batch Loss = 0.324924, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.406086623669, Accuracy = 0.968005537987
    Iter #5038080:  Learning rate = 0.000649:   Batch Loss = 0.341707, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.386436104774, Accuracy = 0.971309363842
    Iter #5042176:  Learning rate = 0.000649:   Batch Loss = 0.345751, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.398112595081, Accuracy = 0.966962277889
    Iter #5046272:  Learning rate = 0.000649:   Batch Loss = 0.363361, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.421673029661, Accuracy = 0.963832378387
    Iter #5050368:  Learning rate = 0.000649:   Batch Loss = 0.359449, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.389439195395, Accuracy = 0.968353331089
    Iter #5054464:  Learning rate = 0.000649:   Batch Loss = 0.345585, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39520868659, Accuracy = 0.96678841114
    Iter #5058560:  Learning rate = 0.000649:   Batch Loss = 0.344879, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.376760601997, Accuracy = 0.973048150539
    Iter #5062656:  Learning rate = 0.000649:   Batch Loss = 0.343013, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.3727863729, Accuracy = 0.974786996841
    Iter #5066752:  Learning rate = 0.000649:   Batch Loss = 0.332491, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.368112444878, Accuracy = 0.97635191679
    Iter #5070848:  Learning rate = 0.000649:   Batch Loss = 0.318938, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.377164721489, Accuracy = 0.972004890442
    Iter #5074944:  Learning rate = 0.000649:   Batch Loss = 0.317795, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37548905611, Accuracy = 0.974439203739
    Iter #5079040:  Learning rate = 0.000649:   Batch Loss = 0.354649, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.387276470661, Accuracy = 0.97165709734
    Iter #5083136:  Learning rate = 0.000649:   Batch Loss = 0.371335, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.376110672951, Accuracy = 0.97165709734
    Iter #5087232:  Learning rate = 0.000649:   Batch Loss = 0.345937, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.384544610977, Accuracy = 0.970439910889
    Iter #5091328:  Learning rate = 0.000649:   Batch Loss = 0.350064, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.371904194355, Accuracy = 0.97426533699
    Iter #5095424:  Learning rate = 0.000649:   Batch Loss = 0.323824, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37808021903, Accuracy = 0.972700417042
    Iter #5099520:  Learning rate = 0.000649:   Batch Loss = 0.350105, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.392836362123, Accuracy = 0.96748393774
    Iter #5103616:  Learning rate = 0.000623:   Batch Loss = 0.345545, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.400825738907, Accuracy = 0.962441325188
    Iter #5107712:  Learning rate = 0.000623:   Batch Loss = 0.346807, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383228778839, Accuracy = 0.971483230591
    Iter #5111808:  Learning rate = 0.000623:   Batch Loss = 0.325948, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.375322014093, Accuracy = 0.973743677139
    Iter #5115904:  Learning rate = 0.000623:   Batch Loss = 0.335010, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37659317255, Accuracy = 0.974091470242
    Iter #5120000:  Learning rate = 0.000623:   Batch Loss = 0.339488, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.369529485703, Accuracy = 0.974786996841
    Iter #5124096:  Learning rate = 0.000623:   Batch Loss = 0.340676, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39725124836, Accuracy = 0.969744384289
    Iter #5128192:  Learning rate = 0.000623:   Batch Loss = 0.334539, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.38592261076, Accuracy = 0.96817946434
    Iter #5132288:  Learning rate = 0.000623:   Batch Loss = 0.346525, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.393477976322, Accuracy = 0.968353331089
    Iter #5136384:  Learning rate = 0.000623:   Batch Loss = 0.332395, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.375483483076, Accuracy = 0.971830964088
    Iter #5140480:  Learning rate = 0.000623:   Batch Loss = 0.336920, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.378014743328, Accuracy = 0.973743677139
    Iter #5144576:  Learning rate = 0.000623:   Batch Loss = 0.331715, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.373218357563, Accuracy = 0.97426533699
    Iter #5148672:  Learning rate = 0.000623:   Batch Loss = 0.337925, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.375514984131, Accuracy = 0.972874283791
    Iter #5152768:  Learning rate = 0.000623:   Batch Loss = 0.340218, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.375407159328, Accuracy = 0.97096157074
    Iter #5156864:  Learning rate = 0.000623:   Batch Loss = 0.340758, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.406250834465, Accuracy = 0.964701771736
    Iter #5160960:  Learning rate = 0.000623:   Batch Loss = 0.369073, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.409626811743, Accuracy = 0.96817946434
    Iter #5165056:  Learning rate = 0.000623:   Batch Loss = 0.331495, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383088260889, Accuracy = 0.97026604414
    Iter #5169152:  Learning rate = 0.000623:   Batch Loss = 0.347029, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.395057290792, Accuracy = 0.966266751289
    Iter #5173248:  Learning rate = 0.000623:   Batch Loss = 0.341097, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.379887461662, Accuracy = 0.970787703991
    Iter #5177344:  Learning rate = 0.000623:   Batch Loss = 0.346309, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.406871646643, Accuracy = 0.964006245136
    Iter #5181440:  Learning rate = 0.000623:   Batch Loss = 0.360852, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.422684401274, Accuracy = 0.965397298336
    Iter #5185536:  Learning rate = 0.000623:   Batch Loss = 0.361746, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.399645537138, Accuracy = 0.966962277889
    Iter #5189632:  Learning rate = 0.000623:   Batch Loss = 0.342881, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.38069280982, Accuracy = 0.974786996841
    Iter #5193728:  Learning rate = 0.000623:   Batch Loss = 0.332559, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.393127262592, Accuracy = 0.970787703991
    Iter #5197824:  Learning rate = 0.000623:   Batch Loss = 0.343202, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.399951845407, Accuracy = 0.96748393774
    Iter #5201920:  Learning rate = 0.000599:   Batch Loss = 0.343396, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.386085599661, Accuracy = 0.96957051754
    Iter #5206016:  Learning rate = 0.000599:   Batch Loss = 0.333606, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.387722373009, Accuracy = 0.969396650791
    Iter #5210112:  Learning rate = 0.000599:   Batch Loss = 0.336054, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.378239512444, Accuracy = 0.970439910889
    Iter #5214208:  Learning rate = 0.000599:   Batch Loss = 0.326877, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.381405234337, Accuracy = 0.96887499094
    Iter #5218304:  Learning rate = 0.000599:   Batch Loss = 0.332346, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.382972747087, Accuracy = 0.96957051754
    Iter #5222400:  Learning rate = 0.000599:   Batch Loss = 0.318550, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.386457294226, Accuracy = 0.97026604414
    Iter #5226496:  Learning rate = 0.000599:   Batch Loss = 0.346592, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.387272417545, Accuracy = 0.968701124191
    Iter #5230592:  Learning rate = 0.000599:   Batch Loss = 0.329421, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.376838505268, Accuracy = 0.971135437489
    Iter #5234688:  Learning rate = 0.000599:   Batch Loss = 0.345968, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.382454752922, Accuracy = 0.97026604414
    Iter #5238784:  Learning rate = 0.000599:   Batch Loss = 0.340974, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.381181776524, Accuracy = 0.971309363842
    Iter #5242880:  Learning rate = 0.000599:   Batch Loss = 0.352359, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383828818798, Accuracy = 0.967831671238
    Iter #5246976:  Learning rate = 0.000599:   Batch Loss = 0.319820, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37679579854, Accuracy = 0.970613777637
    Iter #5251072:  Learning rate = 0.000599:   Batch Loss = 0.346723, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.388952344656, Accuracy = 0.970613777637
    Iter #5255168:  Learning rate = 0.000599:   Batch Loss = 0.352851, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.379122972488, Accuracy = 0.969396650791
    Iter #5259264:  Learning rate = 0.000599:   Batch Loss = 0.322056, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.382319927216, Accuracy = 0.968527197838
    Iter #5263360:  Learning rate = 0.000599:   Batch Loss = 0.351643, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.382350981236, Accuracy = 0.968527197838
    Iter #5267456:  Learning rate = 0.000599:   Batch Loss = 0.341824, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383511871099, Accuracy = 0.96957051754
    Iter #5271552:  Learning rate = 0.000599:   Batch Loss = 0.339837, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383429944515, Accuracy = 0.970092177391
    Iter #5275648:  Learning rate = 0.000599:   Batch Loss = 0.328782, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37895783782, Accuracy = 0.972526490688
    Iter #5279744:  Learning rate = 0.000599:   Batch Loss = 0.333618, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.381318867207, Accuracy = 0.972700417042
    Iter #5283840:  Learning rate = 0.000599:   Batch Loss = 0.351905, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.376354157925, Accuracy = 0.973048150539
    Iter #5287936:  Learning rate = 0.000599:   Batch Loss = 0.346236, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.380794346333, Accuracy = 0.970439910889
    Iter #5292032:  Learning rate = 0.000599:   Batch Loss = 0.330731, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383010149002, Accuracy = 0.972178757191
    Iter #5296128:  Learning rate = 0.000599:   Batch Loss = 0.339785, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383608520031, Accuracy = 0.96957051754
    Iter #5300224:  Learning rate = 0.000575:   Batch Loss = 0.327015, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.3787355721, Accuracy = 0.969048857689
    Iter #5304320:  Learning rate = 0.000575:   Batch Loss = 0.340134, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.372104763985, Accuracy = 0.971483230591
    Iter #5308416:  Learning rate = 0.000575:   Batch Loss = 0.330817, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383251339197, Accuracy = 0.969744384289
    Iter #5312512:  Learning rate = 0.000575:   Batch Loss = 0.333770, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.38296893239, Accuracy = 0.969744384289
    Iter #5316608:  Learning rate = 0.000575:   Batch Loss = 0.344656, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.380849063396, Accuracy = 0.968701124191
    Iter #5320704:  Learning rate = 0.000575:   Batch Loss = 0.326164, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.375900864601, Accuracy = 0.971309363842
    Iter #5324800:  Learning rate = 0.000575:   Batch Loss = 0.342287, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.377291202545, Accuracy = 0.97096157074
    Iter #5328896:  Learning rate = 0.000575:   Batch Loss = 0.354807, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.386448979378, Accuracy = 0.97096157074
    Iter #5332992:  Learning rate = 0.000575:   Batch Loss = 0.338765, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383861869574, Accuracy = 0.971483230591
    Iter #5337088:  Learning rate = 0.000575:   Batch Loss = 0.325852, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.380094081163, Accuracy = 0.969744384289
    Iter #5341184:  Learning rate = 0.000575:   Batch Loss = 0.325037, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37989372015, Accuracy = 0.973395943642
    Iter #5345280:  Learning rate = 0.000575:   Batch Loss = 0.340206, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.378229618073, Accuracy = 0.97235262394
    Iter #5349376:  Learning rate = 0.000575:   Batch Loss = 0.319498, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.375350505114, Accuracy = 0.972178757191
    Iter #5353472:  Learning rate = 0.000575:   Batch Loss = 0.319397, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.365996837616, Accuracy = 0.975134730339
    Iter #5357568:  Learning rate = 0.000575:   Batch Loss = 0.318663, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.378387570381, Accuracy = 0.971309363842
    Iter #5361664:  Learning rate = 0.000575:   Batch Loss = 0.331896, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.366738885641, Accuracy = 0.974786996841
    Iter #5365760:  Learning rate = 0.000575:   Batch Loss = 0.327977, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.371043086052, Accuracy = 0.972700417042
    Iter #5369856:  Learning rate = 0.000575:   Batch Loss = 0.342369, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37689819932, Accuracy = 0.969744384289
    Iter #5373952:  Learning rate = 0.000575:   Batch Loss = 0.331347, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.411539077759, Accuracy = 0.966614484787
    Iter #5378048:  Learning rate = 0.000575:   Batch Loss = 0.342615, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.38421356678, Accuracy = 0.970787703991
    Iter #5382144:  Learning rate = 0.000575:   Batch Loss = 0.343475, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.384953588247, Accuracy = 0.970092177391
    Iter #5386240:  Learning rate = 0.000575:   Batch Loss = 0.340480, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.366874843836, Accuracy = 0.974613130093
    Iter #5390336:  Learning rate = 0.000575:   Batch Loss = 0.341209, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39433914423, Accuracy = 0.964354038239
    Iter #5394432:  Learning rate = 0.000575:   Batch Loss = 0.366593, Accuracy = 0.970703125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.395503342152, Accuracy = 0.965571224689
    Iter #5398528:  Learning rate = 0.000575:   Batch Loss = 0.316462, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.378358930349, Accuracy = 0.972700417042
    Iter #5402624:  Learning rate = 0.000552:   Batch Loss = 0.349459, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.375609576702, Accuracy = 0.972874283791
    Iter #5406720:  Learning rate = 0.000552:   Batch Loss = 0.328208, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39241823554, Accuracy = 0.967310011387
    Iter #5410816:  Learning rate = 0.000552:   Batch Loss = 0.331927, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.381153851748, Accuracy = 0.969396650791
    Iter #5414912:  Learning rate = 0.000552:   Batch Loss = 0.322698, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.373597741127, Accuracy = 0.972178757191
    Iter #5419008:  Learning rate = 0.000552:   Batch Loss = 0.341997, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.401053190231, Accuracy = 0.97026604414
    Iter #5423104:  Learning rate = 0.000552:   Batch Loss = 0.320295, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.392607450485, Accuracy = 0.965571224689
    Iter #5427200:  Learning rate = 0.000552:   Batch Loss = 0.337595, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.391821086407, Accuracy = 0.965918958187
    Iter #5431296:  Learning rate = 0.000552:   Batch Loss = 0.335022, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.386797130108, Accuracy = 0.968527197838
    Iter #5435392:  Learning rate = 0.000552:   Batch Loss = 0.323082, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.371743112803, Accuracy = 0.972526490688
    Iter #5439488:  Learning rate = 0.000552:   Batch Loss = 0.341336, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.386829137802, Accuracy = 0.968353331089
    Iter #5443584:  Learning rate = 0.000552:   Batch Loss = 0.330322, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37084710598, Accuracy = 0.972700417042
    Iter #5447680:  Learning rate = 0.000552:   Batch Loss = 0.331792, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.36312648654, Accuracy = 0.974613130093
    Iter #5451776:  Learning rate = 0.000552:   Batch Loss = 0.313407, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.373301208019, Accuracy = 0.971483230591
    Iter #5455872:  Learning rate = 0.000552:   Batch Loss = 0.323213, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.366744071245, Accuracy = 0.975830316544
    Iter #5459968:  Learning rate = 0.000552:   Batch Loss = 0.326551, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383008360863, Accuracy = 0.969918251038
    Iter #5464064:  Learning rate = 0.000552:   Batch Loss = 0.327162, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.384638667107, Accuracy = 0.968527197838
    Iter #5468160:  Learning rate = 0.000552:   Batch Loss = 0.332171, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.3662802279, Accuracy = 0.975308656693
    Iter #5472256:  Learning rate = 0.000552:   Batch Loss = 0.340883, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.391387045383, Accuracy = 0.968353331089
    Iter #5476352:  Learning rate = 0.000552:   Batch Loss = 0.323536, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.388517677784, Accuracy = 0.969048857689
    Iter #5480448:  Learning rate = 0.000552:   Batch Loss = 0.331331, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.387901365757, Accuracy = 0.966614484787
    Iter #5484544:  Learning rate = 0.000552:   Batch Loss = 0.342199, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.382972002029, Accuracy = 0.97026604414
    Iter #5488640:  Learning rate = 0.000552:   Batch Loss = 0.328001, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.370928049088, Accuracy = 0.97165709734
    Iter #5492736:  Learning rate = 0.000552:   Batch Loss = 0.331922, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.36404004693, Accuracy = 0.974786996841
    Iter #5496832:  Learning rate = 0.000552:   Batch Loss = 0.323742, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.36474403739, Accuracy = 0.97496086359
    Iter #5500928:  Learning rate = 0.000530:   Batch Loss = 0.312698, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.380880653858, Accuracy = 0.971309363842
    Iter #5505024:  Learning rate = 0.000530:   Batch Loss = 0.328031, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.369752764702, Accuracy = 0.972874283791
    Iter #5509120:  Learning rate = 0.000530:   Batch Loss = 0.323791, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.380338579416, Accuracy = 0.97026604414
    Iter #5513216:  Learning rate = 0.000530:   Batch Loss = 0.339003, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.381508857012, Accuracy = 0.968353331089
    Iter #5517312:  Learning rate = 0.000530:   Batch Loss = 0.320598, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.373715102673, Accuracy = 0.972178757191
    Iter #5521408:  Learning rate = 0.000530:   Batch Loss = 0.357435, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37437030673, Accuracy = 0.97165709734
    Iter #5525504:  Learning rate = 0.000530:   Batch Loss = 0.347800, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.368860661983, Accuracy = 0.975482523441
    Iter #5529600:  Learning rate = 0.000530:   Batch Loss = 0.318124, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.374947100878, Accuracy = 0.971135437489
    Iter #5533696:  Learning rate = 0.000530:   Batch Loss = 0.341210, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.374449551105, Accuracy = 0.971135437489
    Iter #5537792:  Learning rate = 0.000530:   Batch Loss = 0.330624, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383373975754, Accuracy = 0.969396650791
    Iter #5541888:  Learning rate = 0.000530:   Batch Loss = 0.323402, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.383772134781, Accuracy = 0.970092177391
    Iter #5545984:  Learning rate = 0.000530:   Batch Loss = 0.350424, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.382238298655, Accuracy = 0.971135437489
    Iter #5550080:  Learning rate = 0.000530:   Batch Loss = 0.355317, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.384718298912, Accuracy = 0.973222076893
    Iter #5554176:  Learning rate = 0.000530:   Batch Loss = 0.326409, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.376055181026, Accuracy = 0.970439910889
    Iter #5558272:  Learning rate = 0.000530:   Batch Loss = 0.344585, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.368128240108, Accuracy = 0.97496086359
    Iter #5562368:  Learning rate = 0.000530:   Batch Loss = 0.352497, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.369182139635, Accuracy = 0.972526490688
    Iter #5566464:  Learning rate = 0.000530:   Batch Loss = 0.322049, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.368973910809, Accuracy = 0.973048150539
    Iter #5570560:  Learning rate = 0.000530:   Batch Loss = 0.329838, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.361681938171, Accuracy = 0.975482523441
    Iter #5574656:  Learning rate = 0.000530:   Batch Loss = 0.329082, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.376102119684, Accuracy = 0.972874283791
    Iter #5578752:  Learning rate = 0.000530:   Batch Loss = 0.324804, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.364635616541, Accuracy = 0.97565639019
    Iter #5582848:  Learning rate = 0.000530:   Batch Loss = 0.321036, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.371152818203, Accuracy = 0.972004890442
    Iter #5586944:  Learning rate = 0.000530:   Batch Loss = 0.313143, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.359475165606, Accuracy = 0.976699709892
    Iter #5591040:  Learning rate = 0.000530:   Batch Loss = 0.324481, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.364450961351, Accuracy = 0.974786996841
    Iter #5595136:  Learning rate = 0.000530:   Batch Loss = 0.320057, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.364053547382, Accuracy = 0.975308656693
    Iter #5599232:  Learning rate = 0.000530:   Batch Loss = 0.320376, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.364260673523, Accuracy = 0.974613130093
    Iter #5603328:  Learning rate = 0.000508:   Batch Loss = 0.340782, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.366690605879, Accuracy = 0.974439203739
    Iter #5607424:  Learning rate = 0.000508:   Batch Loss = 0.322586, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.373117357492, Accuracy = 0.971135437489
    Iter #5611520:  Learning rate = 0.000508:   Batch Loss = 0.337097, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.369542986155, Accuracy = 0.973743677139
    Iter #5615616:  Learning rate = 0.000508:   Batch Loss = 0.323848, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.368138939142, Accuracy = 0.97496086359
    Iter #5619712:  Learning rate = 0.000508:   Batch Loss = 0.305801, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.374329656363, Accuracy = 0.975134730339
    Iter #5623808:  Learning rate = 0.000508:   Batch Loss = 0.319518, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.364336848259, Accuracy = 0.975134730339
    Iter #5627904:  Learning rate = 0.000508:   Batch Loss = 0.321029, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.373994916677, Accuracy = 0.971483230591
    Iter #5632000:  Learning rate = 0.000508:   Batch Loss = 0.322108, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.389405965805, Accuracy = 0.967831671238
    Iter #5636096:  Learning rate = 0.000508:   Batch Loss = 0.332379, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.371570318937, Accuracy = 0.972874283791
    Iter #5640192:  Learning rate = 0.000508:   Batch Loss = 0.307947, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.374131321907, Accuracy = 0.970613777637
    Iter #5644288:  Learning rate = 0.000508:   Batch Loss = 0.343002, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.367169737816, Accuracy = 0.973743677139
    Iter #5648384:  Learning rate = 0.000508:   Batch Loss = 0.326995, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.366967201233, Accuracy = 0.974091470242
    Iter #5652480:  Learning rate = 0.000508:   Batch Loss = 0.344873, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.370021879673, Accuracy = 0.971483230591
    Iter #5656576:  Learning rate = 0.000508:   Batch Loss = 0.321305, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.361325502396, Accuracy = 0.976699709892
    Iter #5660672:  Learning rate = 0.000508:   Batch Loss = 0.318712, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.363937824965, Accuracy = 0.97496086359
    Iter #5664768:  Learning rate = 0.000508:   Batch Loss = 0.314225, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.368672132492, Accuracy = 0.973743677139
    Iter #5668864:  Learning rate = 0.000508:   Batch Loss = 0.318831, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.373628437519, Accuracy = 0.974439203739
    Iter #5672960:  Learning rate = 0.000508:   Batch Loss = 0.324638, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.374691963196, Accuracy = 0.972004890442
    Iter #5677056:  Learning rate = 0.000508:   Batch Loss = 0.343378, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.390074253082, Accuracy = 0.968701124191
    Iter #5681152:  Learning rate = 0.000508:   Batch Loss = 0.334159, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.380882829428, Accuracy = 0.96957051754
    Iter #5685248:  Learning rate = 0.000508:   Batch Loss = 0.342935, Accuracy = 0.9765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.369424194098, Accuracy = 0.971483230591
    Iter #5689344:  Learning rate = 0.000508:   Batch Loss = 0.343056, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.382444143295, Accuracy = 0.968527197838
    Iter #5693440:  Learning rate = 0.000508:   Batch Loss = 0.338274, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.372408658266, Accuracy = 0.973048150539
    Iter #5697536:  Learning rate = 0.000508:   Batch Loss = 0.320191, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37532427907, Accuracy = 0.97165709734
    Iter #5701632:  Learning rate = 0.000488:   Batch Loss = 0.315311, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.411106735468, Accuracy = 0.966962277889
    Iter #5705728:  Learning rate = 0.000488:   Batch Loss = 0.317443, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.388553798199, Accuracy = 0.969048857689
    Iter #5709824:  Learning rate = 0.000488:   Batch Loss = 0.357407, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.397284448147, Accuracy = 0.964701771736
    Iter #5713920:  Learning rate = 0.000488:   Batch Loss = 0.336754, Accuracy = 0.978515625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.367325127125, Accuracy = 0.97356981039
    Iter #5718016:  Learning rate = 0.000488:   Batch Loss = 0.339635, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.370979011059, Accuracy = 0.970092177391
    Iter #5722112:  Learning rate = 0.000488:   Batch Loss = 0.326723, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.358890920877, Accuracy = 0.975482523441
    Iter #5726208:  Learning rate = 0.000488:   Batch Loss = 0.322337, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.357396155596, Accuracy = 0.97635191679
    Iter #5730304:  Learning rate = 0.000488:   Batch Loss = 0.319836, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.363710701466, Accuracy = 0.974439203739
    Iter #5734400:  Learning rate = 0.000488:   Batch Loss = 0.306076, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.372102409601, Accuracy = 0.97356981039
    Iter #5738496:  Learning rate = 0.000488:   Batch Loss = 0.328987, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.358447313309, Accuracy = 0.974786996841
    Iter #5742592:  Learning rate = 0.000488:   Batch Loss = 0.362270, Accuracy = 0.974609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.371372789145, Accuracy = 0.97165709734
    Iter #5746688:  Learning rate = 0.000488:   Batch Loss = 0.313939, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.36424306035, Accuracy = 0.974091470242
    Iter #5750784:  Learning rate = 0.000488:   Batch Loss = 0.330367, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.359485119581, Accuracy = 0.974786996841
    Iter #5754880:  Learning rate = 0.000488:   Batch Loss = 0.309645, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.360657125711, Accuracy = 0.97496086359
    Iter #5758976:  Learning rate = 0.000488:   Batch Loss = 0.314419, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.363823264837, Accuracy = 0.975308656693
    Iter #5763072:  Learning rate = 0.000488:   Batch Loss = 0.322274, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.360407233238, Accuracy = 0.976525843143
    Iter #5767168:  Learning rate = 0.000488:   Batch Loss = 0.322847, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.352842479944, Accuracy = 0.978612422943
    Iter #5771264:  Learning rate = 0.000488:   Batch Loss = 0.311752, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.367961317301, Accuracy = 0.975134730339
    Iter #5775360:  Learning rate = 0.000488:   Batch Loss = 0.330456, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.3642359972, Accuracy = 0.973917603493
    Iter #5779456:  Learning rate = 0.000488:   Batch Loss = 0.323319, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39679852128, Accuracy = 0.96817946434
    Iter #5783552:  Learning rate = 0.000488:   Batch Loss = 0.307056, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.378022938967, Accuracy = 0.972178757191
    Iter #5787648:  Learning rate = 0.000488:   Batch Loss = 0.327037, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.367201566696, Accuracy = 0.972874283791
    Iter #5791744:  Learning rate = 0.000488:   Batch Loss = 0.318969, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.367626041174, Accuracy = 0.973048150539
    Iter #5795840:  Learning rate = 0.000488:   Batch Loss = 0.320270, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.354633957148, Accuracy = 0.978090763092
    Iter #5799936:  Learning rate = 0.000488:   Batch Loss = 0.307528, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.36737844348, Accuracy = 0.97165709734
    Iter #5804032:  Learning rate = 0.000468:   Batch Loss = 0.314872, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.367941439152, Accuracy = 0.975134730339
    Iter #5808128:  Learning rate = 0.000468:   Batch Loss = 0.306551, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.36304962635, Accuracy = 0.97426533699
    Iter #5812224:  Learning rate = 0.000468:   Batch Loss = 0.324173, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.358788400888, Accuracy = 0.973743677139
    Iter #5816320:  Learning rate = 0.000468:   Batch Loss = 0.326890, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.371667742729, Accuracy = 0.97026604414
    Iter #5820416:  Learning rate = 0.000468:   Batch Loss = 0.323844, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.361923098564, Accuracy = 0.97356981039
    Iter #5824512:  Learning rate = 0.000468:   Batch Loss = 0.320361, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.363903999329, Accuracy = 0.975308656693
    Iter #5828608:  Learning rate = 0.000468:   Batch Loss = 0.330237, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.370599806309, Accuracy = 0.973395943642
    Iter #5832704:  Learning rate = 0.000468:   Batch Loss = 0.316018, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.36936506629, Accuracy = 0.97235262394
    Iter #5836800:  Learning rate = 0.000468:   Batch Loss = 0.318306, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.374375283718, Accuracy = 0.971309363842
    Iter #5840896:  Learning rate = 0.000468:   Batch Loss = 0.318010, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.358582913876, Accuracy = 0.975830316544
    Iter #5844992:  Learning rate = 0.000468:   Batch Loss = 0.324378, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.359611332417, Accuracy = 0.975482523441
    Iter #5849088:  Learning rate = 0.000468:   Batch Loss = 0.329125, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.361673563719, Accuracy = 0.97356981039
    Iter #5853184:  Learning rate = 0.000468:   Batch Loss = 0.318978, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.354823589325, Accuracy = 0.977221369743
    Iter #5857280:  Learning rate = 0.000468:   Batch Loss = 0.332345, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.363248378038, Accuracy = 0.975308656693
    Iter #5861376:  Learning rate = 0.000468:   Batch Loss = 0.319686, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.359315276146, Accuracy = 0.97635191679
    Iter #5865472:  Learning rate = 0.000468:   Batch Loss = 0.322597, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.351805329323, Accuracy = 0.978438556194
    Iter #5869568:  Learning rate = 0.000468:   Batch Loss = 0.318172, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.364448010921, Accuracy = 0.974613130093
    Iter #5873664:  Learning rate = 0.000468:   Batch Loss = 0.326831, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.357903718948, Accuracy = 0.977395236492
    Iter #5877760:  Learning rate = 0.000468:   Batch Loss = 0.347076, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.369163572788, Accuracy = 0.972700417042
    Iter #5881856:  Learning rate = 0.000468:   Batch Loss = 0.338433, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.389684587717, Accuracy = 0.967310011387
    Iter #5885952:  Learning rate = 0.000468:   Batch Loss = 0.520088, Accuracy = 0.935546875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.449081748724, Accuracy = 0.952182233334
    Iter #5890048:  Learning rate = 0.000468:   Batch Loss = 0.378159, Accuracy = 0.962890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.411349892616, Accuracy = 0.960354745388
    Iter #5894144:  Learning rate = 0.000468:   Batch Loss = 0.345796, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.394327998161, Accuracy = 0.965223431587
    Iter #5898240:  Learning rate = 0.000468:   Batch Loss = 0.327324, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.379918336868, Accuracy = 0.96748393774
    Iter #5902336:  Learning rate = 0.000450:   Batch Loss = 0.344742, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.38090467453, Accuracy = 0.968527197838
    Iter #5906432:  Learning rate = 0.000450:   Batch Loss = 0.339758, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.365278631449, Accuracy = 0.973222076893
    Iter #5910528:  Learning rate = 0.000450:   Batch Loss = 0.343870, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.357962191105, Accuracy = 0.97356981039
    Iter #5914624:  Learning rate = 0.000450:   Batch Loss = 0.317633, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.362642586231, Accuracy = 0.972526490688
    Iter #5918720:  Learning rate = 0.000450:   Batch Loss = 0.317178, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.359465420246, Accuracy = 0.975830316544
    Iter #5922816:  Learning rate = 0.000450:   Batch Loss = 0.324906, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.350827217102, Accuracy = 0.976004183292
    Iter #5926912:  Learning rate = 0.000450:   Batch Loss = 0.319240, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.353486865759, Accuracy = 0.97704744339
    Iter #5931008:  Learning rate = 0.000450:   Batch Loss = 0.328960, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.355142712593, Accuracy = 0.97635191679
    Iter #5935104:  Learning rate = 0.000450:   Batch Loss = 0.333612, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.364989459515, Accuracy = 0.975134730339
    Iter #5939200:  Learning rate = 0.000450:   Batch Loss = 0.324607, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37278008461, Accuracy = 0.97235262394
    Iter #5943296:  Learning rate = 0.000450:   Batch Loss = 0.305335, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.354783505201, Accuracy = 0.976525843143
    Iter #5947392:  Learning rate = 0.000450:   Batch Loss = 0.331215, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.362099409103, Accuracy = 0.975482523441
    Iter #5951488:  Learning rate = 0.000450:   Batch Loss = 0.320646, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.35679101944, Accuracy = 0.976525843143
    Iter #5955584:  Learning rate = 0.000450:   Batch Loss = 0.314236, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.360171794891, Accuracy = 0.975830316544
    Iter #5959680:  Learning rate = 0.000450:   Batch Loss = 0.307429, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.358300149441, Accuracy = 0.974091470242
    Iter #5963776:  Learning rate = 0.000450:   Batch Loss = 0.334258, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.348255336285, Accuracy = 0.979134082794
    Iter #5967872:  Learning rate = 0.000450:   Batch Loss = 0.310869, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.350438773632, Accuracy = 0.977221369743
    Iter #5971968:  Learning rate = 0.000450:   Batch Loss = 0.309786, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.356820821762, Accuracy = 0.97704744339
    Iter #5976064:  Learning rate = 0.000450:   Batch Loss = 0.302661, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.35641798377, Accuracy = 0.97496086359
    Iter #5980160:  Learning rate = 0.000450:   Batch Loss = 0.317531, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.360315024853, Accuracy = 0.97496086359
    Iter #5984256:  Learning rate = 0.000450:   Batch Loss = 0.321453, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.36546087265, Accuracy = 0.973048150539
    Iter #5988352:  Learning rate = 0.000450:   Batch Loss = 0.317650, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.35184442997, Accuracy = 0.976525843143
    Iter #5992448:  Learning rate = 0.000450:   Batch Loss = 0.319612, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.365522474051, Accuracy = 0.975308656693
    Iter #5996544:  Learning rate = 0.000450:   Batch Loss = 0.341582, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.367043435574, Accuracy = 0.976699709892
    Iter #6000640:  Learning rate = 0.000432:   Batch Loss = 0.340993, Accuracy = 0.98046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.369931697845, Accuracy = 0.97356981039
    Iter #6004736:  Learning rate = 0.000432:   Batch Loss = 0.332722, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.37255948782, Accuracy = 0.97235262394
    Iter #6008832:  Learning rate = 0.000432:   Batch Loss = 0.327019, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.364458590746, Accuracy = 0.973395943642
    Iter #6012928:  Learning rate = 0.000432:   Batch Loss = 0.331116, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.369191467762, Accuracy = 0.970787703991
    Iter #6017024:  Learning rate = 0.000432:   Batch Loss = 0.318375, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.355907797813, Accuracy = 0.97635191679
    Iter #6021120:  Learning rate = 0.000432:   Batch Loss = 0.313482, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.358246147633, Accuracy = 0.97426533699
    Iter #6025216:  Learning rate = 0.000432:   Batch Loss = 0.339562, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.345303475857, Accuracy = 0.977743029594
    Iter #6029312:  Learning rate = 0.000432:   Batch Loss = 0.314014, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.350936114788, Accuracy = 0.978090763092
    Iter #6033408:  Learning rate = 0.000432:   Batch Loss = 0.318218, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.347200393677, Accuracy = 0.980872869492
    Iter #6037504:  Learning rate = 0.000432:   Batch Loss = 0.329746, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.348057031631, Accuracy = 0.978438556194
    Iter #6041600:  Learning rate = 0.000432:   Batch Loss = 0.314875, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.3676417768, Accuracy = 0.973222076893
    Iter #6045696:  Learning rate = 0.000432:   Batch Loss = 0.312536, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.355455428362, Accuracy = 0.978090763092
    Iter #6049792:  Learning rate = 0.000432:   Batch Loss = 0.331789, Accuracy = 0.986328125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.385225832462, Accuracy = 0.967657804489
    Iter #6053888:  Learning rate = 0.000432:   Batch Loss = 0.340125, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.368925452232, Accuracy = 0.970613777637
    Iter #6057984:  Learning rate = 0.000432:   Batch Loss = 0.324325, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.352999418974, Accuracy = 0.975134730339
    Iter #6062080:  Learning rate = 0.000432:   Batch Loss = 0.311135, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.403063803911, Accuracy = 0.962615191936
    Iter #6066176:  Learning rate = 0.000432:   Batch Loss = 0.304270, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.35596460104, Accuracy = 0.975482523441
    Iter #6070272:  Learning rate = 0.000432:   Batch Loss = 0.323294, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.353524297476, Accuracy = 0.977916896343
    Iter #6074368:  Learning rate = 0.000432:   Batch Loss = 0.307359, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.359051376581, Accuracy = 0.97426533699
    Iter #6078464:  Learning rate = 0.000432:   Batch Loss = 0.325550, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.352133750916, Accuracy = 0.977916896343
    Iter #6082560:  Learning rate = 0.000432:   Batch Loss = 0.316612, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.34970036149, Accuracy = 0.977569103241
    Iter #6086656:  Learning rate = 0.000432:   Batch Loss = 0.309421, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.358559757471, Accuracy = 0.976178050041
    Iter #6090752:  Learning rate = 0.000432:   Batch Loss = 0.309581, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.35110321641, Accuracy = 0.978612422943
    Iter #6094848:  Learning rate = 0.000432:   Batch Loss = 0.313317, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.347121149302, Accuracy = 0.980177342892
    Iter #6098944:  Learning rate = 0.000432:   Batch Loss = 0.320677, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.346656054258, Accuracy = 0.980699002743
    Iter #6103040:  Learning rate = 0.000414:   Batch Loss = 0.326251, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.355487883091, Accuracy = 0.97704744339
    Iter #6107136:  Learning rate = 0.000414:   Batch Loss = 0.349790, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.39609092474, Accuracy = 0.971135437489
    Iter #6111232:  Learning rate = 0.000414:   Batch Loss = 0.333831, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.395102053881, Accuracy = 0.967136144638
    Iter #6115328:  Learning rate = 0.000414:   Batch Loss = 0.306874, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.370206534863, Accuracy = 0.971135437489
    Iter #6119424:  Learning rate = 0.000414:   Batch Loss = 0.331348, Accuracy = 0.98828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.367458343506, Accuracy = 0.971309363842
    Iter #6123520:  Learning rate = 0.000414:   Batch Loss = 0.311378, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.358900487423, Accuracy = 0.973222076893
    Iter #6127616:  Learning rate = 0.000414:   Batch Loss = 0.317481, Accuracy = 0.9921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.363278210163, Accuracy = 0.974786996841
    Iter #6131712:  Learning rate = 0.000414:   Batch Loss = 0.315386, Accuracy = 0.994140625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.372086107731, Accuracy = 0.96957051754
    Iter #6135808:  Learning rate = 0.000414:   Batch Loss = 0.300659, Accuracy = 0.998046875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.368809729815, Accuracy = 0.972700417042
    Iter #6139904:  Learning rate = 0.000414:   Batch Loss = 0.336095, Accuracy = 0.982421875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.360386192799, Accuracy = 0.975308656693
    Iter #6144000:  Learning rate = 0.000414:   Batch Loss = 0.304367, Accuracy = 0.99609375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.373414367437, Accuracy = 0.97026604414
    Iter #6148096:  Learning rate = 0.000414:   Batch Loss = 0.323380, Accuracy = 0.990234375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.354190647602, Accuracy = 0.974091470242


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
print len(test_accuracies)
print len(train_accuracies)

plt.title("Training session's Accuracy over Iterations")
plt.legend(loc='lower right', shadow=True)
plt.ylabel('Training Accuracy')
plt.xlabel('Training Iteration')

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
#
#X_val_path = DATASET_PATH + "X_val.txt"
#X_val = load_X(X_val_path)
#print X_val
#
#preds = sess.run(
#    [pred],
#    feed_dict={
#        x: X_val
#   }
#)
#
#print preds
```


```python
#sess.close()
print test_accuracies

```

![Accuracy](images/accuracy.png)
![confusion matrix](images/confusion_matrix.png "Title")

## Conclusion

Final accuracy of >90% is pretty good, considering that training takes about 7 minutes.

Noticeable confusion between activities of Clapping Hands and Boxing, and between Jumping Jacks and Waving Two Hands which is understandable.

In terms of the applicability of this to a wider dataset, I would imagine that it would be able to work for any activities in which the training included a views from all angles to be tested on. It would be interesting to see it's applicability to camera angles in between the 4 used in this dataset, without training on them specifically.

 Overall, this experiment validates the idea that 2D pose can be used for at least human activity recognition, and provides verification to continue onto use of 2D pose for behaviour estimation in both people and animals
 

 ### With regards to Using LSTM-RNNs
 - Batch sampling
     - It is neccessary to ensure you are not just sampling classes one at a time! (ie y_train is ordered by class and batch chosen in order)The use of random sampling of batches without replacement from the training data resolves this.    
 
 - Architecture
     - Testing has been run using a variety of hidden units per LSTM cell, with results showing that testing accuracy achieves a higher score when using a number of hidden cells approximately equal to that of the input, ie 34. The following figure displays the final accuracy achieved on the testing dataset for a variety of hidden units, all using a batch size of 4096 and 300 epochs (a total of 1657 iterations, with testing performed every 8th iteration).
   
 
 

## Future Works

Inclusion of :

 - A pipeline for qualitative results
 - A validation dataset
 - Momentum     
 - Normalise input data (each point with respect to distribution of itself only)
 - Dropout
 - Comparison of effect of changing batch size
 

Further research will be made into the use on more subtle activity classes, such as walking versus running, agitated movement versus calm movement, and perhaps normal versus abnormal behaviour, based on a baseline of normal motion.


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

## 
