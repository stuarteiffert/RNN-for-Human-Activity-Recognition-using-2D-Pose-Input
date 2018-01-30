"""
Creates database from converted OpenPose output files (should now be in .txt format) for use with RNN for Human Activity Recognition - 2D Pose Input

Created by Stuart Eiffert 13/12/2017

All code is provided under the MIT License

"""

import json
from pprint import pprint
import glob, os
import numpy as np

test_file_X = "X_test.txt"
test_file_Y = "Y_test.txt"
train_file_X = "X_train.txt"
train_file_Y = "Y_train.txt"


data_path = "/home/stuart/data/HAR_pose_activities/images"
activity_list =["jumping","jumping_jacks","boxing","waving_2hands","waving_1hand","clapping_hands"]
#activity 3, bending, not to be included due to extreme pose distortion
cluster_nums=4
camera_nums=1
subject_nums=12
activity_nums=7
repetition_nums=5

#num_steps depends on rate of video and window_width to be used
#in this case camera was 22Hz and a window_width of 1.5s was wanted, giving 22*1.5 = 33
num_steps = 32
test_train_split = 0.8
split = False
overlap = 0.8125 # 0 = 0% overlap, 1 = 100%

#Read in all text files and create x_all and y_all files, with no splitting yet
#cut each frameset into 33 frame chunks (the input window width for out network)
#Note: No overlap has been performed yet!

#Redo: 	Make test/train split be done with entire framesets, so that no overlap is found at all
#	Then, expand the set based on required overlap

for activity in range(0,len(activity_list)):
	os.chdir(os.path.join(data_path,activity_list[activity]))
	for file in sorted(glob.glob("*.txt")):
		#1. decide whether this entire frameset will be added to the test or train dataset (based on test_train_split)
		is_train = np.random.rand(1) < test_train_split
		#2. Have decided not to lop off any initial or end frames, eg as the subjects raise/lower hands whilst boxing. 
		#3. Determine how many chunks can be made from the frameset with required overlap and window width
		print is_train
		data_file = open(file,'r')
		file_text = data_file.readlines() 
		#check how many framesets of length 33 we can get out of the total frames in this files
		num_frames = len(file_text)
		num_framesets = int((num_frames - num_steps)/(num_steps*(1-overlap)))+1
		data_file.close
		#copy data to either test or train set dependent on is_train
		if is_train:
			output_file_X = train_file_X
			output_file_Y = train_file_Y
		else:
			output_file_X = test_file_X
			output_file_Y = test_file_Y
		x_file = open(os.path.join(data_path,output_file_X), 'a')
		for frameset in range(0, num_framesets):
			start = int(frameset*num_steps*(1-overlap))
			for line in range(start,(start+num_steps)):
				x_file.write(file_text[line])
		x_file.close()

		y_file = open(os.path.join(data_path,output_file_Y), 'a')
		for line in range(0,num_framesets):
			y_file.write(str(activity+1)+"\n")
		y_file.close()


#check if a test/train split is wanted
if split:
	print "splitting"
	os.chdir(data_path)
	X_file = open(output_file_X,'r')
	X_data = X_file.readlines() 
	X_file.close
	Y_file = open(output_file_Y,'r')
	Y_data = Y_file.readlines() 
	Y_file.close

	msk = np.random.rand(len(Y_data)) < test_train_split
	
	for i in range(len(X_data)): 
		num = int(i / num_steps)
		if msk[num] == True:
			X_train_file = open(train_file_X,'a')
			X_train_file.write(X_data[i])
			X_train_file.close()
		else:
			X_test_file = open(test_file_X,'a')
			X_test_file.write(X_data[i])
			X_test_file.close()

	for i in range(len(Y_data)): 
		num = i % num_steps
		if msk[i] == True:
			Y_train_file = open(train_file_Y,'a')
			Y_train_file.write(Y_data[i])
			Y_train_file.close()
		else:
			Y_test_file = open(test_file_Y,'a')
			Y_test_file.write(Y_data[i])
			Y_test_file.close()
	print msk

				

