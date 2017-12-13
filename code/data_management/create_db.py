import json
from pprint import pprint
import glob, os
import numpy as np

output_file_X = "X_all.txt"
output_file_Y = "Y_all.txt"

test_file_X = "X_test.txt"
test_file_Y = "Y_test.txt"
train_file_X = "X_train.txt"
train_file_Y = "Y_train.txt"


data_path = "/home/stuart/data/activities"
activity_list =["jumping","jumping_jacks","boxing","waving_2hands","waving_1hand","clapping_hands"]
#activity 3, bending, not to be included due to extreme pose distortion
cluster_nums=4
camera_nums=1
subject_nums=12
activity_nums=7
repetition_nums=5

num_steps = 33
test_train_split = 0.8
split = True

for activity in range(0,len(activity_list)):
	os.chdir(os.path.join(data_path,activity_list[activity]))
	for file in sorted(glob.glob("*.txt")):
		data_file = open(file,'r')
		file_text = data_file.readlines() 
		num_frames = len(file_text)
		num_framesets = int(num_frames / num_steps)
		data_file.close
		xoutput_file = open(os.path.join(data_path,output_file_X), 'a')
		for line in range(0,num_framesets*num_steps):
			xoutput_file.write(file_text[line])
		xoutput_file.close()
		youtput_file = open(os.path.join(data_path,output_file_Y), 'a')
		for line in range(0,num_framesets):
			youtput_file.write(str(activity+1)+"\n")
		youtput_file.close()

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

				

