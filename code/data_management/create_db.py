import json
from pprint import pprint
import glob, os

output_file_X = "X_train.txt"
output_file_Y = "Y_train.txt"


data_path = "/home/stuart/data/activities"
activity_list =["jumping","jumping_jacks","boxing","waving_2hands","waving_1hand","clapping_hands"]
#activity 3, bending, not to be included due to extreme pose distortion
cluster_nums=1
camera_nums=1
subject_nums=1
activity_nums=7
repetition_nums=5

num_steps = 33
test_train_split = 0.8

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
			youtput_file.write(str(activity)+"\n")
		youtput_file.close()
				

