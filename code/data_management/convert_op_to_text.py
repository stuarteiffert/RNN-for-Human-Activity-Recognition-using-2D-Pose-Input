import json
from pprint import pprint
import glob, os

data_path = "/home/stuart/data/activities"
activity_list =["empty","jumping","jumping_jacks","deleted","boxing","waving_2hands","waving_1hand","clapping_hands"]
#activity 3, bending, not to be included due to extreme pose distortion
cluster_nums=4
camera_nums=1
subject_nums=12
activity_nums=7
repetition_nums=5

count = 0

for cluster in range(1,cluster_nums+1):
	for camera in range(1,camera_nums+1):
		for subject in range(1,subject_nums+1):
			subject_num = "0"+str(subject)
			if subject > 9:
				subject_num = str(subject)
			for camera in range(1,camera_nums+1):
				for activity in range(1,activity_nums+1):
					if activity != 3:						
						for repetition in range(1,repetition_nums+1):
							frame_set = "l0"+str(cluster)+"_c0"+str(camera)+"_s"+subject_num+"_a0"+str(activity)+"_r0"+str(repetition)
							pose_kp_path = os.path.join(data_path,activity_list[activity],frame_set,"pose")
							#print pose_kp_path
							if ((frame_set == "l03_c01_s05_a02_r05") or (frame_set == "l04_c01_s05_a02_r05")):
								continue
							os.chdir(pose_kp_path)

							#kps is a list of pose keypoints in each frame, where kps[0] is the x position of kp0, kps[1] is the y position of kp0 etc
							kps = []
							#[kps.append([]) for i in range(36)]

							for file in sorted(glob.glob("*.json")):
								with open(file) as data_file: 
									data = json.load(data_file)								
									if len(data["people"]) > 1:
										pprint("More than one detection in file, check the noise:")
										count += 1
										print file
									frame_kps = []
									pose_keypoints = data["people"][0]["pose_keypoints"]
									j = 0
									for i in range(36):
										frame_kps.append(pose_keypoints[j])
										j += 1
										if ((j+1) % 3 == 0):
											j += 1
									kps.append(frame_kps)


									#frames = (len(kps))
									#n_steps = 40 #22Hz dataset, so about 2 secs for sliding window, we will use 50% overlap too
									#num = int(frames/ n_steps)
									#print(str(file)+" json read, frames left out:")
									#print (frames - num)
							os.chdir(os.path.join(data_path,activity_list[activity]))
							output_file = activity_list[activity]+frame_set+".txt"
							with open(output_file, "w") as text_file:
								for i in range(len(kps)):
									for j in range(36):
										text_file.write('{}'.format(kps[i][j]))
										if j < 35:
											text_file.write(',')
									text_file.write('\n')
print count

