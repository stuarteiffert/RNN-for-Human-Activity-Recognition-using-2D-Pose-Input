# RNN for Human Activity Recognition - 2D Pose Input Dataset
The dataset consists of 2D pose estimations, made using the software OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose's) on a subset of the Berkeley Multimodal Human Action Database (MHAD) dataset http://tele-immersion.citris-uc.org/berkeley_mhad.

This dataset is comprised of 12 subjects doing the following 6 actions for 5 repetitions, filmed from 4 angles, repeated 5 times each.
JUMPING,
JUMPING_JACKS,
BOXING,
WAVING_2HANDS,
WAVING_1HAND,
CLAPPING_HANDS.

Only the output 2D poses of each frame have been included. The videos and output frames have not been included in the online dataset due to size (>50GB).
If you would like to download the videos directly from the Berkley site, the videos used were actions 1,2,4,5,6,7 for all repetitions, recordings and subjects.
I will look at hosting the 2D pose image outputs in an accessible location soon.

The data is in the following layout:

HAR_pose_activities  
&nbsp;&nbsp;└──databse  
&nbsp;&nbsp;&nbsp;&nbsp;└──x_test.txt  
&nbsp;&nbsp;&nbsp;&nbsp;└──x_train.txt  
&nbsp;&nbsp;&nbsp;&nbsp;└──y_test.txt  
&nbsp;&nbsp;&nbsp;&nbsp;└──y_traint.txt  

And format (for both test and train):

Input:

	[	[j0_x, j0_y, j1_x, j1_y , j2_x, j2_y, j3_x, j3_y, j4_x, j4_y, j5_x, j5_y, j6_x, j6_y, j7_x, j7_y, j8_x, j8_y, j9_x, j9_y, j10_x, j10_y, j11_x, j11_y, j12_x, j12_y, j13_x, j13_y, j14_x, j14_y, j15_x, j15_y, j16_x, j16_y, j17_x, j17_y] (frame0)  

		[j0_x, j0_y, j1_x, j1_y , j2_x, j2_y, j3_x, j3_y, j4_x, j4_y, j5_x, j5_y, j6_x, j6_y, j7_x, j7_y, j8_x, j8_y, j9_x, j9_y, j10_x, j10_y, j11_x, j11_y, j12_x, j12_y, j13_x, j13_y, j14_x, j14_y, j15_x, j15_y, j16_x, j16_y, j17_x, j17_y] (frame1)  
  
		...  
  
		[j0_x, j0_y, j1_x, j1_y , j2_x, j2_y, j3_x, j3_y, j4_x, j4_y, j5_x, j5_y, j6_x, j6_y, j7_x, j7_y, j8_x, j8_y, j9_x, j9_y, j10_x, j10_y, j11_x, j11_y, j12_x, j12_y, j13_x, j13_y, j14_x, j14_y, j15_x, j15_y, j16_x, j16_y, j17_x, j17_y] (frameN)  
	]

Output:

	[	[class_id] (frame0)  
		[class_id] (frame0)  
		...  
		[class_id] (frameN)  
	]


