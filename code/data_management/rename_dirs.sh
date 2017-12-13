#!/bin/bash

#jumping = a01
#jumping_jacks = a02
#bending = a03
#boxing = a04
#waving_2hands = a05
#waving_1hand = a06
#clapping_hands = a07

ACTIVITIES=("empty" "jumping" "jumping_jacks" "bending" "boxing" "waving_2hands" "waving_1hand" "clapping_hands") 


CLUSTER=4
CAMERA=1
SUBJECT=1
NUM_ACTIVITIES=7
REPETITIONS=5

DOWNLOAD_FOLDER="/home/stuart/Downloads/BerkeleyMHAD/Camera"
DATA_FOLDER="/home/stuart/data/activities"


cd ${DATA_FOLDER}

for k in `seq 1 $SUBJECT`;
do
	for j in `seq 6 $NUM_ACTIVITIES`;
	do
		if [ $j != 3 ]
		then
			for i in `seq 1 $REPETITIONS`;
			do
				if [ $k -gt 9 ]
				then
					mv $DOWNLOAD_FOLDER/"Cluster0"$CLUSTER/"Cam0"$CAMERA/"S"$k/"A0"$j/"R0"$i $DATA_FOLDER/${ACTIVITIES[$j]}/"l0"$CLUSTER"_c0"$CAMERA"_s"$k"_a0"$j"_r0"$i
				else
					mv $DOWNLOAD_FOLDER/"Cluster0"$CLUSTER/"Cam0"$CAMERA/"S0"$k/"A0"$j/"R0"$i $DATA_FOLDER/${ACTIVITIES[$j]}/"l0"$CLUSTER"_c0"$CAMERA"_s0"$k"_a0"$j"_r0"$i
				fi
				#echo $DOWNLOAD_FOLDER/"Cluster0"$CLUSTER/"Cam0"$CAMERA/"S0"$k/"A0"$j/"R0"$i
				#echo $DATA_FOLDER/${ACTIVITIES[$j]}/"l0"$CLUSTER"_c0"$CAMERA"_s0"$k"_a0"$j"_r0"$i
			done
		fi
	done
done

#./build/examples/openpose/openpose.bin --image_dir /home/stuart/data/activities/jumping/subject1_cluster4_cam1 --write_images /home/stuart/data/activities/jumping/subject1_cluster4_cam1/pose --write_keypoint_json /home/stuart/data/activities/jumping/subject1_cluster4_cam1/pose

#./build/examples/openpose/openpose.bin --image_dir /home/stuart/data/activities/jumping/subject1_cluster4_cam1 --write_images /home/stuart/data/activities/jumping/subject1_cluster4_cam1/pose --write_keypoint_json /home/stuart/data/activities/jumping/subject1_cluster4_cam1/pose


