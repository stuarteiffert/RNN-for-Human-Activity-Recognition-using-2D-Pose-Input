#!/bin/bash

OPENPOSE_FOLDER="~/code/openpose"
DATA_FOLDER="~/data/HAR_pose_activities"
ACTIVITY=2
ACTIVITIES=("empty" "jumping" "jumping_jacks" "deleted" "boxing" "waving_2hands" "waving_1hand" "clapping_hands") 


CLUSTERS=4
CAMERA=1
SUBJECTS=12
ACTIVITIES_NUM=7
REPETITIONS=5


cd ${OPENPOSE_FOLDER}

for l in `seq 4 $ACTIVITIES_NUM`;
do
	for j in `seq 1 $CLUSTERS`;
	do
		for k in `seq 1 $SUBJECTS`;
		do
			for i in `seq $REPETITIONS`;
			do
				if [ $k -gt 1 ] || [ $j -gt 1 ]
				then
					if [ $k -gt 9 ]
					then
						IMAGE_DIR=$DATA_FOLDER/${ACTIVITIES[$l]}/"l0"$j"_c0"$CAMERA"_s"$k"_a0"$l"_r0"$i
					else
						IMAGE_DIR=$DATA_FOLDER/${ACTIVITIES[$l]}/"l0"$j"_c0"$CAMERA"_s0"$k"_a0"$l"_r0"$i
					fi
					OUTPUT_DIR=$IMAGE_DIR/"pose"
					#echo $IMAGE_DIR
					./build/examples/openpose/openpose.bin --image_dir $IMAGE_DIR --write_images $OUTPUT_DIR --write_keypoint_json $OUTPUT_DIR
				fi
			done
		done
	done
done

