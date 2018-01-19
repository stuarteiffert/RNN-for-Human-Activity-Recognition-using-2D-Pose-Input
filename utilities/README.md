# Utility Functions

The following scripts were used in the creation of the dataset for RNN for Human Activity Recognition - 2D Pose Input.
They were run in the listed order below.
Please note that any directory references will need to be changed before use, and that no liability is taken. Please read the code before using it.


run\_openpose.sh: runs openpose on all images in IMage\_DIR, outputting to OUTPUT\_DIR

convert\_json\_to\_text.py: Converts output of OpenPose (.json) to a .txt file

create_db.ps: Creates database from converted OpenPose output files (should now be in .txt format) for use with RNN for Human Activity Recognition - 2D Pose Input
