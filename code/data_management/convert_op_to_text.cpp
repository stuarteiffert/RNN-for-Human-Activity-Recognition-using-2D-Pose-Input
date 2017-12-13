#include "json.hpp"
#include <fstream>
#include<stdio.h>

using json = nlohmann::json;

int main(int argc, char **argv)
{
	printf("Hello! This is my first C program with Ubuntu 11.10\n");
	std::ifstream i("img_l01_c01_s01_a01_r01_00000_keypoints.json");
	json j = json::parse(i);
	//for (int i = 0, i < j["people"].length(), i++)
	std::cout << sizeof(j["people"][i]["pose_keypoints"]) << std::endl;

}
