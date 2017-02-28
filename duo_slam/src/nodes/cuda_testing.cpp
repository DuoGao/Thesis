#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 
#include "ros/ros.h"

typedef unsigned char GRID_TYPE;

int create_grid();
int delete_grid();
int fill_grid(GRID_TYPE value = 0, bool auto_value = true);
int get_grid_size();
void increase_value(int index);
GRID_TYPE retrieve_value(int index);

ros::NodeHandle* nh;

int main(int argc, char** argv) {
//    ros::init(argc, argv, "slam_test");
//    ROS_INFO("slam_test node started...");
//    nh = new ros::NodeHandle("~");
//
//    double start, end;
//
//    start = ros::Time::now().toSec();
//    create_grid();
//    end = ros::Time::now().toSec();
//    printf("Creation time %f\n", end - start);
//
//    start = ros::Time::now().toSec();
//    fill_grid(0, false);
//    end = ros::Time::now().toSec();
//    printf("Fill time %f\n", end - start);
//
//    srand(time(NULL));
//    int index = rand() % (get_grid_size() - 1);
//
//    start = ros::Time::now().toSec();
//    for (int i = 0; i < 100; i++) {
//        increase_value(index);
//    }
//    end = ros::Time::now().toSec();
//    printf("Increase time %f\n", end - start);
//
//
//    start = ros::Time::now().toSec();
//    GRID_TYPE value = retrieve_value(index);
//    end = ros::Time::now().toSec();
//    printf("Retrieve time %f\n", end - start);
//
//    printf("Value of %d is %d\n", index, value);
//    delete_grid();
    return 0;
}