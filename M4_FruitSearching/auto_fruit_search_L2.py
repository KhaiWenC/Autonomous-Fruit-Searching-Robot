# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time

# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure

# -- New Code -- #
from operate import Operate
from dijkstra import Dijkstra # path planning algorithm
import pygame # python package for GUI
import matplotlib.pyplot as plt
show_animation = True
# -- End -- # 

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    
    # -- New Code -- # 
    wheel_vel = 25 # tick to move the robot
    
    # Extract x-coordinate and y-coordinate of the waypoint
    x_waypoint, y_waypoint = waypoint
    # Extract current x-coordinate, y-coordinate and orientation of the robot from robot_pose 
    x, y, theta = robot_pose
    # Compute horizontal distance between x-coordinate of the waypoint and the current x-coordinate of the robot	
    x_diff = x_waypoint - x
    # Compute vertical distance between y-coordinate of the waypoint and the current y-coordinate of the robot	
    y_diff = y_waypoint - y
    # Compute Euclidean distance between the robot and the waypoint
    distance_between_waypoint_and_robot = np.hypot(x_diff, y_diff)
    # Compute turning angle required by the robot to face the waypoint
    turning_angle = np.arctan2(y_diff, x_diff) - theta 
    if (turning_angle > np.pi):
        turning_angle -= 2*np.pi
    elif (turning_angle < -np.pi):
        turning_angle += 2*np.pi
    # Compute turn_time 
    turn_time = ((abs(turning_angle)*baseline) / (2*wheel_vel*scale))
    print("Turning for {:.2f} seconds".format(turn_time))
    
    # turn towards the waypoint
    if (turning_angle > 0): # rotating anticlockwise
        drive_and_update_slam([0, 1], wheel_vel, turn_time)
    elif (turning_angle < 0): # rotating clockwise
        drive_and_update_slam([0, -1], wheel_vel, turn_time)

    wheel_vel = 20 
    # after turning, drive straight to the waypoint
    drive_time = distance_between_waypoint_and_robot / (wheel_vel * scale) 
    print("Driving for {:.2f} seconds".format(drive_time))
    drive_and_update_slam([1, 0], wheel_vel, drive_time)
    # -- End -- # 
    ####################################################
 
    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    

def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    # -- New Code -- #
    x_new = operate.ekf.robot.state[0,0]
    y_new = operate.ekf.robot.state[1,0]
    theta_new = operate.ekf.robot.state[2,0]
    robot_pose = [x_new, y_new, theta_new] 
    # -- End -- #
    ####################################################

    return robot_pose

# -- New Code -- #
def drive_to_fruit(robot_pose, rx, ry):
    for i in range(len(rx)):
        x = rx[len(rx)-i-1]
        y = ry[len(rx)-i-1]
        x = round(x, 1)
        y = round(y, 1)
        
        # robot drives to the waypoint
        waypoint = [x,y]
        operate.notification = "Moving to Waypoint (X,Y): " + str(x)+','+ str(y)
        drive_to_point(waypoint,robot_pose)
        operate.notification = "Reached Waypoint (X,Y): " + str(x)+','+ str(y)
        
        # scan aruco markers after two steps
        if ((i+1) % 2 == 0): 
            scan_aruco_markers()
            
        # estimate the robot's pose
        robot_pose = get_robot_pose()
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
        drive_and_update_slam([0, 0], 0, 0)

def drive_and_update_slam(command, wheel_vel, run_time):
    time_offset = 0
    if (command[1] == 1):
        time_offset = 0.001
    elif (command[1] == -1):
        time_offset = 0.001
    if time_offset:
        lv,rv = ppi.set_velocity(command,tick=wheel_vel, turning_tick=wheel_vel, time=run_time+time_offset)
    else:
        lv,rv = ppi.set_velocity(command,tick=wheel_vel, turning_tick=wheel_vel, time=run_time)
    drive_meas = measure.Drive(lv, rv, run_time)
    time.sleep(1)
    operate.take_pic()
    operate.update_slam(drive_meas)
    operate.draw(canvas)
    pygame.display.update()

def scan_aruco_markers():
    for k in range(8):
        turning_required = (np.pi/180)*45
        wheel_velo = 20
        turning_time_required = ((abs(turning_required)*baseline) / (2*wheel_velo*scale))
        drive_and_update_slam([0, -1], wheel_velo, turning_time_required)
        
def drive_back_to_origin(robot_pose, rx_return, ry_return):
    for i in range(len(rx_return)):
        x = rx_return[i]
        y = ry_return[i]
        x = round(x, 1)
        y = round(y, 1)
        
        # robot drives to the waypoint
        waypoint = [x,y]
        operate.notification = "Moving to Waypoint (X,Y): " + str(x)+','+ str(y)
        drive_to_point(waypoint,robot_pose)
        operate.notification = "Reached Waypoint (X,Y): " + str(x)+','+ str(y)
        
        # scan aruco markers after three steps
        if ((i+1) % 3 == 0): 
            scan_aruco_markers()
            
        # estimate the robot's pose
        robot_pose = get_robot_pose()
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
        drive_and_update_slam([0, 0], 0, 0)
        
# -- End -- # 

# main loop
if __name__ == "__main__":
    # -- New Code -- #
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M5_estimated_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.60')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()
    ppi = Alphabot(args.ip,args.port)
    operate = Operate(args)
    # -- End -- # 
    
    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    # -- New Code -- # 
    # GUI 
    pygame.font.init() 
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2
    
    # Run SLAM 
    n_observed_markers = len(operate.ekf.taglist)
    if n_observed_markers == 0:
        if not operate.ekf_on:
            operate.notification = 'SLAM is running'
            operate.ekf_on = True
        else:
            operate.notification = '> 2 landmarks is required for pausing'
    elif n_observed_markers < 3:
        operate.notification = '> 2 landmarks is required for pausing'
    else:
        if not operate.ekf_on:
            operate.request_recover_robot = True
        operate.ekf_on = not operate.ekf_on
        if operate.ekf_on:
            operate.notification = 'SLAM is running'
        else:
            operate.notification = 'SLAM is paused'
    
    # Add landmarks (aruco markers)
    lms = []
    for i,aruco in enumerate(aruco_true_pos):
        lm = measure.Marker(np.array([[aruco[0]],[aruco[1]]]),i+1)
        lms.append(lm)
    operate.ekf.add_landmarks(lms)
    
    # Path planning 
    # Obstacles positions
    ox, oy = [], []
    # Grid resolution 
    resolution = 0.1
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    robot_radius = 0.15 # manual offset (baseline/2)
    robot_pose = get_robot_pose()
    
    # For loop to store x-coordinate and y-coordinate of aruco markers into ox and oy respectively
    for i in range(len(aruco_true_pos)):
        aruco_x = aruco_true_pos[i, :][0]
        aruco_y = aruco_true_pos[i, :][1]
        ox.append(aruco_x)
        oy.append(aruco_y)
    
    # For loop to store x-coordinate and y-coordinate of arena boundaries into ox and oy respectively
    for i in range(-160, 160): # Upper boundary 
        ox.append(i/100)
        oy.append(1.6)
    for i in range(-160, 160): # Lower Boundary
        ox.append(i/100)
        oy.append(-1.6)
    for i in range(-160, 160): # Left Boundary
        ox.append(-1.6)
        oy.append(i/100)
    for i in range(-160, 160): # Right Boundary 
        ox.append(1.6)
        oy.append(i/100)
    
    # For loop to loop through each fruit in search list 
    for i in range(len(search_list)):
        
        drive_and_update_slam([0, 0], 0, 0)  
            
        # For each fruit search, scan through all the aruco markers    
        scan_aruco_markers()
        
        robot_pose = get_robot_pose()
        sx = float(robot_pose[0])
        sy = float(robot_pose[1])
        current_fruit = search_list[i]
        gx = 0
        gy = 0
        search_index = 0
        for n in range(len(fruits_list)):
            if (current_fruit == fruits_list[n]):
                search_index = n 
                gx = fruits_true_pos[n, :][0]
                gy = fruits_true_pos[n, :][1]
        print("Goal x: {}".format(gx))
        print("Goal y: {}".format(gy))
        for j in range(len(fruits_list)):
            if (fruits_list[j] != current_fruit):
                ox.append(fruits_true_pos[j, :][0])
                oy.append(fruits_true_pos[j, :][1])
        
        if show_animation:  # pragma: no cover
            plt.plot(ox, oy, ".k")
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xb")
            plt.grid(True)
            space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])    
            plt.xlabel("X"); plt.ylabel("Y")
            plt.xticks(space); plt.yticks(space)
            
        dijk = Dijkstra(ox, oy, resolution, robot_radius)
        rx, ry = dijk.planning(sx, sy, gx, gy)
        
        rx_return, ry_return = dijk.planning(sx, sy, gx, gy)       
        rx_return.pop(0)
        ry_return.pop(0)
        rx_return.pop(0)
        ry_return.pop(0)
        
        # Remove starting point     
        rx.pop(-1)
        ry.pop(-1)
        # Remove goal to avoid collision
        rx.pop(0)
        ry.pop(0)
        # rx.pop(0)
        # ry.pop(0)
        # rx.pop(0)
        # ry.pop(0)
        
        if show_animation:  # pragma: no cover
            plt.title("Generated Path for {}".format(fruits_list[search_index]))
            plt.plot(rx, ry, "-r")
            #plt.pause(0.01)
            #plt.show() 
            plt.savefig("{}_map".format(fruits_list[search_index]))
            plt.close()
            
        operate.notification = "Searching for {} ...".format(fruits_list[search_index])
        drive_to_fruit(robot_pose, rx, ry)
        operate.notification = "Reached {} !".format(fruits_list[search_index])
        
        # - Newly Added - #
        # operate.detect_target()
        # while (operate.detector_output != search_list[i]):
            # turning_angle = np.pi/6
            # wheel_vel = 20
            # turn_time = ((abs(turning_angle)*baseline) / (2*wheel_vel*scale))
            # for j in range(13):   
                # drive_and_update_slam([0, 1], wheel_vel, turn_time)
                # operate.detect_target()
                # if (operate.detector_output == search_list[i]):
                    # break
        
        # now_robot_pose = get_robot_pose()
        # rx, ry = dijk.planning(now_robot_pose[0], now_robot_pose[1], gx, gy)
        # rx.pop(-1)
        # ry.pop(-1)
        # rx.pop(0)
        # ry.pop(0)
        # rx.pop(0)
        # ry.pop(0)
        # operate.notification = "Searching for {} ...".format(fruits_list[search_index])
        # drive_to_fruit(robot_pose, rx, ry)
        # operate.notification = "Reached {} !".format(fruits_list[search_index])
        # - End - # 
        
        for p in range(4):
            ox.pop(-1)
            oy.pop(-1)
           
        time.sleep(3)
        drive_and_update_slam([0, 0], 0, 0)
        
        # Return to origin 
        if (i < 2):
            fig = plt.figure
            plt.plot(ox, oy, ".k")
            robot_pose_back = get_robot_pose()
            plt.plot(robot_pose_back[0], robot_pose_back[1], "og")
            plt.plot(0, 0, "xb")
            plt.grid(True)
            space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])    
            plt.xlabel("X"); plt.ylabel("Y")
            plt.xticks(space); plt.yticks(space)
            plt.title("Generated Path for {} to return to Origin".format(fruits_list[search_index]))
            plt.plot(rx_return, ry_return, "-r")
            #plt.pause(0.01)
            #plt.show() 
            plt.savefig("{}_return_to_origin_map".format(fruits_list[search_index]))
            plt.close()
            fig = plt.figure
            operate.notification = "Driving Back to Origin"
            drive_back_to_origin(robot_pose_back, rx_return, ry_return)
            operate.notification = "Reached Origin"
        
    # -- End -- # 