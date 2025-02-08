# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os, sys
import time
import json
from pathlib import Path
import ast
import matplotlib.pyplot as plt
from SLAM_eval import *

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

class Operate:

    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = Alphabot(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False,
                        'output2': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)

        ## transform parameters
        self.theta_trans = 0
        self.x_trans = 0
        self.y_trans = 0
        self.mode = 0
        
        self.network_vis = np.ones((240, 320,3))* 100
        
        self.bg = pygame.image.load('pics/gui_mask.jpg')

    # wheel control
    def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
            print('in ip')
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        ## paint SLAM outputs
        # Camera View
        ekf_view = self.ekf.draw_slam_state(res=(480, 480+v_pad), #280,480
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))

        ## Drawing Real-Time Map
        res = (480, 480) #480, 320
        map = cv2.resize(cv2.imread(f'./pics/map.jpg'), res)
        map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
        map = pygame.surfarray.make_surface(np.rot90(map))
        map = pygame.transform.flip(map, True, False)
        canvas.blit(map, (2 * h_pad +320+530, v_pad))
        self.put_caption(canvas, caption='Map', position=(2*h_pad+320+530, v_pad))

        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))


    ### Custom Transformation
    def parse_user_map(self,fname: str) -> dict:
        with open(fname, 'r') as f:
            usr_dict = ast.literal_eval(f.read())
            aruco_dict = {}
            for (i, tag) in enumerate(usr_dict["taglist"]):
                aruco_dict[tag] = np.reshape([usr_dict["map"][0][i], usr_dict["map"][1][i]], (2, 1))
        return aruco_dict

    def parse_robot_pose(self,fname: str) -> dict:
        with open(fname, 'r') as f:
            robot_pose_dict = ast.literal_eval(f.read())
            robot_pose = []
            robot_pose.append(robot_pose_dict["rpose_x"])
            robot_pose.append(robot_pose_dict["rpose_y"])
            robot_pose.append(robot_pose_dict["theta"])
        return robot_pose

    def arrange_aruco_points(self,aruco0: dict):
        points0 = []
        keys = []
        for i in range(10):
            for key in aruco0:
                if key == i + 1:
                    points0.append(aruco0[key])
                    keys.append(key)
        return keys, np.hstack(points0)

    ##From Slam_Eval
    def apply_transform(self,theta, x, points):
        # Apply an SE(2) transform to a set of 2D points
        assert (points.shape[0] == 2)

        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        points_transformed = R @ points + x
        return points_transformed

    #Map plotting
    def map_plot(self):
        us_aruco = self.parse_user_map("lab_output/slam.txt")

        taglist, us_vec = self.arrange_aruco_points(us_aruco)

        us_vec_transformed = self.apply_transform(self.theta_trans, [[self.x_trans],[self.y_trans]], us_vec)

        ax = plt.gca()
        ax.scatter(us_vec_transformed[0, :], us_vec_transformed[1, :], marker='x', color='b', s=100)
        for i in range(len(taglist)):
            ax.text(us_vec_transformed[0, i] + 0.05, us_vec_transformed[1, i] + 0.05, taglist[i], color='b', size=12)
        plt.title('Arena')
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        plt.legend(['Pred'])
        plt.grid(linewidth=2)
        plt.savefig('pics\map.jpg')
        plt.close()


    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [3, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-3, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 3]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -3]
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            ####################################################
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_KP5:
                # TODO: replace with your code to make the robot drive backward
                self.command['motion'] = [1, 0]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_KP2:
                # TODO: replace with your code to make the robot drive backward
                self.command['motion'] = [1, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_KP4:
                # TODO: replace with your code to make the robot turn left
                self.command['motion'] = [0, 1]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_KP6:
                # TODO: replace with your code to make the robot turn right
                self.command['motion'] = [0, -1]
            ####################################################
            ### stop
            #elif event.type == pygame.keyup and event.key == pygame.k_up:
            #    # todo: replace with your code to make the robot drive backward
            #    self.command['motion'] = [0, 0]
            #elif event.type == pygame.keyup and event.key == pygame.k_down:
            #    # todo: replace with your code to make the robot drive backward
            #    self.command['motion'] = [0, 0]
            ## turn left
            #elif event.type == pygame.keyup and event.key == pygame.k_left:
            #    # todo: replace with your code to make the robot turn left
            #    self.command['motion'] = [0, 0]
            ## drive right
            #elif event.type == pygame.keyup and event.key == pygame.k_right:
            #    # todo: replace with your code to make the robot turn right
            #    self.command['motion'] = [0, 0]
            #    ################

            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                x_robot = self.ekf.robot.state[0][0]
                y_robot = self.ekf.robot.state[1][0]
                robot_angle = self.ekf.robot.state[2][0]

                print('--------')
                print("robot pos and angle")
                print(np.array([[x_robot], [y_robot]]))
                print(robot_angle)
                print('--------')

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                x_robot = self.ekf.robot.state[0][0]
                y_robot = self.ekf.robot.state[1][0]

                robot_angle = abs((self.ekf.robot.state[2][0]) % (2 * np.pi))
                if (self.ekf.robot.state[2][0] < 0):
                    robot_angle = np.pi * 2 - robot_angle

                d = {}
                with open(base_dir /'lab_output/robot_pose.txt', 'w') as fo:
                    d = {"rpose_x": x_robot, "rpose_y": y_robot,"theta": robot_angle}
                    json.dump(d, fo)
                    self.notification = 'Robot pose is saved'
                    
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                us_aruco = self.parse_user_map("lab_output/slam.txt")

                taglist, us_vec = self.arrange_aruco_points(us_aruco)
                print(self.theta_trans)
                print(self.x_trans)
                print(self.y_trans)
                us_vec_transformed = self.apply_transform(self.theta_trans, [[self.x_trans],[self.y_trans]], us_vec)

                d = {}
                for i in range(len(taglist)):
                    d["aruco" + str(i + 1) + "_0"] = {"x": us_vec_transformed[0][i], "y": us_vec_transformed[1][i]}

                with open(base_dir / 'PredMap.txt', 'w') as f:
                    json.dump(d, f, indent=4)

                self.notification = 'Predicted Map generated'

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.map_plot()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_z:
                self.mode += 1
                
                if self.mode == 3:
                    self.mode = 0

                if self.mode == 0:
                    self.notification = 'Mode 0, theta'

                elif self.mode == 1:
                    self.notification = 'Mode 1, x_translation'
                elif self.mode == 2:
                    self.notification = 'Mode 2, y_translation'

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                if self.mode == 0:
                    self.theta_trans -= 0.01
                elif self.mode == 1:
                    self.x_trans -= 0.01
                elif self.mode == 2:
                    self.y_trans -= 0.01

                self.map_plot()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                if self.mode == 0:
                    self.theta_trans += 0.01
                elif self.mode == 1:
                    self.x_trans += 0.01
                elif self.mode == 2:
                    self.y_trans += 0.01

                self.map_plot()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                if self.mode == 0:
                    self.theta_trans -= 0.1
                elif self.mode == 1:
                    self.x_trans -= 0.1
                elif self.mode == 2:
                    self.y_trans -= 0.1

                self.map_plot()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                if self.mode == 0:
                    self.theta_trans += 0.1
                elif self.mode == 1:
                    self.x_trans += 0.1
                elif self.mode == 2:
                    self.y_trans += 0.1

                self.map_plot()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                slam_eval2()


        if self.quit:
            pygame.quit()
            sys.exit()



        
if __name__ == "__main__":
    import argparse
    base_dir = Path('./')
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.69')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 1600, 660
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

    operate = Operate(args)


    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        # visualise
        operate.draw(canvas)
        pygame.display.update()




