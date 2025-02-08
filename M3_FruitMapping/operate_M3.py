# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os, sys
import time
import json
import ast
import matplotlib.pyplot as plt
from pathlib import Path
from CV_eval import *

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

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector

## import target pose
from TargetPoseEst import *


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
                        'save_image': False}
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
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.ckpt, use_gpu=False)
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
                self.ekf_on = True
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    def init_markers(self, aruco_true_pos): #init markers
        x = []
        y = []
        meas = []

        for i in range(aruco_true_pos.shape[0]):
            x = aruco_true_pos[i][0]
            y = aruco_true_pos[i][1]
            tag = i + 1
            lms = measure.Marker(np.array([[x] ,[y]]), tag, 0.001*np.eye(2))
            meas.append(lms)

        self.ekf.add_landmarks(meas)

    def read_true_map(self, fname):
        with open(fname, 'r') as f:
            try:
                gt_dict = json.load(f)
            except ValueError as e:
                with open(fname, 'r') as f:
                    gt_dict = ast.literal_eval(f.readline())
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
                        aruco_true_pos[marker_id - 1][0] = x
                        aruco_true_pos[marker_id - 1][1] = y

        return aruco_true_pos

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

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

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(480, 480+v_pad),
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

        ## Drawing map
        res = (480, 480)
        map = cv2.resize(cv2.imread(f'./pics/map.jpg'), res)
        map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
        map = pygame.surfarray.make_surface(np.rot90(map))
        map = pygame.transform.flip(map, True, False)
        canvas.blit(map, (2 * h_pad + 320 + 530, v_pad))
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

    def parse_predmap(self,fname: str) -> dict:
        with open(fname, 'r') as f:
            gt_dict = ast.literal_eval(f.read())

            aruco_dict = {}
            for key in gt_dict:
                if key.startswith("aruco"):
                    aruco_num = int(key.strip('aruco')[:-2])
                    aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2, 1))
        return aruco_dict

    def arrange_aruco_points(self,aruco0: dict):
        points0 = []
        keys = []
        for i in range(10):
            for key in aruco0:
                if key == i + 1:
                    points0.append(aruco0[key])
                    keys.append(key)
        return keys, np.hstack(points0)

    def parse_fruit_map(self,fname: str) -> dict:
        with open(fname, 'r') as f:
            try:
                gt_dict = json.load(f)
            except ValueError as e:
                with open(fname, 'r') as f:
                    gt_dict = ast.literal_eval(f.readline())
            redapple_gt, greenapple_gt, orange_gt, mango_gt, capsicum_gt = [], [], [], [], []

            # remove unique id of targets of the same type
            for key in gt_dict:
                if key.startswith('redapple'):
                    redapple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                elif key.startswith('greenapple'):
                    greenapple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                elif key.startswith('orange'):
                    orange_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                elif key.startswith('mango'):
                    mango_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                elif key.startswith('capsicum'):
                    capsicum_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
        # if more than 1 estimation is given for a target type, only the first estimation will be used
        num_per_target = 1  # max number of units per target type. We are only use 1 unit per fruit type
        if len(redapple_gt) > num_per_target:
            redapple_gt = redapple_gt[0:num_per_target]
        if len(greenapple_gt) > num_per_target:
            greenapple_gt = greenapple_gt[0:num_per_target]
        if len(orange_gt) > num_per_target:
            orange_gt = orange_gt[0:num_per_target]
        if len(mango_gt) > num_per_target:
            mango_gt = mango_gt[0:num_per_target]
        if len(capsicum_gt) > num_per_target:
            capsicum_gt = capsicum_gt[0:num_per_target]

        return redapple_gt, greenapple_gt, orange_gt, mango_gt, capsicum_gt

    #Map Plotting
    def map_plot(self):
        #Import map from M2
        predmap = "PredMap.txt"
        us_aruco = self.parse_predmap(predmap)

        taglist, us_vec = self.arrange_aruco_points(us_aruco)

        #Fruit map
        fruits_pos()
        redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = self.parse_fruit_map("lab_output/targets.txt")

        ax = plt.gca()
        ax.scatter(us_vec[0, :], us_vec[1, :], marker='x', color='b', s=100)

        if(len(redapple_est) != 0):
            ax.scatter(redapple_est[0][0], redapple_est[0][1], marker='o', color='r', s=100)
        if (len(greenapple_est) != 0):
            ax.scatter(greenapple_est[0][0], greenapple_est[0][1], marker='o', color='mediumseagreen', s=100)
        if (len(orange_est) != 0):
            ax.scatter(orange_est[0][0], orange_est[0][1], marker='o', color='tab:orange', s=100)
        if (len(mango_est) != 0):
            ax.scatter(mango_est[0][0], mango_est[0][1], marker='o', color='y', s=100)
        if (len(capsicum_est) != 0):
            ax.scatter(capsicum_est[0][0], capsicum_est[0][1], marker='o', color='g', s=100)

        for i in range(len(taglist)):
            ax.text(us_vec[0, i] + 0.05, us_vec[1, i] + 0.05, taglist[i], color='b', size=12)
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
                        ########### replace with your M1 codes ###########
           # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                # TODO: replace with your code to make the robot drive forward
                self.command['motion'] = [3, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                # TODO: replace with your code to make the robot drive backward
                self.command['motion'] = [-3, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                # TODO: replace with your code to make the robot turn left
                self.command['motion'] = [0, 3]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                # TODO: replace with your code to make the robot turn right
                self.command['motion'] = [0, -3]
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
                self.command['motion'] = [0, 2]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_KP6:
                # TODO: replace with your code to make the robot turn right
                self.command['motion'] = [0, -2]
            ####################################################
            ### stop
            #elif event.type == pygame.KEYUP and event.key == pygame.K_UP:
            #    # TODO: replace with your code to make the robot drive backward
            #    self.command['motion'] = [0, 0]
            #elif event.type == pygame.KEYUP and event.key == pygame.K_DOWN:
            #    # TODO: replace with your code to make the robot drive backward
            #    self.command['motion'] = [0, 0]
            ## turn left
            #elif event.type == pygame.KEYUP and event.key == pygame.K_LEFT:
            #    # TODO: replace with your code to make the robot turn left
            #    self.command['motion'] = [0, 0]
            ## drive right
            #elif event.type == pygame.KEYUP and event.key == pygame.K_RIGHT:
            #    # TODO: replace with your code to make the robot turn right
            #    self.command['motion'] = [0, 0]
            #    ###

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
            #custom
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                x_robot = operate.ekf.robot.state[0][0]
                y_robot = operate.ekf.robot.state[1][0]

                robot_angle = abs((operate.ekf.robot.state[2][0]) % (2 * np.pi))
                if (operate.ekf.robot.state[2][0] < 0):
                    robot_angle = np.pi * 2 - robot_angle

                print('--------')
                print("robot pos and angle")
                print(np.array([[x_robot], [y_robot]]))
                print(robot_angle)
                print('--------')
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.map_plot()


        if self.quit:
            pygame.quit()
            sys.exit()
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default='PredMap.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
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
    aruco_true_pos = operate.read_true_map(args.map)
    print(aruco_true_pos)
    operate.init_markers(aruco_true_pos)
    print(len(operate.ekf.taglist))

    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()




