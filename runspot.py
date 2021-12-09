# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to use the Boston Dynamics API"""
from __future__ import print_function
import argparse
import sys
import time
import os
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand

# VELOCITY_BASE_SPEED = 0.5  # m/s
# VELOCITY_BASE_ANGULAR = 0.8  # rad/sec
# VELOCITY_CMD_DURATION = 0.6  # seconds
# COMMAND_INPUT_RATE = 0.1


def gesture_control(config):
    """A simple example of using the Boston Dynamics API to command a Spot robot."""

    # The Boston Dynamics Python library uses Python's logging module to
    # generate output. Applications using the library can specify how
    # the logging information should be output.
    bosdyn.client.util.setup_logging(config.verbose)

    # The SDK object is the primary entry point to the Boston Dynamics API.
    # create_standard_sdk will initialize an SDK object with typical default
    # parameters. The argument passed in is a string identifying the client.
    sdk = bosdyn.client.create_standard_sdk('HelloSpotClient')

    # A Robot object represents a single robot. Clients using the Boston
    # Dynamics API can manage multiple robots, but this tutorial limits
    # access to just one. The network address of the robot needs to be
    # specified to reach it. This can be done with a DNS name
    # (e.g. spot.intranet.example.com) or an IP literal (e.g. 10.0.63.1)
    robot = sdk.create_robot(config.hostname)

    # Clients need to authenticate to a robot before being able to use it.
    robot.authenticate(config.username, config.password)

    # Establish time sync with the robot. This kicks off a background thread to establish time sync.
    # Time sync is required to issue commands to the robot. After starting time sync thread, block
    # until sync is established.
    robot.time_sync.wait_for_sync()

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    # Only one client at a time can operate a robot. Clients acquire a lease to
    # indicate that they want to control a robot. Acquiring may fail if another
    # client is currently controlling the robot. When the client is done
    # controlling the robot, it should return the lease so other clients can
    # control it. Note that the lease is returned as the "finally" condition in this
    # try-catch-finally block.
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    lease = lease_client.acquire()
    try:
        with bosdyn.client.lease.LeaseKeepAlive(lease_client):
            # Now, we are ready to power on the robot. This call will block until the power
            # is on. Commands would fail if this did not happen. We can also check that the robot is
            # powered at any point.
            robot.logger.info("Powering on robot... This may take several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            robot.logger.info("Robot powered on.")

            #_robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

            fileHands = mp.solutions.hands
            hands = fileHands.Hands(max_num_hands=1, min_detection_confidence=0.98)
            drawHands = mp.solutions.drawing_utils
            
            gestureRecognizer = load_model("mp_hand_gesture")
            print("here")
            file = open("gesture.names", "r")
            gestureNames = file.read().split('\n')
            file.close()
            vid = cv.VideoCapture(0)

            while True:
                _, frame = vid.read()
    
                x, y, z = frame.shape
                frame = cv.flip(frame, 1)
                frameColored = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                gesturePrediction = hands.process(frameColored)
                gestureName = ""
                if (gesturePrediction.multi_hand_landmarks):
                    handLandmarks = []
                    for handDetails in gesturePrediction.multi_hand_landmarks:
                        for landmark in handDetails.landmark:
                            landmarkX = int(landmark.x * x)
                            landmarkY = int(landmark.y * y)
                            handLandmarks.append([landmarkX, landmarkY])
                        drawHands.draw_landmarks(frame, handDetails, fileHands.HAND_CONNECTIONS)
                        gesture = gestureRecognizer.predict([handLandmarks])
                        gestureIndex = np.argmax(gesture)
                        gestureName = gestureNames[gestureIndex]
                        print(gestureName)
                        if(gestureName == "walkForward"):
                            print("Forward")
                            #call function for Spot to walk forward
                        if(gestureName == "walkBackward"):
                            robot.power_off(cut_immediately=False, timeout_sec=20)
                            assert not robot.is_powered_on(), "Robot power off failed."
                            robot.logger.info("Robot safely powered off.")
                            print("Spot, shut down!")
                            #call function for Spot to walk backward
                        if(gestureName == "walkLeft"):
                            print("Spot, take a step to the left!")
                            #call function for Spot to walk left
                        if(gestureName == "walkRight"):
                            print("Spot, take a step to the right!")
                            #call function for Spot to walk right
                        if(gestureName == "turnLeft"):
                            print("Spot, turn to the left!")
                            #call function for Spot to turn left
                        if(gestureName == "turnRight"):
                            print("Spot, turn to the right!")
                            #call function for Spot to turn right
                        if(gestureName == "tiltUp"):
                            robot.logger.info("Commanding robot to stand...")
                            command_client = robot.ensure_client(RobotCommandClient.default_service_name)
                            blocking_stand(command_client, timeout_sec=10)
                            robot.logger.info("Robot standing.")
                            time.sleep(3)
                            print("Spot, stand up!")
                            #call function for Spot to tilt up
                        if(gestureName == "tiltDown"):
                            print("Spot, tilt down!")
                            #call function for Spot to tilt down
                        #potential implementation of time.sleep()
                cv.imshow("Gesture", frame) 
                if (cv.waitKey(1) == ord("x")):
                    break
            vid.release()
            cv.destroyAllWindows()

            # Tell the robot to stand up. The command service is used to issue commands to a robot.
            # The set of valid commands for a robot depends on hardware configuration. See
            # SpotCommandHelper for more detailed examples on command building. The robot
            # command service requires timesync between the robot and the client.
            

            # Tell the robot to stand in a twisted position.
            #
            # The RobotCommandBuilder constructs command messages, which are then
            # issued to the robot using "robot_command" on the command client.
            #
            # In this example, the RobotCommandBuilder generates a stand command
            # message with a non-default rotation in the footprint frame. The footprint
            # frame is a gravity aligned frame with its origin located at the geometric
            # center of the feet. The X axis of the footprint frame points forward along
            # the robot's length, the Z axis points up aligned with gravity, and the Y
            # axis is the cross-product of the two.
            footprint_R_body = bosdyn.geometry.EulerZXY(yaw=0.4, roll=0.0, pitch=0.0)
            cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body)
            command_client.robot_command(cmd)
            robot.logger.info("Robot standing twisted.")
            time.sleep(3)

            # Now tell the robot to stand taller, using the same approach of constructing
            # a command message with the RobotCommandBuilder and issuing it with
            # robot_command.
            cmd = RobotCommandBuilder.synchro_stand_command(body_height=0.1)
            command_client.robot_command(cmd)
            robot.logger.info("Robot standing tall.")
            time.sleep(3)

            # Capture an image.
            # Spot has five sensors around the body. Each sensor consists of a stereo pair and a
            # fisheye camera. The list_image_sources RPC gives a list of image sources which are
            # available to the API client. Images are captured via calls to the get_image RPC.
            # Images can be requested from multiple image sources in one call.
            image_client = robot.ensure_client(ImageClient.default_service_name)
            sources = image_client.list_image_sources()
            image_response = image_client.get_image_from_sources(['frontleft_fisheye_image'])
            #_maybe_display_image(image_response[0].shot.image)
            #if config.save or config.save_path is not None:
                #maybe_save_image(image_response[0].shot.image, config.save_path)

            # Log a comment.
            # Comments logged via this API are written to the robots test log. This is the best way
            # to mark a log as "interesting". These comments will be available to Boston Dynamics
            # devs when diagnosing customer issues.
            log_comment = "HelloSpot tutorial user comment."
            robot.operator_comment(log_comment)
            robot.logger.info('Added comment "%s" to robot log.', log_comment)

            # Power the robot off. By specifying "cut_immediately=False", a safe power off command
            # is issued to the robot. This will attempt to sit the robot before powering off.
            
    finally:
        # If we successfully acquired a lease, return it.
        lease_client.return_lease(lease)



# def _move_forward():
#         _velocity_cmd_helper('move_forward', v_x=VELOCITY_BASE_SPEED)

# def _velocity_cmd_helper(desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
#         _start_robot_command(
#             desc, RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
#             end_time_secs=time.time() + VELOCITY_CMD_DURATION)

# def _start_robot_command(self, desc, command_proto, end_time_secs=None):

#         def _start_command():
#             _robot_command_client.robot_command(lease=None, command=command_proto,
#                                                      end_time_secs=end_time_secs)

#         self._try_grpc(desc, _start_command)

# def _maybe_display_image(image, display_time=3.0):
#     """Try to display image, if client has correct deps."""
#     try:
#         from PIL import Image
#         import io
#     except ImportError:
#         logger = bosdyn.client.util.get_logger()
#         logger.warning("Missing dependencies. Can't display image.")
#         return
#     try:
#         image = Image.open(io.BytesIO(image.data))
#         image.show()
#         time.sleep(display_time)
#     except Exception as exc:
#         logger = bosdyn.client.util.get_logger()
#         logger.warning("Exception thrown displaying image. %r", exc)

# def _maybe_save_image(image, path):
#     """Try to save image, if client has correct deps."""
#     logger = bosdyn.client.util.get_logger()
#     try:
#         from PIL import Image
#         import io
#     except ImportError:
#         logger.warning("Missing dependencies. Can't save image.")
#         return
#     name = "hello-spot-img.jpg"
#     if path is not None and os.path.exists(path):
#         path = os.path.join(os.getcwd(), path)
#         name = os.path.join(path, name)
#         logger.info("Saving image to: {}".format(name))
#     else:
#         logger.info("Saving image to working directory as {}".format(name))
#     try:
#         image = Image.open(io.BytesIO(image.data))
#         image.save(name)
#     except Exception as exc:
#         logger = bosdyn.client.util.get_logger()
#         logger.warning("Exception thrown saving image. %r", exc)


def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_common_arguments(parser)
    parser.add_argument('-s', '--save', action='store_true', help='Save the image captured by Spot to the working directory. To chose the save location, use --save_path instead.')
    parser.add_argument('--save-path', default=None, nargs='?', help='Save the image captured by Spot to the provided directory. Invalid path saves to working directory.')
    options = parser.parse_args(argv)
    try:
        gesture_control(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.error("Hello, Spot! threw an exception: %r", exc)
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
