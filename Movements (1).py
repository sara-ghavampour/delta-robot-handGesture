import cv2
import time
import HandTrackingModule as HandTrack
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from numba import vectorize
# import trajectory, simulation, runOnDelta


# Check If The Hand Is Above Center
def in_zone_up():
    return landmarks_list[21][2] < center_y - 5


# Check If The Hand Is Below Center
def in_zone_down():
    return landmarks_list[21][2] > center_y + 5


# Check If The Hand Is In Right Area Of The Center
def in_zone_right():
    return landmarks_list[21][1] > center_x + 5


# Check If The Hand Is In Left Area Of The Center
def in_zone_left():
    return landmarks_list[21][1] < center_x - 5


# Return Hand's Distance From The Center (Landmark 9 - Middle Finger MCP)
def distance():
    return math.hypot((center_x - landmarks_list[9][1]), (center_y - landmarks_list[9][2]))


@vectorize()
# Calculate The Angle Between 3 Points In Radian (Current Position, Previous Position, Previous Position + 1)
def get_angle(a_x, a_y, b_x, b_y, c_x, c_y):
    ab_x = b_x - a_x
    ab_y = b_y - a_y
    cb_x = b_x - c_x
    cb_y = b_y - c_y
    dot = (ab_x * cb_x + ab_y * cb_y)
    cross = (ab_x * cb_y - ab_y * cb_x)

    alpha = math.atan2(cross, dot)

    return alpha


@vectorize()
# Converting Radian To Degree
def radian_to_degree(radian):
    return int(radian * 180. / math.pi + 0.5)


# Putting A Message On The Webcam Window
def status(message, x, color=(0, 255, 0)):
    if draw_flag:
        cv2.rectangle(image, (center_x - 301, center_y + 300), (center_x + x - 301, center_y + 330), color, -1)
        cv2.putText(image, message, (center_x - 300, center_y + 320), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)


# Returning Finger's Angle Between Current And Previous Position And The Area
def figure_finger_area(lm_id):
    theta = radian_to_degree(
        get_angle(landmarks_list[lm_id][1], landmarks_list[lm_id][2], previous_landmarks_list[lm_id][1],
                  previous_landmarks_list[lm_id][2], previous_landmarks_list[lm_id][1] + 0.1,
                  previous_landmarks_list[lm_id][2]))

    if in_zone_up() and in_zone_right() and 22.5 < theta < 67.5:
        return 1, 1, 1
    elif in_zone_up() and 67.5 < theta < 112.5:
        return 2, 0, 1
    elif in_zone_up() and in_zone_left() and 112.5 < theta < 157.5:
        return 3, -1, 1
    elif in_zone_left():
        if theta > 157.5 or theta < -157.5:
            return 4, -1, 0
    elif in_zone_down() and in_zone_left() and -157.5 < theta < -112.5:
        return 5, -1, -1
    elif in_zone_down() and -112.5 < theta < -67.5:
        return 6, 0, -1
    elif in_zone_down() and in_zone_right() and -67.5 < theta < -22.5:
        return 7, 1, -1
    elif in_zone_right() and -22.5 < theta < 22.5:
        return 8, 1, 0

    return 0, 0, 0


# Returning Hand's Angle Between Current And Previous Position And The Area
def get_area_and_values():
    # For Returning Hand's Angle, Calculate Average Angle Of Five Finger(Landmarks 4, 8, 12, 16, 20) Tips And Wrist(landmark 0)
    area, axis1, axis2 = figure_finger_area(0)

    for lm_id in range(5):
        area_temp, axis1_temp, axis2_temp = figure_finger_area((lm_id + 1) * 4)

        if area_temp != area:
            return 0, 0, 0

    return area, axis1, axis2


# Check If A Finger Is Closed By Checking If Finger's Tip Is Lower Than It's MPC
def closed_finger(finger_tip):
    return landmarks_list[finger_tip - 2][2] < landmarks_list[finger_tip][2]


# Checking If The Hand Is In Stop Position (Fist) By Checking If All Fingers Are Closed
def stop():
    return closed_finger(8) and closed_finger(12) and closed_finger(16) and closed_finger(20)


# Checking If The Finger Tip Is Lower Than The Wrist
def finger_down(finger_tip):
    return landmarks_list[0][2] < landmarks_list[finger_tip][2]


# checking If The Hand Is In Grab Position By Checking If All Fingers Are Down
def grab():
    return finger_down(4) and finger_down(8) and finger_down(12) and finger_down(16) and finger_down(20)


def one():
    return not closed_finger(8) and closed_finger(12) and closed_finger(16) and closed_finger(20)


def two():
    return not closed_finger(8) and not closed_finger(12) and closed_finger(16) and closed_finger(20)


# Checking The Move That The Hand Is Making And Saving Its Name And Angle And R And Z In A CSV File
def motions():

    global x_y_flag
    global y_z_flag
    global x_z_flag

    area, axis1, axis2 = get_area_and_values()

    # if stop():
    #     quit()
    if x_y_flag and not y_z_flag and not x_z_flag:
        if area == 1:
            status("x+ & y+", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, axis2, 0, "x+ & y+"])
            return time.time() / 1000000, axis1, axis2, 0, "x+ & y+"
        elif area == 2:
            status("y+", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, axis2, 0, "y+"])
            return time.time() / 1000000, axis1, axis2, 0, "y+"
        elif area == 3:
            status("x- & y+", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, axis2, 0, "x- & y+"])
            return time.time() / 1000000, axis1, axis2, 0, "x- & y+"
        elif area == 4:
            status("x-", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, axis2, 0, "x-"])
            return time.time() / 1000000, axis1, axis2, 0, "x-"
        elif area == 5:
            status("x- & y-", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, axis2, 0, "x- & y-"])
            return time.time() / 1000000, axis1, axis2, 0, "x- & y-"
        elif area == 6:
            status("y-", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, axis2, 0, "y-"])
            return time.time() / 1000000, axis1, axis2, 0, "y-"
        elif area == 7:
            status("x+ & y-", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, axis2, 0, "x+ & y-"])
            return time.time() / 1000000, axis1, axis2, 0, "x+ & y-"
        elif area == 8:
            status("x+", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, axis2, 0, "x+"])
            return time.time() / 1000000, axis1, axis2, 0, "x+"
    if not x_y_flag and y_z_flag and not x_z_flag:
        if area == 1:
            status("y+ & z+", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, 0, axis1, axis2, "y+ & z+"])
            return time.time() / 1000000, 0, axis1, axis2, "y+ & z+"
        elif area == 2:
            status("z+", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, 0, axis1, axis2, "z+"])
            return time.time() / 1000000, 0, axis1, axis2, "z+"
        elif area == 3:
            status("y- & z+", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, 0, axis1, axis2, "y- & z+"])
            return time.time() / 1000000, 0, axis1, axis2, "y- & z+"
        elif area == 4:
            status("y-", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, 0, axis1, axis2, "y-"])
            return time.time() / 1000000, 0, axis1, axis2, "y-"
        elif area == 5:
            status("y- & z-", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, 0, axis1, axis2, "y- & z-"])
            return time.time() / 1000000, 0, axis1, axis2, "y- & z-"
        elif area == 6:
            status("z-", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, 0, axis1, axis2, "z-"])
            return time.time() / 1000000, 0, axis1, axis2, "z-"
        elif area == 7:
            status("y+ & z-", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, 0, axis1, axis2, "y+ & z-"])
            return time.time() / 1000000, 0, axis1, axis2, "y+ & z-"
        elif area == 8:
            status("y+", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, 0, axis1, axis2, "y+"])
            return time.time() / 1000000, 0, axis1, axis2, "y+"
    if not x_y_flag and not y_z_flag and x_z_flag:
        if area == 1:
            status("x+ & z+", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, 0, axis2, "x+ & z+"])
            return time.time() / 1000000, axis1, 0, axis2, "x+ & z+"
        elif area == 2:
            status("z+", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, 0, axis2, "z+"])
            return time.time() / 1000000, axis1, 0, axis2, "z+"
        elif area == 3:
            status("x- & z+", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, 0, axis2, "x- & z+"])
            return time.time() / 1000000, axis1, 0, axis2, "x- & z+"
        elif area == 4:
            status("x-", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, 0, axis2, "x-"])
            return time.time() / 1000000, axis1, 0, axis2, "x-"
        elif area == 5:
            status("x- & z-", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, 0, axis2, "x- & z-"])
            return time.time() / 1000000, axis1, 0, axis2, "x- & z-"
        elif area == 6:
            status("z-", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, 0, axis2, "z-"])
            return time.time() / 1000000, axis1, 0, axis2, "z-"
        elif area == 7:
            status("x+ & z-", 110)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, 0, axis2, "x+ & z-"])
            return time.time() / 1000000, axis1, 0, axis2, "x+ & z-"
        elif area == 8:
            status("x+", 40)
            data_time.append(cTime - pTime)
            data_z.append(0)
            writer.writerow([time.time() / 1000000, axis1, 0, axis2, "x+"])
            return time.time() / 1000000, axis1, 0, axis2, "x+"

    return 0, 0, 0, 0, 0


# Checking If The Hand Is In The Rectangle
def in_zone():
    counter = 0
    for lm in range(6):
        if (center_x - 300) < landmarks_list[lm * 4][1] < (center_x + 300) and (center_y - 300) < \
                landmarks_list[lm * 4][2] < (center_y + 300):
            counter += 1

    return counter == 6


# Checking If The Hand Is In The Circle
def in_circle():
    return (landmarks_list[9][1] - 900) ** 2 + (landmarks_list[9][2] - 360) ** 2 <= 22500


# Checking If The Index Finger's Tip Is In Plot Button's Area
def in_plot_circle():
    return (landmarks_list[8][1] - 70) ** 2 + (landmarks_list[8][2] - 110) ** 2 <= 625


# Checking If The Index Finger's Tip Is In Plot Button's Area
def in_x_y_circle():
    return (landmarks_list[8][1] - center_x + 330) ** 2 + (landmarks_list[8][2] - center_y + 275) ** 2 <= 400


# Checking If The Index Finger's Tip Is In Plot Button's Area
def in_y_z_circle():
    return (landmarks_list[8][1] - center_x + 330) ** 2 + (landmarks_list[8][2] - center_y + 190) ** 2 <= 400


# Checking If The Index Finger's Tip Is In Plot Button's Area
def in_x_z_circle():
    return (landmarks_list[8][1] - center_x + 330) ** 2 + (landmarks_list[8][2] - center_y + 105) ** 2 <= 400


# Drawing Rectangle With Its Color Given And The Plot Button
def draw_rectangle(rec_color, state_two=False):
    if draw_flag:
        cv2.rectangle(image, (center_x - 300, center_y - 300), (center_x + 300, center_y + 300), rec_color, 2)

        cv2.circle(image, (70, 110), 25, (128, 0, 0), cv2.FILLED)
        cv2.putText(image, "Tap to plot!", (5, 160), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 0, 0), 1)

        if not x_y_flag:
            cv2.circle(image, (center_x - 330, center_y - 275), 20, (128, 0, 0), cv2.FILLED)
            cv2.putText(image, "X_Y", (center_x - 355, center_y - 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 0, 0), 1)
        else:
            cv2.circle(image, (center_x - 330, center_y - 275), 20, (255, 0, 255), cv2.FILLED)
            cv2.putText(image, "X_Y", (center_x - 355, center_y - 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 1)

        if not y_z_flag:
            cv2.circle(image, (center_x - 330, center_y - 190), 20, (128, 0, 0), cv2.FILLED)
            cv2.putText(image, "Y_Z", (center_x - 355, center_y - 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 0, 0), 1)
        else:
            cv2.circle(image, (center_x - 330, center_y - 190), 20, (255, 0, 255), cv2.FILLED)
            cv2.putText(image, "Y_Z", (center_x - 355, center_y - 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 1)

        if not x_z_flag:
            cv2.circle(image, (center_x - 330, center_y - 105), 20, (128, 0, 0), cv2.FILLED)
            cv2.putText(image, "X_Z", (center_x - 355, center_y - 65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 0, 0), 1)
        else:
            cv2.circle(image, (center_x - 330, center_y - 105), 20, (255, 0, 255), cv2.FILLED)
            cv2.putText(image, "X_Z", (center_x - 355, center_y - 65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 1)

        if rec_color == (0, 0, 155):
            status("Place Your Hand Here!", 289, color=rec_color)
        if rec_color == (0, 255, 0):
            cv2.arrowedLine(image, (center_x, center_y + 300), (center_x, center_y - 300), (0, 0, 0), 1, 1, 0, 0.03)
            cv2.arrowedLine(image, (center_x - 300, center_y), (center_x + 300, center_y), (0, 0, 0), 1, 8, 0, 0.03)
            cv2.line(image, (center_x - 300, center_y + 300), (center_x + 300, center_y - 300), (129, 133, 137), 1)
            cv2.line(image, (center_x - 300, center_y - 300), (center_x + 300, center_y + 300), (129, 133, 137), 1)
            cv2.circle(image, (landmarks_list[21][1], landmarks_list[21][2]), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (center_x, center_y), 6, (0, 0, 0), cv2.FILLED)



# packet_max = 1000000
# packet_counter = 1
# flag_theta = True
#
#
# def simulate_bot_with_hand_guesture(time, x, y, z, name):
#     bot = simulation.SimulatedDeltaBot(servo_link_length=270, parallel_link_length=740,
#                                        servo_displacement=172.03559556, effector_displacement=42.2945773)
#     global current_pos_x
#     global current_pos_y
#     global current_pos_z
#     global flag_theta
#     global packet_max
#     global packet_counter
#
#     if packet_counter > 1:
#         packet_counter += 1
#         return current_pos_x, current_pos_y, current_pos_z
#
#     if packet_counter == packet_max:
#         packet_counter = 1
#
#     next_x = current_pos_x + x * 0.01
#     next_y = current_pos_y + y * 0.01
#     next_z = current_pos_z + z * 0.01
#
#     if -0.22 < next_x < 0.20 and -0.22 < next_y < 0.22 and -0.72 < next_z < -0.60 and (x != 0 or y != 0 or z != 0):
#         q, v, a, j = trajectory.LIN(current_pos_x, current_pos_y, current_pos_z, 0, next_x, next_y, next_z, 0,
#                                     max_speed, max_acc, max_jerk, 0)
#         theta, forwarded_points = bot.to_theta(q)
#         print(f"next: {next_x} {next_y} {next_z}")
#         # print(forwarded_points)
#         current_pos_x, current_pos_y, current_pos_z = next_x, next_y, next_z
#         print(theta)
#
#         runOnDelta.run_on_delta_single(theta)
#
#     elif x == 0 and y == 0 and z == 0:
#         print("#########OUT OF RANGE##############")
#     return current_pos_x, current_pos_y, current_pos_z


# Creating Previous Time And Current Time For Creating FPS Later
pTime = 0
cTime = 0
fps = 0

# Opening Webcam And Setting Its Height And Width
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandTrack.HandDetector()
previous_landmarks_list = []

# Using Run Flag To Check If The Center Has Been Formed (Its Value Will Be True After Forming Center And Never Become False Again)
run_flag = False
# Using Plot Flag To Check If The Index Finger's Tip Is In Plot Button's Area To Open Plot Only Once (It's Value Will Be True When The Hand Gets Back In The Triangle Area)
plot_flag = False
# Using Draw Flag To Show GUI On Webcam Window
draw_flag = True

x_y_flag = False
y_z_flag = False
x_z_flag = False

# current_pos_x = 0
# current_pos_y = 0
# current_pos_z = -0.62
#
# max_speed = 0.2
# max_acc = 10
# max_jerk = 500

# Forming Black Window To Be Able To Erase Drawn Objects
black_window = np.zeros((720, 1280, 3), np.uint8)

# Creating CSV File For Saving Movements Data
file = open(time.strftime("%H_%M_%S_%b_%d_%Y") + '.csv', 'w', encoding='UTF8')
writer = csv.writer(file)
writer.writerow(['time', 'x', 'y', 'z', 'type', 'speed'])

# Using Data Time And Data Z Arrays To Save Movements Data For Plotting
data_time = []
data_z = []

loop_counter = 1

while True:
    success, flipped_image = cap.read()
    image = cv2.flip(flipped_image, 1)

    image = detector.find_hands(image)

    landmarks_list = detector.find_position(image)
    if len(landmarks_list) != 0:
        dis_9_0 = int(math.hypot((landmarks_list[9][1] - landmarks_list[0][1]),
                                 (landmarks_list[9][2] - landmarks_list[0][2])) / 3)
        landmarks_list.append([21, landmarks_list[9][1], landmarks_list[9][2] + dis_9_0])

    # Drawing Red Circle If The Center Was Not Formed
    if not run_flag:
        cv2.circle(black_window, (900, 360), 150, (0, 0, 255), 4)
    # Removing Circle If The Center Has Been Formed
    else:
        cv2.circle(black_window, (900, 360), 150, (0, 0, 0), 4)

    # Giving Previous Landmarks List Its First Value
    if detector.results.multi_hand_landmarks:
        if len(previous_landmarks_list) == 0:
            previous_landmarks_list = landmarks_list

    if len(landmarks_list) != 0:
        # Setting Center's X And Y And Height
        if not run_flag:
            if in_circle():
                cv2.circle(black_window, (900, 360), 150, (0, 255, 0), 4)
                detector.find_hands(image, draw=True)
                if not stop():
                    center_x = landmarks_list[9][1]
                    center_y = landmarks_list[9][2]
                    center_height = math.hypot((landmarks_list[5][1] - landmarks_list[0][1]),
                                               (landmarks_list[5][2] - landmarks_list[0][2]))

                    run_flag = True

        # Detecting Motions And Updating Previous Landmarks List
        if run_flag and in_zone():
            detector.find_hands(image, draw=True)

            draw_rectangle((0, 255, 0))

            motions()

            # t_t, x, y, z, name = motions()
            # c_x, c_y, c_z = simulate_bot_with_hand_guesture(time, x, y, z, name)

            # cv2.putText(image, "x: " + str(c_x), (5, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 1)
            # cv2.putText(image, "y: " + str(c_y), (5, 230), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 1)
            # cv2.putText(image, "z: " + str(c_z), (5, 260), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 1)

            previous_landmarks_list = landmarks_list

            plot_flag = True

        # Drawing Plot
        if run_flag and not in_zone():
            detector.find_hands(image)

            draw_rectangle((0, 0, 155))
            # cv2.putText(image, "x: " + str(current_pos_x), (5, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 1)
            # cv2.putText(image, "y: " + str(current_pos_y), (5, 230), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 1)
            # cv2.putText(image, "z: " + str(current_pos_z), (5, 260), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 1)

            if in_plot_circle() and plot_flag:
                plt.scatter(data_time, data_z)

                plt.xlabel('time')
                plt.ylabel('z')
                plt.title('Data')

                plt.show()

                plot_flag = False
            elif in_x_y_circle():
                x_y_flag = True
                y_z_flag = False
                x_z_flag = False
            elif in_y_z_circle():
                x_y_flag = False
                y_z_flag = True
                x_z_flag = False
            elif in_x_z_circle():
                x_y_flag = False
                y_z_flag = False
                x_z_flag = True
    elif run_flag:
        draw_rectangle((0, 0, 155))
        # cv2.putText(image, "x: " + str(current_pos_x), (5, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 1)
        # cv2.putText(image, "y: " + str(current_pos_y), (5, 230), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 1)
        # cv2.putText(image, "z: " + str(current_pos_z), (5, 260), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 1)

    # Creating FPS And Putting It On Webcam Window
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    if draw_flag:
        cv2.putText(image, "fps: " + str(int(fps)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 1)

    # Showing Black And Webcam Window
    img_gray = cv2.cvtColor(black_window, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, img_inv)
    image = cv2.bitwise_or(image, black_window)
    cv2.imshow("Image", image)
    cv2.waitKey(1)

    # loop_counter += 1
    # print(f"LOOP_COUNTER: {loop_counter}")
