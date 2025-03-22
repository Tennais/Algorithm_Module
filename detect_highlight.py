import os, sys
sys.path.append(".")
from tqdm import tqdm
import argparse
import json
import shutil
import numpy as np
import copy
from glob import glob
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors

colors = Colors()

def parse_args():
    parser = argparse.ArgumentParser(description="ExtractFrame")
    parser.add_argument("--image_dir", default="/hetu_group/fandewen/code/Tennis/data/video_frame_10fps/", type=str, required=False)
    parser.add_argument("--output_dir", default="/hetu_group/fandewen/code/Tennis/高亮时刻_架拍", type=str, required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    frame_height, frame_width = 1080, 1920
    line_thickness = 2
    tl = line_thickness or round(0.002 * (frame_height + frame_width) / 2) + 1
    tf = max(tl - 1, 1)
    sub_image_dirs = glob(f"{args.image_dir}/*")
    for sub_image_dir in tqdm(sub_image_dirs):
        if not os.path.isdir(sub_image_dir):
            continue
        sub_dir_name = os.path.basename(sub_image_dir)
        video_info_file = os.path.join(args.image_dir, f"{sub_dir_name}_info.json") 
        video_info = json.load(open(video_info_file))
        visual_idr = os.path.join(args.output_dir, sub_dir_name)
        if os.path.exists(visual_idr):
            shutil.rmtree(visual_idr)
        os.makedirs(visual_idr, exist_ok=True)
        image_paths = glob(f"{sub_image_dir}/*.jpg")
        image_paths = sorted(image_paths, key=lambda x: int(x.split("/")[-1].split("_")[1]))

        sequential_keypoints = []
        sequential_rackets = []
        highlight_idxs = []
        for idx, image_path in enumerate(tqdm(image_paths)):
            frame = cv2.imread(image_path)
            if frame.shape != (frame_height, frame_width, 3):
                frame = cv2.resize(frame, (frame_width, frame_height))
            base_name = os.path.basename(image_path)
            annotated_frame = copy.deepcopy(frame)
            if base_name not in video_info:
                continue
            poses = video_info[base_name]["poses"]
            color = colors(int(0))
            for kp in poses:
                if kp[2] > 0.5:
                    cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 3, color, -1)
            pose_connections = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,13),(13,15), (12,14), (14,16)]
            for connection in pose_connections:
                if (poses[connection[0]][2] > 0.5 and 
                    poses[connection[1]][2] > 0.5):
                    pt1 = (int(poses[connection[0]][0]), int(poses[connection[0]][1]))
                    pt2 = (int(poses[connection[1]][0]), int(poses[connection[1]][1]))
                    cv2.line(annotated_frame, pt1, pt2, color, 2)

            racket_score, racket_coords = video_info[base_name]["racket"]
            
            if racket_score > 0.5: # Racket detection
                color = colors(int(3))
                cv2.rectangle(annotated_frame, (int(racket_coords[0]), int(racket_coords[1])), (int(racket_coords[2]), int(racket_coords[3])), color, thickness=line_thickness, lineType=cv2.LINE_AA)
                label = f"Racket:{racket_score}"
                c1, c2 = (int(racket_coords[0]), int(racket_coords[1])), (int(racket_coords[2]), int(racket_coords[3]))
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(annotated_frame, c1, c2, color, -1, cv2.LINE_AA)
                cv2.putText(annotated_frame, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            
            sequential_keypoints.append(poses)
            sequential_rackets.append([racket_score, racket_coords])

            # 正手击球 1.右臂达到最长距离 2.球拍平行于地面
            forehand_flat = False
            # 架拍 1.左臂伸直 2.右臂弯曲 3.球拍呈竖直方向 4.肩部旋转 5.双脚间有一定距离
            raising_racket = False
            # 随挥 1.右臂围绕到身体左侧 2.球拍位于身体左侧，并指向后方
            casually_swing = False

            sequence_len = 3
            hl_seq_idx = -2
            if len(sequential_keypoints) >= sequence_len:
                right_arm_visible = True
                for i in range(hl_seq_idx, -sequence_len-1, -1):
                    right_arm_visible = right_arm_visible and float(sequential_keypoints[i][6][2]) > 0.5 and float(sequential_keypoints[i][8][2]) > 0.5 and float(sequential_keypoints[i][10][2]) > 0.5
                left_arm_visible = True
                for i in range(-1, -sequence_len-1, -1):
                    left_arm_visible = left_arm_visible and float(sequential_keypoints[i][5][2]) > 0.5 and float(sequential_keypoints[i][7][2]) > 0.5 and float(sequential_keypoints[i][9][2]) > 0.5

                dis1_x = int(sequential_keypoints[hl_seq_idx][8][0] - sequential_keypoints[hl_seq_idx][6][0])
                dis1_y = int(abs(sequential_keypoints[hl_seq_idx][6][1] - sequential_keypoints[hl_seq_idx][8][1]))
                dis2_x = int(sequential_keypoints[hl_seq_idx][10][0] - sequential_keypoints[hl_seq_idx][8][0])
                dis2_y = int(abs(sequential_keypoints[hl_seq_idx][8][1] - sequential_keypoints[hl_seq_idx][10][1]))
                right_arm_location = dis2_x >= 15 and dis2_y <= 30 and dis2_y / dis2_x <= 0.3

                right_arm_right = float(sequential_keypoints[hl_seq_idx][16][2]) > 0.5 and float(sequential_keypoints[hl_seq_idx][10][0]) > float(sequential_keypoints[hl_seq_idx][16][0])
                left_arm_left = True
                if float(sequential_keypoints[hl_seq_idx][5][2]) > 0.5 and  float(sequential_keypoints[hl_seq_idx][6][0]) - float(sequential_keypoints[hl_seq_idx][5][0]) <= 15:
                    left_arm_left = False
                if float(sequential_keypoints[hl_seq_idx][7][2]) > 0.5 and float(sequential_keypoints[hl_seq_idx][7][0]) > float(sequential_keypoints[hl_seq_idx][6][0]):
                    left_arm_left = False
                if float(sequential_keypoints[hl_seq_idx][9][2]) > 0.5 and float(sequential_keypoints[hl_seq_idx][9][0]) > float(sequential_keypoints[hl_seq_idx][6][0]):
                    left_arm_left = False

                sequence_longest_arm = sequential_keypoints[hl_seq_idx+1][10][0] < sequential_keypoints[hl_seq_idx][10][0] and sequential_keypoints[hl_seq_idx][10][0] > sequential_keypoints[hl_seq_idx-1][10][0]
                
                foot_right_y_greater_left = sequential_keypoints[hl_seq_idx][15][0] > 0.5 and sequential_keypoints[hl_seq_idx][16][0] > 0.5 and sequential_keypoints[hl_seq_idx][16][1] >= sequential_keypoints[hl_seq_idx][15][1]
                
                racket_w_greater_h = False
                racket_h_greater_w = False
                racket_center = [0, 0]
                highlight_racket_score, highlight_racket_coords = sequential_rackets[hl_seq_idx]
                if highlight_racket_score > 0.5 and highlight_racket_coords[3] > frame_height / 2:
                    racket_center =[int((highlight_racket_coords[0] + highlight_racket_coords[2]) / 2), int((highlight_racket_coords[1] + highlight_racket_coords[3]) / 2)]
                    # 相机近端
                    racket_w_greater_h = highlight_racket_coords[2] - highlight_racket_coords[0] > highlight_racket_coords[3] - highlight_racket_coords[1]
                    racket_h_greater_w = not racket_w_greater_h
                
                racket_location_forehand_flat = False
                if racket_w_greater_h and racket_center[0] > sequential_keypoints[hl_seq_idx][10][0] + 15:
                    racket_location_forehand_flat = True

                forehand_flat = right_arm_visible and right_arm_location and right_arm_right and left_arm_left and sequence_longest_arm and racket_location_forehand_flat and foot_right_y_greater_left
                
                body_rotation = False
                if sequential_keypoints[hl_seq_idx][7][2] > 0.5:
                    if sequential_keypoints[hl_seq_idx][6][0] < sequential_keypoints[hl_seq_idx][7][0] and sequential_keypoints[hl_seq_idx][8][0] < sequential_keypoints[hl_seq_idx][7][0]:
                        body_rotation = True
                if sequential_keypoints[hl_seq_idx][9][2] > 0.5:
                    if sequential_keypoints[hl_seq_idx][6][0] < sequential_keypoints[hl_seq_idx][9][0] and sequential_keypoints[hl_seq_idx][8][0] < sequential_keypoints[hl_seq_idx][9][0]:
                        body_rotation = True
                
                right_hand_right = sequential_keypoints[hl_seq_idx][10][0] > sequential_keypoints[hl_seq_idx][6][0] and sequential_keypoints[hl_seq_idx][10][0] > sequential_keypoints[hl_seq_idx][8][0]
                right_relbow_greater_hand_y = sequential_keypoints[hl_seq_idx][8][1] > sequential_keypoints[hl_seq_idx][10][1]
                right_foot_greater_left = sequential_keypoints[hl_seq_idx][16][2] > 0.5 and sequential_keypoints[hl_seq_idx][15][2] > 0.5 and sequential_keypoints[hl_seq_idx][16][0] > sequential_keypoints[hl_seq_idx][15][0] + 10

                racket_location_raising = False
                if racket_h_greater_w and racket_center[0] > sequential_keypoints[hl_seq_idx][10][0] and racket_center[1] < sequential_keypoints[hl_seq_idx][10][1]:
                    racket_location_raising = True
                
                left_arm_distance = sequential_keypoints[hl_seq_idx][5][1] - sequential_keypoints[hl_seq_idx][9][1] < 50
                right_arm_distance = sequential_keypoints[hl_seq_idx][6][1] - sequential_keypoints[hl_seq_idx][10][1] < 50

                raising_racket = right_arm_visible and body_rotation and right_hand_right and right_foot_greater_left and racket_location_raising and left_arm_distance and right_arm_distance and right_relbow_greater_hand_y and foot_right_y_greater_left

                left_right_shoulder = sequential_keypoints[hl_seq_idx][6][2] > 0.5 and sequential_keypoints[hl_seq_idx][6][0] - sequential_keypoints[hl_seq_idx][5][0] >= 20 and abs(sequential_keypoints[hl_seq_idx][6][1] - sequential_keypoints[hl_seq_idx][5][1]) < 30
                right_arm_in_left = sequential_keypoints[hl_seq_idx][10][2] > 0.5 and  sequential_keypoints[hl_seq_idx][10][0] <  sequential_keypoints[hl_seq_idx][5][0]
                casually_foot_location = sequential_keypoints[hl_seq_idx][16][2] > 0.5 and sequential_keypoints[hl_seq_idx][15][2] > 0.5 and 150 > sequential_keypoints[hl_seq_idx][16][0] - sequential_keypoints[hl_seq_idx][15][0] >= 15 and abs(sequential_keypoints[hl_seq_idx][16][1] - sequential_keypoints[hl_seq_idx][15][1]) <= 15
                left_relbow_greater_hand_y = sequential_keypoints[hl_seq_idx][7][1] - sequential_keypoints[hl_seq_idx][9][1] >= -10
                
                casually_swing = left_arm_visible and casually_foot_location and left_right_shoulder and right_arm_in_left and left_relbow_greater_hand_y

            if forehand_flat or raising_racket or casually_swing:
                if highlight_idxs == [] or idx - highlight_idxs[-1] > 1:
                    highlight_base_name, highlight_annotated_frame = last_visual_image
                    highlight_visual_path = os.path.join(visual_idr, highlight_base_name)
                    cv2.imwrite(highlight_visual_path, highlight_annotated_frame)
                highlight_idxs.append(idx)

            last_visual_image = [base_name, annotated_frame]
            sequential_keypoints = sequential_keypoints[-sequence_len:]
            sequential_rackets = sequential_rackets[-sequence_len:]


if __name__ == '__main__':
    main()
