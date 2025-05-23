import argparse
import os
import shutil
import math
import numpy as np
from tqdm import tqdm
import pickle
import pdb
import torch
import torch.nn as nn
import pandas as pd

import utils.rotation_conversions as geometry


motions_dir = "/mnt/pub/sign_language_datasets/features/OpenASL/m2t2m/motions"
def loss_check(input, target):
    loss = nn.MSELoss()
    return loss(input, target)

def get_gt(args):
    return args.df.loc[args.df['tokenized-text'] == args.text, 'vid'].iloc[0]

def convert(args):
    data = np.load(args.input_path, allow_pickle=True)
    motions = torch.from_numpy(data[None][0]['motion']).permute(0, 3, 1, 2)
    return data[None][0]['text'], geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(motions))


def get_render_files(args):
    texts, data_converted = convert(args)
    files, generated, translation, scores, gt = [], [], [], [], []
    loss_score = math.inf

    for index, (txt, motion) in enumerate(zip(texts, data_converted)):
        # if index > 3:
        #     break
        args.text = txt.replace('[CLS]', '').replace('[SEP]', '') #.replace(' ', '_')
        file_name = get_gt(args)
        gt_data = torch.load(os.path.join(motions_dir, file_name + ".pt"))
        
        if len(motion) > len(gt_data):
            motion_selected = motion[:len(gt_data)]
        elif (len(gt_data) - len(motion)) < 20:
                max_len = len(gt_data)
                res = torch.zeros((max_len, motion.shape[1], motion.shape[2]))
                res[max_len - len(motion):] = motion
                motion_selected = res
        else:
            print("Skipping...", file_name, "because the length... ", len(gt_data), (len(gt_data) - len(motion)), 150)
            continue
        
        loss = loss_check(motion_selected, gt_data)
        scores.append(loss)
        files.append(file_name)
        # generated.append(motion)
        generated.append(motion_selected)
        translation.append(txt)
        gt.append(gt_data)
        
        # if loss < loss_score:
        #     loss_score = loss
        #     if index == 0:
        #         scores.append(loss_score)
        #         files.append(file_name)
        #         # generated.append(motion)
        #         generated.append(motion_selected)
        #         translation.append(txt)
        #         gt.append(gt_data)
        #     else:
        #         # sorted_list = sorted(scores)
        #         # for v, w, x, y, z in sorted(zip(scores, files, generated, translation, gt)):
        #         #     sorted_list = v
        #         #     files = w
        #         #     generated = x
        #         #     translation = y
        #         #     gt = z
        #         # if sorted_list[-1] > loss_score:
        #         #     sorted_list.pop()
        #         #     files.pop()
        #         #     generated.pop()
        #         #     translation.pop()
        #         #     gt.pop()
        #         #     #################################
        #         #     sorted_list.append(loss_score)
        #         #     files.append(file_name)
        #         #     # generated.append(motion)
        #         #     generated.append(motion_selected)
        #         #     translation.append(txt)
        #         #     gt.append(gt_data)
        #         # else:
        #             scores.append(loss_score)
        #             files.append(file_name)
        #             # generated.append(motion)
        #             generated.append(motion_selected)
        #             translation.append(txt)
        #             gt.append(gt_data)
        # else:
        #     scores.append(loss)
        #     files.append(file_name)
        #     # generated.append(motion)
        #     generated.append(motion_selected)
        #     translation.append(txt)
        #     gt.append(gt_data)
    return scores, files, generated, translation, gt



def create_render_files(args):
    scores, files, generated, translation, gt = get_render_files(args)
    # pdb.set_trace()
    for index, (_, file_name, gen_motion, text, gt_motion) in enumerate(sorted(zip(scores, files, generated, translation, gt))):
        if index > 5:
            break
        print("Index:", index, " Processing and Storing...", file_name)
        os.makedirs(os.path.join(args.output, file_name), exist_ok=True)
        # if len(gen_motion) > len(gt_motion):
        #     gen_motion = gen_motion[:len(gt_motion)]

        for sub_index, frame in tqdm(enumerate(gen_motion)):
            frame = frame.detach().cpu().numpy()
            output_path = os.path.join(args.output, file_name, str(sub_index) + ".pkl")
            global_orient = frame[0, :]
            rotations = frame[1:, :]

            with open(output_path, "wb") as f:
                pickle.dump({'global_orient': np.zeros((1, 3)), 
                            'body_pose': rotations[:21, :].reshape(1, -1), 
                            'left_hand_pose': rotations[21:36, :].reshape(1, -1), 
                            'right_hand_pose': rotations[36:51, :].reshape(1, -1), 
                            'jaw_pose': rotations[51, :].reshape(1, -1), 
                            'betas': np.zeros((1, 10)), 
                            'expression': np.zeros((1, 10)),
                            'gender': 'NEUTRAL'}, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--output", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()
    
    meta = pd.read_csv("/mnt/pub/sign_language_datasets/features/OpenASL/m2t2m/meta/top35k_unique.tsv", sep="\t")
    params.df = meta

    # get_render_files(params)
    create_render_files(params)