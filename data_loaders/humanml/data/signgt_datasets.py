import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
# import spacy
import pandas as pd
import pickle as pkl

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import BertWordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
import utils.rotation_conversions as geometry


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetGT(data.Dataset):
    def __init__(self, opt, data_file, split, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        if self.opt.fixed_len > 0:
            self.max_length = self.opt.fixed_len
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        save_raw = False
        load_direct = True
        data_type = 'pt'
        split_file = data_file[data_file['split'] == split]
        id_list = split_file['vid'].tolist()


        new_name_list = []
        length_list = []
        samples_pose = []
        sample_texts = []

        if not load_direct:
            for name in tqdm(id_list):
                try:
                    if data_type == "npz":
                        frames = sorted(os.listdir(os.path.join(opt.motion_dir, name, "smplx_params")))
                        for index, frame in enumerate(frames):
                            frame_dict= np.load(os.path.join(opt.motion_dir, name, "smplx_params", frame), allow_pickle=True)                       
                            # parse the npz files
                            root_pose= frame_dict['root_pose'] #(1,3)
                            body_pose=frame_dict['body_pose'].reshape(21, 3)   # 21
                            lhand_pose=frame_dict['lhand_pose'].reshape(15, 3)  #15
                            rhand_pose=frame_dict['rhand_pose'].reshape(15, 3) #15
                            jaw_pose=frame_dict['jaw_pose'] # (1.3)

                            full_pose=np.vstack((root_pose,body_pose,lhand_pose,rhand_pose,jaw_pose))   #(53,3)
                            full_pose = np.expand_dims(full_pose, axis=0)   #(1,53,3)  numpy
                    
                            if index == 0:                       
                                motion = full_pose                           
                            else:    
                                motion = np.vstack([motion, full_pose]) 
                    elif data_type == 'pt':
                        motion = torch.load(os.path.join(opt.motion_dir, name + '.pt'))
                        
                    motion = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(motion))
                    samples_pose.append(motion)

                    if (len(motion)) < min_motion_len or (len(motion) >= 600):
                        continue

                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip()
                            caption = line_split
                            # tokens = line_split[1].split(' ')
                            f_tag = 0.0
                            to_tag = 0.0
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = caption.split(" ")

                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                                sample_texts.append(text_data)
                            else:
                                # Not needed 
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                           'length': len(n_motion),
                                                           'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    if flag:
                        data_dict[name] = {'motion': motion,
                                           'length': len(motion),
                                           'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except:
                    print("in Except")
                    pass

            if save_raw:
                save_pkl = self.opt.save_pkl    # '/home/exx/Akib/motion-diffusion-model/data-bin/openasl'
                save_poses = os.path.join(save_pkl, split, 'samples.35k.pkl')

                with open(save_poses, "wb") as fs:
                    pkl.dump({'name_list': new_name_list,
                        'length_list': length_list,
                        'data_dict': data_dict},fs)
                    print(f"save all {split } sampls pose succ!")
        else:
            save_pkl = self.opt.save_pkl    #'/home/exx/Akib/motion-diffusion-model/data-bin/openasl'
            load_poses = os.path.join(save_pkl, split, 'samples.35k.pkl')
            with open(load_poses, "rb") as fs:
                data_loaded = pkl.load(fs)

            data_dict = data_loaded['data_dict']
            new_name_list, length_list = data_loaded['name_list'], data_loaded['length_list']
            # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            print("==================================================")
            print(len(data_dict))
            print("==================================================")
            print("datset loded")

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        key = self.name_list[idx]
        data = self.data_dict[key]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        # text_data = random.choice(text_list)
        caption, tokens = text_list[0]['caption'], text_list[0]['tokens']
        sent_len = len(tokens)

        word_embeddings = self.w_vectorizer(caption)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        
        original_length = None
        if self.opt.fixed_len > 0:
            # Crop fixed_len
            original_length = m_length
            m_length = self.opt.fixed_len
        
        idx = random.randint(0, len(motion) - m_length)
        if self.opt.disable_offset_aug:
            idx = random.randint(0, self.opt.unit_length)
        motion = motion[idx:idx+m_length]
        _, _, n_feat = motion.shape

        motion_pose = motion.reshape(motion.shape[0], motion.shape[1] * n_feat)

        length = (original_length, m_length) if self.opt.fixed_len > 0 else m_length
        caption = "[CLS]" + caption + "[SEP]"
    

        return word_embeddings, None, caption, sent_len, motion_pose, length, '_'.join(tokens)


# A wrapper class for t2m original dataset for MDM purposes
class OpenASLGT(data.Dataset):
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train", **kwargs):
        self.mode = mode

        self.dataset_name = 'openasl'
        self.dataname = 'openasl'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = kwargs.get('abs_path', '.')
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = kwargs.get('device', None)
        opt = get_opt(dataset_opt_path, device)
        # opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.cache_dir = kwargs.get('cache_path', '.')
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = pjoin(abs_base_path, './dataset')
        opt.use_cache = kwargs.get('use_cache', True)
        opt.fixed_len = kwargs.get('fixed_len', 0)
        if opt.fixed_len > 0:
            opt.max_motion_length = opt.fixed_len
        is_autoregressive = kwargs.get('autoregressive', False)
        opt.disable_offset_aug = is_autoregressive and (opt.fixed_len > 0) and (mode == 'eval')  # for autoregressive evaluation, use the start of the motion and not something from the middle
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)


        data = pd.read_csv(pjoin(opt.data_root, 'meta', 'top35k_unique.tsv'), sep="\t")
       
        self.w_vectorizer = BertWordVectorizer()
        self.t2m_dataset = Text2MotionDatasetGT(self.opt, data, split, self.w_vectorizer)
        self.num_actions = 1 # dummy placeholder


        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()