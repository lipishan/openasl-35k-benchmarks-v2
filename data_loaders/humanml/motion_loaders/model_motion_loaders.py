from torch.utils.data import DataLoader, Dataset
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.motion_loaders.comp_v6_model_dataset import CompMDMGeneratedDataset
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
import numpy as np
from torch.utils.data._utils.collate import default_collate
import torch


# def collate_fn(batch):
#     batch.sort(key=lambda x: x[3], reverse=True)
#     return default_collate(batch)

def collate_signs_fn(data_list):
    """Convert a list of 2d tensors into a padded 3d tensor.

    input is a list of k_i * 1024 features.
    We need to find the max K and pad all others to have the same length.
    Return NUM_SAMPLES * K * 1024 tensor.
    adapted from: https://github.com/verashira/TSPNet
    """
    word_embedds, _, caption, sent_len, values, m_length, tokens = zip(*data_list)
    lens = [len(v) for v in values]
    word_lens = [len(v) for v in word_embedds]
    
    min_len = min(lens)
    min_word_len = min(word_lens)
    max_word_len = max(word_lens)

    if max(lens) % 4 == 0:
        max_len = max(lens)
    else :
        add_n = 4 - (max(lens) % 4)
        max_len = max(lens) + add_n


    assert min_len > 0
    assert min_word_len > 0

    res = np.zeros((len(values), max_len, values[0].shape[1]))
    word_res = torch.zeros(size=(len(word_embedds), max_word_len, word_embedds[0].shape[1]))

    for i, v in enumerate(values):
        res[i][max_len - len(v):] = v

    for i, v in enumerate(word_embedds):
        word_res[i][max_word_len - len(v):] = v
   
    return word_res, caption, np.array(sent_len), torch.from_numpy(res), np.array(m_length), tokens

class MMGeneratedDataset(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            # We don't need the following logic because our sample func generates the full tensor anyway:
            # if len(motion) < self.opt.max_motion_length:
            #     motion = np.concatenate([motion,
            #                              np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
            #                              ], axis=0)
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens



def get_motion_loader(opt_path, batch_size, ground_truth_dataset, mm_num_samples, mm_num_repeats, device):
    opt = get_opt(opt_path, device)

    # Currently the configurations of two datasets are almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
    else:
        raise KeyError('Dataset not recognized!!')
    print('Generating %s ...' % opt.name)

    if 'v6' in opt.name:
        dataset = CompV6GeneratedDataset(opt, ground_truth_dataset, w_vectorizer, mm_num_samples, mm_num_repeats)
    else:
        raise KeyError('Dataset not recognized!!')

    mm_dataset = MMGeneratedDataset(opt, dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_signs_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader

# our loader
def get_mdm_loader(args, model, diffusion, batch_size, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale):
    opt = {
        'name': 'test',  # FIXME
    }
    print('Generating %s ...' % opt['name'])
    # dataset = CompMDMGeneratedDataset(opt, ground_truth_dataset, ground_truth_dataset.w_vectorizer, mm_num_samples, mm_num_repeats)
    dataset = CompMDMGeneratedDataset(args, model, diffusion, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale)

    mm_dataset = MMGeneratedDataset(opt, dataset, ground_truth_loader.dataset.w_vectorizer)

    # NOTE: bs must not be changed! this will cause a bug in R precision calc!
    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_signs_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader