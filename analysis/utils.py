import torch
from tqdm import tqdm
import numpy as np

def generate_detect_sg(det_result, det_info, vg_dict, obj_thres = 0.1):
    num_img = len(det_info)
    groundtruths = det_result['groundtruths']
    predictions = det_result['predictions']
    assert len(groundtruths) == num_img
    assert len(predictions) == num_img

    output = {}
    for i in tqdm(range(num_img), desc="Generating scene graphs "):
        # load detect result
        image_id = det_info[i]['img_file'].split('/')[-1].split('.')[0]
        
        all_obj_labels = predictions[i].get_field('pred_labels')
        all_obj_scores = predictions[i].get_field('pred_scores')
        all_rel_pairs = predictions[i].get_field('rel_pair_idxs')
        all_rel_prob = predictions[i].get_field('pred_rel_scores')
        all_rel_scores, all_rel_labels = all_rel_prob.max(-1)

        # filter objects and relationships
        all_obj_scores[all_obj_scores < obj_thres] = 0.0
        obj_mask = all_obj_scores >= obj_thres
        triplet_score = all_obj_scores[all_rel_pairs[:, 0]] * all_obj_scores[all_rel_pairs[:, 1]] * all_rel_scores
        rel_mask = ((all_rel_labels > 0) + (triplet_score > 0)) > 0

        # generate filterred result
        num_obj = obj_mask.shape[0]
        num_rel = rel_mask.shape[0]
        rel_matrix = torch.zeros((num_obj, num_obj))
        triplet_scores_matrix = torch.zeros((num_obj, num_obj))
        for k in range(num_rel):
            if rel_mask[k]:
                rel_matrix[int(all_rel_pairs[k, 0]), int(all_rel_pairs[k, 1])], triplet_scores_matrix[int(all_rel_pairs[k, 0]), int(all_rel_pairs[k, 1])] = all_rel_labels[k], triplet_score[k]
        rel_matrix = rel_matrix[obj_mask][:, obj_mask].long()
        triplet_scores_matrix = triplet_scores_matrix[obj_mask][:, obj_mask].float()
        filter_obj = all_obj_labels[obj_mask]
        filter_pair = torch.nonzero(rel_matrix > 0)
        filter_rel = rel_matrix[filter_pair[:, 0], filter_pair[:, 1]]
        filter_scores = triplet_scores_matrix[filter_pair[:, 0], filter_pair[:, 1]]
        # assert that filter_rel and filter_scores are same shape:
        assert(filter_rel.size() == filter_scores.size())
        # generate labels
        pred_objs = [vg_dict['idx_to_label'][str(i)] for i in filter_obj.tolist()]
        pred_rels = [[i[0], i[1], vg_dict['idx_to_predicate'][str(j)], s] for i, j, s in zip(filter_pair.tolist(), filter_rel.tolist(), filter_scores.tolist())]

        output[str(image_id)] = [{'entities' : pred_objs, 'relations' : pred_rels}, ]
    return output

def compute_metrics(det_result, mode):

    for k, v in det_result[mode + '_recall'].items():
        result_str += '    R @ %d: %.4f; ' % (k, np.mean(v))
    result_str += ' for mode=%s, type=Recall(Main).' % mode
    result_str += '\n'

