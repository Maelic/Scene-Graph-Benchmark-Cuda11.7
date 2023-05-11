import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import to_agraph

import os
import numpy as np
import csv
import io

from PIL import Image, ImageDraw, ImageFont
import cv2

def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1+50, y1+10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)

def get_size(image_size):
    min_size = 600
    max_size = 1000
    w, h = image_size
    size = min_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (ow, oh)

def get_graphs(boxes, box_labels, all_rel_pairs, all_rel_labels, ind_to_predicates, all_rel_scores, box_scores, rel_thres=0.5):
    cat_path = "/home/maelic/Documents/PhD/MyModel/Scene_Graphs_Visualization/relations_categories_davinci_filtered.csv"

    spatial_graph = {}
    functional_graph = {}
    partonomy_graph = {}
    attributive_graph = {}

    # read csv file in a dictionary:
    with open(cat_path, mode='r') as infile:
        reader = csv.reader(infile)
        rel_dict = {rows[0]:rows[1] for rows in reader}
    top_score = 0
    func_score = 0
    parto_score = 0
    attr_score = 0

    for i in range(len(all_rel_pairs)):
        score = all_rel_scores[i]
        if True:
            subj_label = box_labels[all_rel_pairs[i][0]]
            obj_label =  box_labels[all_rel_pairs[i][1]]
            pred_label = ind_to_predicates[str(all_rel_labels[i])]
            rel = subj_label + ' ' + pred_label + ' ' + obj_label
            try:
                rel_cat = rel_dict[rel]
            except:
                #print('Relationship not found in the csv file')
                continue
            
            subj_label = str(all_rel_pairs[i][0]) + '_' + box_labels[all_rel_pairs[i][0]]
            obj_label =  str(all_rel_pairs[i][1]) + '_' + box_labels[all_rel_pairs[i][1]]

            if rel_cat == 'topological':
                # only keep the top rel for each pair of boxes
                if (subj_label, obj_label) not in spatial_graph.keys():
                    spatial_graph[(subj_label, obj_label)] = {'subj': subj_label, 'obj': obj_label, 'pred': pred_label, 'score': score}
                    top_score = score
                elif spatial_graph[(subj_label, obj_label)]['score'] > top_score:
                    spatial_graph[(subj_label, obj_label)] = {'subj': subj_label, 'obj': obj_label, 'pred': pred_label, 'score': score}
            elif rel_cat == 'functional':
                if (subj_label, obj_label) not in functional_graph.keys():
                    functional_graph[(subj_label, obj_label)] = {'subj': subj_label, 'obj': obj_label, 'pred': pred_label, 'score': score}
                    func_score = score
                elif functional_graph[(subj_label, obj_label)]['score'] > func_score:
                    functional_graph[(subj_label, obj_label)] = {'subj': subj_label, 'obj': obj_label, 'pred': pred_label, 'score': score}
            elif rel_cat == 'partonomy':
                if (subj_label, obj_label) not in partonomy_graph.keys():
                    partonomy_graph[(subj_label, obj_label)] = {'subj': subj_label, 'obj': obj_label, 'pred': pred_label, 'score': score}
                    parto_score = score
                elif partonomy_graph[(subj_label, obj_label)]['score'] > parto_score:
                    partonomy_graph[(subj_label, obj_label)] = {'subj': subj_label, 'obj': obj_label, 'pred': pred_label, 'score': score}
            elif rel_cat == 'attributive':
                if (subj_label, obj_label) not in attributive_graph.keys():
                    attributive_graph[(subj_label, obj_label)] = {'subj': subj_label, 'obj': obj_label, 'pred': pred_label, 'score': score}
                    attr_score = score
                elif attributive_graph[(subj_label, obj_label)]['score'] > attr_score:
                    attributive_graph[(subj_label, obj_label)] = {'subj': subj_label, 'obj': obj_label, 'pred': pred_label, 'score': score}

    return spatial_graph, functional_graph, partonomy_graph, attributive_graph

def draw_graphs(image, boxes, box_labels, all_rel_pairs, all_rel_labels, ind_to_predicates, all_rel_scores, box_scores, box_topk=10, rel_topk=20):
    # select top k boxes
    global_score = {}
    final_output = {'topological': [], 'functional': [], 'partonomy': [], 'attribute': []}

    size = get_size((image.shape[1], image.shape[0]))
    pic = Image.fromarray(image).resize(size)
    # get topk boxes:
    boxes = np.array(boxes[:box_topk])

    # get rel_topk relations from the set of relations between boxes from boxes[:box_topk]
    for i in range(len(all_rel_pairs)):
        if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk:
            global_score[i] = all_rel_scores[i]
    # sort the relationship by global score:
    sorted_rel = sorted(global_score.items(), key=lambda x: x[1], reverse=True)
    # slice the topk relationship:
    sorted_rel = dict(sorted_rel[:rel_topk])

    topo_graphs, func_graphs, parto_graph, attr_graphs = get_graphs(boxes, box_labels, all_rel_pairs, all_rel_labels, ind_to_predicates, all_rel_scores, box_scores)

    for i in list(sorted_rel.keys()):
        # get corresponding labels:
        subj_label = str(all_rel_pairs[i][0]) + '_' + box_labels[all_rel_pairs[i][0]]
        obj_label =  str(all_rel_pairs[i][1]) + '_' + box_labels[all_rel_pairs[i][1]]

        # add topological rel to final_output if it exist
        if (subj_label, obj_label) in topo_graphs.keys():
            final_output['topological'].append(topo_graphs[(subj_label, obj_label)])
        # add functional rel to final_output if it exist
        if (subj_label, obj_label) in func_graphs.keys():
            final_output['functional'].append(func_graphs[(subj_label, obj_label)])
        # add partonomy rel to final_output if it exist
        if (subj_label, obj_label) in parto_graph.keys():
            final_output['partonomy'].append(parto_graph[(subj_label, obj_label)])
        # add attribute rel to final_output if it exist
        if (subj_label, obj_label) in attr_graphs.keys():
            final_output['attribute'].append(attr_graphs[(subj_label, obj_label)])

        # get corresponding boxes:
        subj_box = boxes[all_rel_pairs[i][0]]
        obj_box = boxes[all_rel_pairs[i][1]]

        draw_single_box(pic, subj_box, draw_info=subj_label)
        draw_single_box(pic, obj_box, draw_info=obj_label)
    
    return pic, final_output

def draw_single_graph(graph, color='blue'):
    G_top = nx.MultiDiGraph()
    for i in range(len(graph)):
        r = graph[i]
        if r is None:
            continue
        G_top.add_edge(r['subj'], r['obj'], label=r['pred'])
        #print(r['subj'], r['pred'], r['obj'])

    # draw networkx graph with graphviz, display edge labels
    G_top.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
    G_top.graph['graph'] = {'scale': '3'}
    G_top.graph['node'] = {'shape': 'rectangle'}
    # all graph color to blue
    G_top.graph['edge']['color'] = color
    G_top.graph['node']['color'] = color
    A = to_agraph(G_top)
    A.layout('dot')
    return A

def graph_post_processing(custom_prediction, image, ind_to_classes, ind_to_predicates, box_topk=17, rel_thres=0.0):


    # Write your comments here.
    # This code draws the graphs of the topological, functional, partonomy and attribute relations,
    # with the top 20 of each type of relation.
    # The graphs are drawn with different colors, and the image is also drawn with the boxes
    # of the objects and the labels of the objects and the relations.

    boxes = custom_prediction['bbox']
    box_labels = custom_prediction['bbox_labels'].tolist()
    box_scores = np.array(custom_prediction['bbox_scores'], dtype=float)
    all_rel_labels = np.array(custom_prediction['rel_labels'], dtype=int)
    all_rel_scores = np.array(custom_prediction['rel_scores'], dtype=float)
    all_rel_pairs = np.array(custom_prediction['rel_pairs'], dtype=int)

    for i in range(len(box_labels)):
        box_labels[i] = ind_to_classes[str(box_labels[i])]

    img_with_boxes, graphs = draw_graphs(image, boxes, box_labels, all_rel_pairs, all_rel_labels, ind_to_predicates, all_rel_scores, box_scores, box_topk=20, rel_topk=20)

    top_graph = draw_single_graph(graphs['topological'], color='blue')

    func_graph = draw_single_graph(graphs['functional'], color='orange')

    parto_graph = draw_single_graph(graphs['partonomy'], color='green')

    attr_graph = draw_single_graph(graphs['attribute'], color='red')    
   
    width, height = 600, 600

    # Create a blank image
    img = Image.new('RGB', (width*2, height*2), (255, 255, 255))

    # Paste the four graph layouts onto the image

    with io.BytesIO() as output:
        top_graph.draw(output, format='png')
        img.paste(Image.open(output), (0, 0))
    with io.BytesIO() as output:
        func_graph.draw(output, format='png')
        img.paste(Image.open(output), (width, 0))
    with io.BytesIO() as output:
        parto_graph.draw(output, format='png')
        img.paste(Image.open(output), (0, height))
    with io.BytesIO() as output:
        attr_graph.draw(output, format='png')
        img.paste(Image.open(output), (width, height))

    # img.paste(Image.fromarray(np.uint8(top_graph.draw(format='png'))), (0, 0))
    # img.paste(Image.fromarray(np.uint8(func_graph.draw(format='png'))), (width, 0))
    # img.paste(Image.fromarray(np.uint8(parto_graph.draw(format='png'))), (0, height))
    # img.paste(Image.fromarray(np.uint8(attr_graph.draw(format='png'))), (width, height))

    # Show the image
    cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2_img_boxes = cv2.cvtColor(np.array(img_with_boxes), cv2.COLOR_RGB2BGR)
    cv2_img_boxes = cv2.cvtColor(cv2_img_boxes, cv2.COLOR_BGR2RGB)

    return cv2_img, cv2_img_boxes

