# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
# Code Developed by: Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
# 2018.11.07
 
# python3 -m homogenus.tf.homogenus_infer_PHALP -ii PHALP_data/img -oi PHALP_data/bboxes/ -io PHALP_data/img_gendered

import tensorflow as tf
import numpy as np
import os, glob
import joblib

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

class Homogenus_infer(object):

    def __init__(self, trained_model_dir, sess=None):
        '''

        :param trained_model_dir: the directory where you have put the homogenus TF trained models
        :param sess:
        '''

        best_model_fname = sorted(glob.glob(os.path.join(trained_model_dir , '*.ckpt.index')), key=os.path.getmtime)
        if len(best_model_fname):
            self.best_model_fname = best_model_fname[-1].replace('.index', '')
        else:
            raise ValueError('Couldnt find TF trained model in the provided directory --trained_model_dir=%s. Make sure you have downloaded them there.' % trained_model_dir)


        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        # Load graph.
        self.saver = tf.train.import_meta_graph(self.best_model_fname+'.meta')
        self.graph = tf.get_default_graph()
        self.prepare()

    def prepare(self):
        print('Restoring checkpoint %s..' % self.best_model_fname)
        self.saver.restore(self.sess, self.best_model_fname)


    def predict_genders(self, images_indir, phalp_data, start_frame=1, end_frame=200, images_outdir=None, allow_bbox_overlap=False):

        '''
            Given a directory with images and another directory with corresponding openpose genereated jsons will
            augment openpose jsons with gender labels.

        :param images_indir: Input directory of images with common extensions
        :param openpose_indir: Input directory of openpose jsons
        :param images_outdir: If given will overlay the detected gender on detected humans that pass the criteria
        :param openpose_outdir: If given will dump the gendered openpose files in this directory. if not will augment the origianls
        :return:
        '''

        import os, sys
        import json

        from homogenus.tools.image_tools import put_text_in_image, fontColors, read_prep_image, save_images
        from homogenus.tools.body_cropper import cropout_openpose, should_accept_pose

        import cv2
        from homogenus.tools.omni_tools import makepath
        import glob

        sys.stdout.write('\nRunning homogenus on --images_indir=%s s\n'%(images_indir))

        im_fnames_orig = []
        for img_ext in ['png', 'jpg', 'jpeg', 'bmp']:
            im_fnames_orig.extend(glob.glob(os.path.join(images_indir, '*.%s'%img_ext)))

        if len(im_fnames_orig):
            sys.stdout.write('Found %d images\n' % len(im_fnames_orig))
        else:
            raise ValueError('No images could be found in %s'%images_indir)

        # print(im_fnames_orig)


        im_fnames = []

        for img_f in im_fnames_orig:
            frame_str = img_f[-10:-4]
            frame = int(frame_str)
            if frame >= start_frame and frame <= end_frame:
                im_fnames.append(img_f)

        accept_threshold = 0.55
        crop_margin = 0.08

        if images_outdir is not None: makepath(images_outdir)
    
        Iph = self.graph.get_tensor_by_name(u'input_images:0')

        probs_op = self.graph.get_tensor_by_name(u'probs_op:0')

        summed_probs = []
        counts = []

        for j, im_fname in enumerate(im_fnames):
            im_basename = os.path.basename(im_fname)
            img_ext = im_basename.split('.')[-1]

            frame_str = im_fname[-10:-4] 
            bboxes_data = phalp_data[f'{frame_str}.jpg']['tracked_bbox']
            bboxes = [x.tolist() for x in bboxes_data]

            # print(im_fname)
            
            im_orig = cv2.imread(im_fname, 3)[:,:,::-1].copy()
            
            execute_frame = False

            # only execute if 2 people were detected
            if len(bboxes) == 2:
                execute_frame = True

                if allow_bbox_overlap:
                    execute_frame = True
                else: 
                    for i, bbox in enumerate(bboxes):
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[0] + bbox[2]) 
                        y2 = int(bbox[1] + bbox[3])
                        if i == 0:
                            bbox1 = {'x1': x1, 'x2':x2, 'y1':y1, 'y2':y2}
                        else:
                            bbox2 = {'x1': x1, 'x2':x2, 'y1':y1, 'y2':y2} 
                    iou = get_iou(bbox1, bbox2)
                    # print("IOU:")
                    # print(iou)
                    if iou > 0:
                        execute_frame = False
                
                if execute_frame:
                    for i, bbox in enumerate(bboxes):
                        # print(i, bbox)
                        
                        # import ipdb; ipdb.set_trace()
                        # print(bbox)
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[0] + bbox[2]) 
                        y2 = int(bbox[1] + bbox[3])
                        cropped_image = im_orig[y1:y2,x1:x2,  :]
                        # if cropped_image.shape[0] < 200 or cropped_image.shape[1] < 200: continue

                        img = read_prep_image(cropped_image)[np.newaxis]

                        probs_ob = self.sess.run(probs_op, feed_dict={Iph: img})[0]
                        # print(probs_ob)
                        gender_id = np.argmax(probs_ob, axis=0)

                        gender_prob = probs_ob[gender_id]
                        gender_pd = 'male' if gender_id == 0 else 'female'

                        if gender_prob>accept_threshold and gender_pd == 'male':
                            color = 'blue'
                            text = 'pred:%s[%.3f]' % (gender_pd, gender_prob)
                        elif gender_prob>accept_threshold and gender_pd == 'female':
                            color = 'red'
                            text = 'pred:%s[%.3f]' % (gender_pd, gender_prob)
                        else:
                            text = 'pred:%s[%.3f]' % (gender_pd, gender_prob)
                            gender_pd = 'neutral'
                            color = 'grey'

                        
                        im_orig = cv2.rectangle(im_orig, (x1, y1), (x2, y2), fontColors[color], 2)
                        im_orig = put_text_in_image(im_orig, [text], color, (x1, y1))[0]

                        # import ipdb; ipdb.set_trace()
                        if j == 0 or i >= len(summed_probs):
                            
                            summed_probs.append(probs_ob)
                            counts.append(1)

                        else:
                            # print(i)
                            # print(summed_probs[i])
                            summed_probs[i] += probs_ob
                            counts[i] += 1
                        # print("summed_probs")
                        # print(summed_probs)
                        # print(counts)

                    if images_outdir != None:
                        # print("writing image")
                        save_images(im_orig, images_outdir, [os.path.basename(im_fname)])
            
        for p in range(len(summed_probs)):
            print(f"person at index {p}:")
            
            prob_male = round(summed_probs[p][0]/counts[p], 3)
            prob_female = round(summed_probs[p][1]/counts[p], 3)
            
            print(counts)
            print(f"avg prob female: {prob_female}, avg prob male: {prob_male}")

                
                
        if images_outdir is not None:
            sys.stdout.write('Dumped overlayed images at %s'%images_outdir)



if __name__ == '__main__':
    print("processing PHALP data")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-tm", "--trained_model_dir", default="./homogenus/trained_models/tf/", help="The path to the directory holding homogenus trained models in TF.")
    # parser.add_argument("-n", "--mp4_name", required= True, help="Directory of the input images.")
    # parser.add_argument("-oi", "--bboxes_indir", required=True, help="Directory of bounding boxes, e.g. json files.")
    # parser.add_argument("-io", "--images_outdir", default=None, help="Directory to put predicted gender overlays. If not given, wont produce any overlays.")
    # # parser.add_argument("-oo", "--openpose_outdir", default=None, help="Directory to put the openpose gendered keypoints. If not given, it will augment the original openpose json files.")

    ps = parser.parse_args()
    
    mp4s = ["00LQjxAgHHA", "JC8ATckUGAo", "trSum93Z_6I", "_13iG-MvGtg", "6W8VGnEHf-I", "6xfLkqXWC8o", "bwFbVCxIpRo", "k8175B7vQZ4", "laLyyC_RXa4"]
    for mp4_name in mp4s:
        print(mp4_name)
        pkl = f"/home/neerja/swing_dancing/out/phalp_dance_data/results/mp4_file_{mp4_name}.pkl"
        # print(pkl)
        phalp_data = joblib.load(pkl)
        # frame = "000001"
        # bbox = phalp_data[f'{frame}.jpg']['tracked_bbox']
        # bbox_list = [x.tolist() for x in bbox]

        images_indir = f"/home/neerja/swing_dancing/dance_data/{mp4_name}/img/"
        images_outdir = f"PHALP_data/{mp4_name}_img_gendered"

        # print(bbox_list)

        hg = Homogenus_infer(trained_model_dir=ps.trained_model_dir)

        print("\nallowing bbox overlap:")
        hg.predict_genders(images_indir=images_indir, phalp_data=phalp_data, start_frame=1, end_frame=1000,
                        images_outdir=images_outdir, allow_bbox_overlap=True)
        print("\nnot allowing bbox overlap:")
        hg.predict_genders(images_indir=images_indir, phalp_data=phalp_data, start_frame=1, end_frame=1000,
                        images_outdir=images_outdir, allow_bbox_overlap=False)

