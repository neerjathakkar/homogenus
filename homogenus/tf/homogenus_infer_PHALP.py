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


    def predict_genders(self, images_indir, bboxes_indir, images_outdir=None):

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

        sys.stdout.write('\nRunning homogenus on --images_indir=%s --bboxes_indir=%s\n'%(images_indir, bboxes_indir))

        im_fnames = []
        for img_ext in ['png', 'jpg', 'jpeg', 'bmp']:
            im_fnames.extend(glob.glob(os.path.join(images_indir, '*.%s'%img_ext)))

        if len(im_fnames):
            sys.stdout.write('Found %d images\n' % len(im_fnames))
        else:
            raise ValueError('No images could be found in %s'%images_indir)

        accept_threshold = 0.55
        crop_margin = 0.08

        if images_outdir is not None: makepath(images_outdir)
    
        Iph = self.graph.get_tensor_by_name(u'input_images:0')

        probs_op = self.graph.get_tensor_by_name(u'probs_op:0')

        # im_fnames = ["/home/neerja/swing_dancing/dance_data/00LQjxAgHHA/img/000001.jpg"]
        # im_fnames = ["/home/neerja/000001.jpg"]
        bboxes = [np.array([587.50415, 116.60168, 138.37842, 470.2865 ]).astype(int), np.array([510.2559 , 157.07776, 107.97354, 431.63757]).astype(int)]
        for im_fname in im_fnames:
            im_basename = os.path.basename(im_fname)
            img_ext = im_basename.split('.')[-1]
            bbox_in_fname = os.path.join(bboxes_indir, im_basename.replace('.%s'%img_ext, '.json'))
            
            with open(bbox_in_fname, 'r') as f: bbox_data = json.load(f)
            print(bbox_data)
            bboxes = [np.array(x).astype(int) for x in bbox_data]
            print(bboxes)
            

            print(im_fname)
            
            im_orig = cv2.imread(im_fname, 3)[:,:,::-1].copy()
            for bbox in bboxes:
                # import ipdb; ipdb.set_trace()
                # print(bbox)
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[0] + bbox[2] 
                y2 = bbox[1] + bbox[3]
                cropped_image = im_orig[y1:y2,x1:x2,  :]
                # if cropped_image.shape[0] < 200 or cropped_image.shape[1] < 200: continue

                img = read_prep_image(cropped_image)[np.newaxis]

                probs_ob = self.sess.run(probs_op, feed_dict={Iph: img})[0]
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

                
                # x1 = crop_info['crop_boundary']['offset_width']
                # y1 = crop_info['crop_boundary']['offset_height']
                # x2 = crop_info['crop_boundary']['target_width'] + x1
                # y2 = crop_info['crop_boundary']['target_height'] + y1
                im_orig = cv2.rectangle(im_orig, (x1, y1), (x2, y2), fontColors[color], 2)
                im_orig = put_text_in_image(im_orig, [text], color, (x1, y1))[0]

                # pose_data['people'][opnpose_pIdx]['gender_pd'] = gender_pd

                # sys.stdout.write('%s -- peron_id %d --> %s\n'%(im_fname, opnpose_pIdx, gender_pd))
                
            if images_outdir != None:
                print("writing image")
                save_images(im_orig, images_outdir, [os.path.basename(im_fname)])
                
            # openpose_out_fname = os.path.join(openpose_outdir, im_basename.replace('.%s'%img_ext, '_keypoints.json'))
            # with open(openpose_out_fname, 'w') as f: json.dump(pose_data, f)

        if images_outdir is not None:
            sys.stdout.write('Dumped overlayed images at %s'%images_outdir)
        # sys.stdout.write('Dumped gendered openpose keypoints at %s'%openpose_outdir)



if __name__ == '__main__':
    print("processing PHALP data")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-tm", "--trained_model_dir", default="./homogenus/trained_models/tf/", help="The path to the directory holding homogenus trained models in TF.")
    parser.add_argument("-ii", "--images_indir", required= True, help="Directory of the input images.")
    parser.add_argument("-oi", "--bboxes_indir", required=True, help="Directory of bounding boxes, e.g. json files.")
    parser.add_argument("-io", "--images_outdir", default=None, help="Directory to put predicted gender overlays. If not given, wont produce any overlays.")
    # parser.add_argument("-oo", "--openpose_outdir", default=None, help="Directory to put the openpose gendered keypoints. If not given, it will augment the original openpose json files.")

    ps = parser.parse_args()

    hg = Homogenus_infer(trained_model_dir=ps.trained_model_dir)
    hg.predict_genders(images_indir=ps.images_indir, bboxes_indir=ps.bboxes_indir,
                       images_outdir=ps.images_outdir)
