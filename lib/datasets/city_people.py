# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval, myvoc_eval
from model.config import cfg


class city_people(imdb):
  def __init__(self, image_set, year, use_diff=False):
    name = 'citypeople_' + year + '_' + image_set
    if use_diff:
      name += '_diff'
    imdb.__init__(self, name)
    self._year = year
    self._image_set = image_set
    self._devkit_path = '/data/sparks/user/imosnoi/datasets/cityscopes/leftImg8bit'
    self.anot = '/home/imosnoi/code/tf-faster-rcnn/annotations/anno_'+image_set+'.mat'
    #mat = sio.loadmat(self.anot)#self.anot
    #mat1 = scipy.io.loadmat('annotations/anno_'+image_set+'.mat')
    #self._get_default_path()
    self._data_path = os.path.join(self._devkit_path, image_set)
    '''
    class_label =0: ignore regions (fake humans, e.g. people on posters, reflections etc.)
    class_label =1: pedestrians
    class_label =2: riders
    class_label =3: sitting persons
    class_label =4: other persons with unusual postures
    class_label =5: group of people
    '''
    self._classes = ('__background__',  # always index 0
                     'pedestrians', 'riders', 'sitting persons', 'other persons with unusual postures',
                     'group of people')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.png'
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': use_diff,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'CityPersons path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self._image_index[i] + self._image_ext

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_index = []
    self.boxs = []
    print(self.anot)
    mat2 = sio.loadmat(self.anot)
    for  i  in mat2['anno_'+self._image_set+'_aligned'][0]:
        c, n, b = i[0].tolist()[0][:]
        pa = '/data/sparks/user/imosnoi/datasets/cityscopes/leftImg8bit/'+self._image_set+'/'+c[0]+'/'+n[0]
        bb = self._load_annotation(b)
        if len(bb['boxes'])==0:
            #print('error-'*10)
            continue
        image_index.append(pa.split('.')[0])
        self.boxs.append(b)
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    mat = sio.loadmat(self.anot)#self.anot
    if os.path.exists(cache_file) and False:
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb
    
    gt_roidb = []
    mat2 = sio.loadmat(self.anot)
    for  i  in mat2['anno_'+self._image_set+'_aligned'][0]:
        c, n, b = i[0].tolist()[0][:]
        pa = '/data/sparks/user/imosnoi/datasets/cityscopes/leftImg8bit/'+self._image_set+'/'+c[0]+'/'+n[0]
        bb = self._load_annotation(b)
        if len(gt_roidb) == 133332:
            print(b)
        if len(bb['boxes'])==0:
            #print('error-'*10)
            continue
        gt_roidb.append(bb)
        
    print('info',len(np.array([len(i['boxes']) for i in gt_roidb if len(i['boxes'])==0] )), len(gt_roidb))
    
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    
    #print(gt_roidb[12]['boxes'])

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_annotation(self, objs):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """    
    num_objs = len(objs)#the problem!!!!

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    ix = 0
    h,w = 1024, 2048
    for ixx, obj in enumerate(objs):
      
      # Make pixel indexes 0-based
      '''
            x1 = int(v[1])
            x2 = int(v[1]+v[3])
            y1 = int(v[2])
            y2 = int(v[2]+v[4])
      '''
      if obj[0] == 0:
            continue
      ok = 1
      for jf in range(1,10):
            obj[jf] = max(1,obj[jf])
      
      if True or obj[0] == 1 or obj[0] == 2:
          x1 = float(obj[1]) - 1
          y1 = float(obj[2]) - 1
          x2 = float(obj[1])+float(obj[3]) - 1
          y2 = float(obj[2])+float(obj[4]) - 1
      if x2> w or y2>h:
          x1 = float(obj[1+5]) - 1
          y1 = float(obj[2+5]) - 1
          x2 = float(obj[1+5])+float(obj[3+5]) - 1
          y2 = float(obj[2+5])+float(obj[4+5]) - 1
        
      if x2 == w:
        x2 -= 1
      if x2> w or y2>h or x1>w or y1>h:
        continue
      
      if y1 >= y2:
        print('error', obj)
        continue
      if x1 >= x2:
        print('error', obj)
        continue
      cls = obj[0]
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      overlaps[ix, cls] = 1.0
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
      ix+=1
    
    boxes = boxes[:ix]
    gt_classes = gt_classes[:ix]
    overlaps = overlaps[:ix]
    seg_areas = seg_areas[:ix]

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id


  def evaluate_detections(self, all_boxes, output_dir):
    #self._write_voc_results_file(all_boxes)
    #self._do_python_eval1(all_boxes, output_dir)
    db = self.gt_roidb()
    aps = []
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      rec, prec, ap = myvoc_eval(all_boxes[i], i, db, ovthresh=0.5)
      aps += [ap]
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      #print(('Recall for {} = {:.4f}'.format(cls, rec)))
      #print(('Precition for {} = {:.4f}'.format(cls, prec)))
      #print('-'*40)
    

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.city_people import city_people

  d = city_people('train', '2016')
  res = d.roidb
  from IPython import embed;

  embed()
