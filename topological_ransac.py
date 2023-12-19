# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:49:11 2022

@author: Guoyuan An
"""

import numpy as np
from WM_control import Control
from WM_patch_basic_sift import check
import cv2


sift = cv2.SIFT_create()



def analysis(a_descriptors, a_locations,b_descriptors, b_locations, Match_Loc,imga,imgb):
    #Match_Loc=Ransac_Match_Loc
    controller=Control(a_locations, b_locations, Match_Loc,a_is_query=True)
    
    finished,la,lb=controller._next()
    #la=[570,200]
    #lb=[660,390]
    for i in range(100000):
        #la,lb=La[-i],Lb[-i]     
        n_matched,a_center,b_center,a_border,b_border=check( la, lb, a_descriptors, a_locations, b_descriptors, b_locations,imga,imgb)
             
        controller.update(n_matched,a_center,b_center,a_border,b_border)
        finished,la,lb=controller._next(n_matched)
        
        
        if finished==True:
            #print('finished')
            break
       
    return controller.All_a_verified,controller.All_b_verified,controller.All_a_trace,controller.All_b_trace



def initial_set(des1,loc1,des2,loc2,initial_method='ratio'):


    #match
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
 

    # store all the good matches as per Lowe's ratio test.
    # we don't allow a feature in index image to be matched by several query features
    good = []
    matched_features=set()
    for m,n in matches:    
        if m.distance < 0.8*n.distance and (m.trainIdx not in matched_features):
            good.append(m)
            matched_features.add(m.trainIdx)
    
    if len(good)<4:
        return [np.zeros((0,2)),np.zeros((0,2))]
    
    src_pts = np.float32([ loc1[m.queryIdx] for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ loc2[m.trainIdx] for m in good ]).reshape(-1,1,2)

    if initial_method=='ratio':
        query_inlier_locations=src_pts.squeeze(1)
        index_inlier_locations=dst_pts.squeeze(1)

        return [query_inlier_locations, index_inlier_locations ]


    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,2.0,maxIters = 1000)
    matchesMask = mask.ravel().tolist()
    
    if sum(matchesMask)<1:
        return [np.zeros((0,2)),np.zeros((0,2))]
    
    query_inlier_locations=np.array([src_pts[i] for i,p in enumerate(matchesMask) if p==1]).squeeze(1)
    index_inlier_locations=np.array([dst_pts[i] for i,p in enumerate(matchesMask) if p==1]).squeeze(1)
    #vis_matches(query,index, [query_inlier_locations,index_inlier_locations],img1_is_query=True)

    return [query_inlier_locations, index_inlier_locations ]

def T_RANSAC(imga, imgb, feature='sift'):
    if feature=='sift':
        #detect sift features for imga and imgb
        keypoints_a, a_descriptors = sift.detectAndCompute(imga, None)
        keypoints_b, b_descriptors = sift.detectAndCompute(imgb, None)
        a_locations = np.array([kp.pt for kp in keypoints_a])
        b_locations=np.array([kp.pt for kp in keypoints_b])

    try:
        Initial_set=initial_set( a_descriptors, a_locations,b_descriptors, b_locations,initial_method='ratio')
        a_verified,b_verified, a_trace, b_trace=analysis( a_descriptors, a_locations,b_descriptors, b_locations, Initial_set,imga,imgb)

        return a_verified,b_verified, a_trace, b_trace
    except:
        print('failed')

    







