# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 14:13:42 2022

@author: Guoyuan An
"""

from scipy import spatial
import numpy as np
import cv2
sift = cv2.SIFT_create()

def check( la,lb, a_descriptors, a_locations,b_descriptors, b_locations,imga,imgb,a_radius=50,b_radius=50 ):
    
    a_region=(la[0]-a_radius,la[0]+a_radius,la[1]-a_radius,la[1]+a_radius)
    b_region=(lb[0]-b_radius,lb[0]+b_radius,lb[1]-b_radius,lb[1]+b_radius)
    
    #check 
    a_local_indices=[i  for i,l in enumerate(a_locations) if l[0]>a_region[0] and l[0]<a_region[1] and l[1]>a_region[2] and l[1]<a_region[3]]
    a_local_descriptors=a_descriptors[a_local_indices]
    a_local_locations=a_locations[a_local_indices] #(110,2)
    #len(a_local_locations)
    #vis_locals(a,a_local_locations)
    #vis_locals(a,a_local_locations,is_query=True)
    
    
    b_local_indices=[i  for i,l in enumerate(b_locations) if l[0]>b_region[0] and l[0]<b_region[1] and l[1]>b_region[2] and l[1]<b_region[3]]
    b_local_descriptors=b_descriptors[b_local_indices]
    b_local_locations=b_locations[b_local_indices] #(62,2)
    #len(b_local_locations)
    #vis_locals(b,b_local_locations)
    
    if len(a_local_locations)<10 or len(b_local_locations)<10:
        return 0,np.array([la]),np.array([lb]),(),()
    
    index_image_tree = spatial.cKDTree(b_local_descriptors)
    _, indices = index_image_tree.query(
        a_local_descriptors,
        workers=-1) 
    
    b_matched_location=b_local_locations[indices] #(110,2)
    #vis_matches(a,b,[a_local_locations, b_matched_location])
    #vis_matches(a,b,[a_local_locations, b_matched_location],img1_is_query=True)
    
    def feature_location(locations):
        up,down=np.min(locations[:,0]),np.max(locations[:,0])+1
        left,right=np.min(locations[:,1]),np.max(locations[:,1])+1
        
        h_size=(down-up)/3
        w_size=(right-left)/3
        
        locations=locations-(up,left)
        feature_locations=np.stack((locations[:,0]/h_size,locations[:,1]/w_size),axis=1).astype(int)
        return feature_locations,(up,down,left,right)
    
    
    a_local_feature_locations,(aup,adown,aleft,aright)=feature_location(a_local_locations) #(110,2)
    b_matched_feature_location,(bup,bdown,bleft,bright)=feature_location(b_matched_location) #(110,2)
    
    metric_method=3
    if metric_method==3:
        result,verified_indices=metric3(a_local_feature_locations,b_matched_feature_location)
        #return result, a_local_locations[verified_indices], b_matched_location[verified_indices]
        ##return result,np.mean(a_local_locations[verified_indices],axis=0),np.mean(b_matched_location[verified_indices],axis=0)
        if result==0.0:
            return result, _,_,_,_
        
        a_points, b_points=a_local_locations[verified_indices], b_matched_location[verified_indices]
        a_center,b_center=np.mean(a_points,axis=0),np.mean(b_points,axis=0)
        (down, right),( up, left)=np.max(a_points,axis=0).tolist(),np.min(a_points,axis=0).tolist()
        a_border=(up,down, left, right)
        (down, right),( up, left)=np.max(b_points,axis=0).tolist(),np.min(b_points,axis=0).tolist()
        b_border=(up,down, left, right)
        #vis_matches(a,b, [a_points, b_points])
        #vis_matches(a,b, [a_points, b_points],img1_is_query=True)
        return result, a_center,b_center,a_border,b_border
    
    elif metric_method==4:
        result=metric4((aup,adown,aleft,aright),(bup,bdown,bleft,bright),imga,imgb)
        a_center=(int((aup+adown)/2), int((aleft+aright)/2))
        b_center=(int((bup+bdown)/2), int((bleft+bright)/2))
        
        #visualize
# =============================================================================
#         cv2.rectangle(imga,(int(aleft),int(aup)),(int(aright),int(adown)),(0,255,0),2)
#         cv2.imshow('img',imga)
#         cv2.waitKey(0)
# =============================================================================
        return result, a_center,b_center,(aup,adown,aleft,aright),(bup,bdown,bleft,bright)
    
    #visualize
# =============================================================================
#     threshold=5
#     if result>threshold:
#         vis_matches(a,b,[a_local_locations, b_matched_location]) 
#         vis_matches(a,b,[a_local_locations[verified_indices], b_matched_location[verified_indices]]) 
#
#         vis_matches(a,b,[a_local_locations, b_matched_location],img1_is_query=True) 
#         vis_matches(a,b,[a_local_locations[verified_indices], b_matched_location[verified_indices]],img1_is_query=True) 
#     
# =============================================================================
    
    
   
def metric1(a_local_feature_locations,b_matched_feature_location): 
    
    verified_indices=np.sum(a_local_feature_locations==b_matched_feature_location,axis=1)==2
    
    n_matched=sum(verified_indices)
    
    return n_matched,verified_indices
  
def metric2(a_local_feature_locations,b_matched_feature_location):  
    #check how many cells have at least one matched feature
    
    verified_indices=np.sum(a_local_feature_locations==b_matched_feature_location,axis=1)==2
    locations=a_local_feature_locations[verified_indices]
    
    unique_locs=np.unique(locations,axis=0)
    return unique_locs.shape[0], verified_indices

def metric3(a_local_feature_locations,b_matched_feature_location):  
    #for each cell check the portion
    
    verified_indices=np.sum(a_local_feature_locations==b_matched_feature_location,axis=1)==2
    ratio=np.sum(verified_indices)/verified_indices.shape[0]
    
    return ratio, verified_indices
    
def metric4(a_region,b_region,imga,imgb):
    aup,adown,aleft,aright=a_region
    bup,bdown,bleft,bright=b_region
    ap=imga[int(aup):int(adown), int(aleft):int(aright)]
    bp=imgb[int(bup):int(bdown), int(bleft):int(bright)]
    
    #extract features for a patch 
    sizey,sizex=int(ap.shape[0]/3),int(ap.shape[1]/3)
    ya =[int(0+sizey/2),int(sizey+sizey/2),int(2*sizey+sizey/2) ]
    xa =[int(0+sizex/2),int(sizex+sizex/2),int(2*sizex+sizex/2) ]
    xv, yv = np.meshgrid(xa, ya)
    xv,yv=xv.flatten().astype(float),yv.flatten().astype(float)
    size=max(sizey,sizex)
    kpa=[cv2.KeyPoint(x,y,size) for x,y in zip(xv,yv)]
    (_,desa)=sift.compute(ap, kpa)
    #gray1=cv2.drawKeypoints(ap, kpa, ap,(0,255,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('image',gray1)
    #cv2.waitKey(0)

    
    #extract features for b patch 
    sizey,sizex=int(bp.shape[0]/3),int(bp.shape[1]/3)
    yb =[int(0+sizey/2),int(sizey+sizey/2),int(2*sizey+sizey/2) ]
    xb =[int(0+sizex/2),int(sizex+sizex/2),int(2*sizex+sizex/2) ]
    xv, yv = np.meshgrid(xb, yb)
    xv,yv=xv.flatten().astype(float),yv.flatten().astype(float)
    kpb=[cv2.KeyPoint(x,y,size) for x,y in zip(xv,yv)]
    (_,desb)=sift.compute(bp, kpb)
    #gray1=cv2.drawKeypoints(bp, kpb, bp,(0,255,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('image',gray1)
    #cv2.waitKey(0)
    
    #nearest neighbor search
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desa,desb,k=1)
    nearest=[i[0] for i in matches ]
    b_idx=[m.trainIdx for m in nearest]
    n_matched=np.sum(np.array(b_idx)==np.array([i for i in range(9)]))
    return n_matched

# =============================================================================
#     draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                        singlePointColor = None,
#                       flags = 2)
#     img3 = cv2.drawMatches(ap,kpa,bp,kpb,nearest,None,**draw_params)
# 
# 
#     cv2.imshow('img',img3)
#     cv2.waitKey(0)
#     
#     cv2.imshow('img',ap)
#     cv2.waitKey(0)
#     cv2.imshow('img',bp)
#     cv2.waitKey(0)
#     
# =============================================================================
    
    