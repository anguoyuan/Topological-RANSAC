# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:23:59 2022

@author: Guoyuan An
"""

import numpy as np
from shapely.geometry import MultiPoint, Point


def _no_repeat(origin_list):
    # remove the repeat ones and 
    list_set=set(origin_list)
    unique_number=len(list_set)
    
    new_list=[]
    for x in origin_list:
        if x in list_set:
            new_list.append(x)
            list_set.remove(x)
        if len(new_list)==unique_number:
            break
    return new_list

class Control:
    
    def __init__(self, a_locations,b_locations, Match_Loc,radius=50, a_is_query=False):
        self.a_locations=a_locations
        self.b_locations=b_locations
        self.Match_Loc=Match_Loc
        self.radius=30

        self.a_starting_points=Match_Loc[0].tolist()#[-100:]
        self.b_starting_points=Match_Loc[1].tolist()#[-100:]

        self.a_starting_points.reverse()
        self.b_starting_points.reverse()
    
        self.restart=True
        self.la,self.lb=[],[]
        
        self.Max=0
        self.Max_a_verified=[]
        self.Max_b_verified=[]
        self.All_a_verified=[]
        self.All_b_verified=[]
        
        #for a fixed starting_point
        self.verified=set()
        self.a_verified=[]
        self.b_verified=[]
        self.to_explore=[]
        
        self.finished=False
        
        #trace
        self.All_a_trace=[]
        self.All_b_trace=[]
        self.a_trace=[] #[(n_matched,center, border),...  ]
        self.b_trace=[] 
        
    def _next(self,n_matched=0, a_points=[], b_points=[]):
        if len(self.to_explore)==0:
            self.renew_persistence()
            if self.finished==True:
                return True, [],[]
   
            
        self.la,self.lb=self.to_explore.pop()
        #print(len(self.to_explore),self.restart)
    
        return False, self.la, self.lb
    
    
    def renew_persistence(self):
            
        if len(self.a_verified)!=0:
            self.All_a_verified[-1]=np.array(self.a_locations[_no_repeat(self.a_verified)])
            self.All_b_verified[-1]=np.array(self.b_locations[_no_repeat(self.b_verified)])
# =============================================================================
#             self.All_a_verified.append(np.array(self.a_locations[_no_repeat(self.a_verified)]))
#             self.All_b_verified.append(np.array(self.b_locations[_no_repeat(self.b_verified)]))
# =============================================================================
            
            self.All_a_trace[-1]=self.a_trace
            self.All_b_trace[-1]=self.b_trace

        while True:
            if len(self.a_starting_points)==0:
                self.finished=True
                return 
            new_start_a, new_start_b=self.a_starting_points.pop(),self.b_starting_points.pop()
            if MultiPoint(self.a_locations[list(self.verified)]).convex_hull.contains(Point(new_start_a))==False:
                break
        self.to_explore.append((new_start_a, new_start_b))
        self.latent=set(range(self.a_locations.shape[0]))
        self.a_verified=[]
        self.b_verified=[]
        self.All_a_verified.append(np.array(new_start_a).reshape((1,2)))
        self.All_b_verified.append(np.array(new_start_b).reshape((1,2)))
        
        #
        self.a_trace=[]
        self.b_trace=[]
        self.All_a_trace.append([])
        self.All_b_trace.append([])
    
    def update(self,n_matched,a_center, b_center,a_border,b_border):
        ifpass=n_matched>=0.2
        radius=self.radius
        a_region=(self.la[0]-radius,self.la[0]+radius,self.la[1]-radius,self.la[1]+radius)
        b_region=(self.lb[0]-radius,self.lb[0]+radius,self.lb[1]-radius,self.lb[1]+radius)
        
        a_local_indices=[i  for i,l in enumerate(self.a_locations) if l[0]>a_region[0] and l[0]<a_region[1] and l[1]>a_region[2] and l[1]<a_region[3]]
        b_local_indices=[i  for i,l in enumerate(self.b_locations) if l[0]>b_region[0] and l[0]<b_region[1] and l[1]>b_region[2] and l[1]<b_region[3]]
        
        #print(len(a_local_indices))
        self.latent=self.latent-set(a_local_indices)
        if ifpass==True:
            self.verified.update(a_local_indices)
            self.a_verified=self.a_verified+a_local_indices
            self.b_verified=self.b_verified+b_local_indices
            self.look_around(a_center,b_center)
            
            self.a_trace.append((n_matched, a_center, a_border))
            self.b_trace.append((n_matched, b_center, b_border))
            
            
        
    def look_around(self,a_center,b_center):
            
            
        self.look_down(a_center,b_center)
        self.look_up(a_center,b_center)
        self.look_left(a_center,b_center)
        self.look_right(a_center,b_center)
        
    def look_up(self,a_center, b_center):
        new_center=([a_center[0]-self.radius, a_center[1]],[b_center[0]-self.radius, b_center[1]])
        region=(new_center[0][0]-self.radius, new_center[0][0]+self.radius, new_center[0][1]-self.radius, new_center[0][1]+self.radius)
        po=[l for l in self.a_locations[list(self.latent)] if l[0]>region[0] and l[0]<region[1] and l[1]>region[2] and l[1]<region[3] ]
        if len(po)>0:
            self.to_explore.append(new_center)
        
    def look_down(self,a_center, b_center):
        new_center=([a_center[0]+self.radius, a_center[1]],[b_center[0]+self.radius, b_center[1]])
        region=(new_center[0][0]-self.radius, new_center[0][0]+self.radius, new_center[0][1]-self.radius, new_center[0][1]+self.radius)
        po=[l for l in self.a_locations[list(self.latent)] if l[0]>region[0] and l[0]<region[1] and l[1]>region[2] and l[1]<region[3] ]
        if len(po)>0:
            self.to_explore.append(new_center)
            
    def look_left(self,a_center, b_center):
        new_center=([a_center[0], a_center[1]-self.radius],[b_center[0], b_center[1]-self.radius])
        region=(new_center[0][0]-self.radius, new_center[0][0]+self.radius, new_center[0][1]-self.radius, new_center[0][1]+self.radius)
        po=[l for l in self.a_locations[list(self.latent)] if l[0]>region[0] and l[0]<region[1] and l[1]>region[2] and l[1]<region[3] ]
        if len(po)>0:
            self.to_explore.append(new_center)
            
    def look_right(self,a_center, b_center):
        new_center=([a_center[0], a_center[1]+self.radius],[b_center[0], b_center[1]+self.radius])
        region=(new_center[0][0]-self.radius, new_center[0][0]+self.radius, new_center[0][1]-self.radius, new_center[0][1]+self.radius)
        po=[l for l in self.a_locations[list(self.latent)] if l[0]>region[0] and l[0]<region[1] and l[1]>region[2] and l[1]<region[3] ]
        if len(po)>0:
            self.to_explore.append(new_center)
    