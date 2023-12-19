import cv2
from topological_ransac import T_RANSAC

query_list=['all_souls_000013.jpg']
index_list=['all_souls_000015.jpg','all_souls_000027.jpg']

for query in query_list:
    for img in index_list:

        imga=cv2.imread(query,cv2.IMREAD_COLOR)  
        # left, upper, right, lower=gnd[a]['bbx']
        # left, upper, right, lower=int(left),int(upper),int(right),int(lower) #(left, upper, right, lower)
        # imga=imga[upper:lower,left:right] #[y1:y2, x1:x2]

        imgb = cv2.imread(img,cv2.IMREAD_COLOR)  

        a_verified,b_verified, a_trace, b_trace= T_RANSAC(imga, imgb)

        print('Topological RANSAC result for: ', query, img)
        print('detected ', len(a_verified), 'homeomorphism regions in total')
        print('the number of patches in the largest homeomorphism region is: ', max([len(hr) for hr in a_verified]))
        print()

