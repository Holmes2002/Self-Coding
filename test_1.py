import cv2 as cv
import os
1,6+8,7+10
import numpy as np
background = [0,2,3,4,8,13,14,15,16,17]
hat = [1]
shirt = [2]
pants = [3]
shoe  = [4]
color=[[0,0,0],[255,0,0],[0,0,255],[0,128,0],[255,255,0],[255,165,0],[0,255,255],[165,42,42],[128,128,128]]
accept_label=[1,2,3,4]
def main(img):
    h,w=img.shape
    new_img=[]
    print(np.amax(img))
    for i in range(h):
        row_img=[]
        for j in range(w):
            
            # if img[i][j] in hat:
            #     row_img.append(color[1])
            # elif     img[i][j] in shirt:
            #     row_img.append(color[2])
            # elif     img[i][j] in pants:
            #     row_img.append(color[3])
            # elif     img[i][j] in shoe:
            #     row_img.append(color[4])

            # else :
            #     row_img.append(color[0])
            if img[i][j] in accept_label:
                row_img.append(img[i][j])
            else:
                row_img.append(0)

        new_img.append(row_img)
    return np.array(new_img)
small=open("train_small.txt",'r').read().splitlines()
large=os.listdir('segment human data/train_folder/large_fol/train_segmentations')
total=list(set(large)-set(small))
total.reverse()


for file in total:
    img = cv.imread(f'segment human data/train_folder/large_fol/train_segmentations/{file}')
    img=img[:,:,0]
    
    img=main(img).astype('uint8')
    cv.imwrite(f"segment human data/train_folder/accept_labels/{file}",img)
    # cv.imshow('A',img)
    # print(file)
    # file=file.replace('png','jpg')
    # new= cv.imread(f'segment human data/train_folder/large_fol/train_images/{file}').astype('uint8')
    # cv.imshow('B',new)
    # cv.waitKey(0)

