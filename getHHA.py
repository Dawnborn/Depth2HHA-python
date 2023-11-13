# --*-- coding:utf-8 --*--
#%%
import math
import cv2
import os
import math

from utils.rgbd_util import *
from utils.getCameraParam import *

'''
must use 'COLOR_BGR2GRAY' here, or you will get a different gray-value with what MATLAB gets.
'''
def getImage(root='demo'):
    D = cv2.imread(os.path.join(root, '0.png'), cv2.COLOR_BGR2GRAY)/10000
    RD = cv2.imread(os.path.join(root, '0_raw.png'), cv2.COLOR_BGR2GRAY)/10000
    return D, RD


def getHHA(C, D, RD):
    """
    Input:
        C: Camera matrix, np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
        D: Depth image, numpy array of shape (H,W), the unit of each element in it is "meter"
        RD: Raw depth image, the unit of each element in it is "meter"
    Return:
        I[:,:,2]: 水平视差
        
        I[:,:,1]: 像素相较于地面高度, unit: centimeter

        I[:,:,0]: 像素表面法线与重力方向夹角, degree表示
    """
    missingMask = (RD == 0)
    pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C)

    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    '''
    Must convert nan to 180 as the MATLAB program actually does. 
    Or we will get a HHA image whose border region is different
    with that of MATLAB program's output.
    '''
    angle[np.isnan(angle)] = 180        


    pc[:,:,2] = np.maximum(pc[:,:,2], 100)
    I = np.zeros(pc.shape)

    # opencv-python save the picture in BGR order.
    I[:,:,2] = 31000/pc[:,:,2]
    I[:,:,1] = h
    I[:,:,0] = (angle + 128-90)

    # print(np.isnan(angle))

    '''
    np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
    So I convert it to integer myself.
    '''
    I = np.rint(I)

    # np.uint8: 256->1, but in MATLAB, uint8: 256->255
    I[I>255] = 255
    HHA = I.astype(np.uint8)
    return HHA

#%%
if __name__ == "__main__":
    D, RD = getImage() # D: depth, RD: raw_depth
    camera_matrix = getCameraParam('color')
    print('max gray value: ', np.max(D))        # make sure that the image is in 'meter'
    #%%
    hha = getHHA(camera_matrix, D, RD) # 参考raw_depth作为mask，只输出有深度值的部分
    hha_complete = getHHA(camera_matrix, D, D) # 无mask，整个图片均输出
    cv2.imwrite('demo/hha.png', hha)
    cv2.imwrite('demo/hha_complete.png', hha_complete)
    
    
    ''' multi-peocessing example '''
    '''
    from multiprocessing import Pool
    
    def generate_hha(i):
        # generate hha for the i-th image
        return
    
    processNum = 16
    pool = Pool(processNum)

    for i in range(img_num):
        print(i)
        pool.apply_async(generate_hha, args=(i,))
        pool.close()
        pool.join()
    ''' 
