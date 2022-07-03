import numba
import torch
import numpy as np
from utils.pc_utils import bbox_corner_dist_measure

# boxes are axis aigned 2D boxes of shape (n,5) in FLOAT numbers with (x1,y1,x2,y2,score)
''' Ref: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
Ref: https://github.com/vickyboy47/nms-python/blob/master/nms.py 
'''
def nms_2d(boxes, overlap_threshold):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    score = boxes[:,4]
    area = (x2-x1)*(y2-y1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)
        suppress = [last-1]
        for pos in range(last-1):
            j = I[pos]
            xx1 = max(x1[i],x1[j])
            yy1 = max(y1[i],y1[j])
            xx2 = min(x2[i],x2[j])
            yy2 = min(y2[i],y2[j])
            w = xx2-xx1
            h = yy2-yy1
            if (w>0 and h>0):
                o = w*h/area[j]
                print('Overlap is', o)
                if (o>overlap_threshold):
                    suppress.append(pos)
        I = np.delete(I,suppress)
    return pick

def nms_2d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    score = boxes[:,4]
    area = (x2-x1)*(y2-y1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])

        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)

        if old_type:
            o = (w*h)/area[I[:last-1]]
        else:
            inter = w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0])))

    return pick

def nms_3d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    area = (x2-x1)*(y2-y1)*(z2-z1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])

        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type:
            o = (l*w*h)/area[I[:last-1]]
        else:
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0])))

    return pick

# @numba.jit(nopython=True)
def nms_3d_faster_samecls(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    cls = boxes[:,7]
    area = (x2-x1)*(y2-y1)*(z2-z1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])
        cls1 = cls[i]
        cls2 = cls[I[:last-1]]

        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type:
            o = (l*w*h)/area[I[:last-1]]
        else:
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)
        o = o * (cls1==cls2)
        # import ipdb
        # ipdb.set_trace()

        I = np.delete(I, np.concatenate((np.array([last-1]), np.where(o>overlap_threshold)[0])))

    return pick

# @numba.jit(nopython=True)
@torch.no_grad()
def nms_3d_faster_samecls_torch(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    cls = boxes[:,7]
    area = (x2-x1)*(y2-y1)*(z2-z1)
    _, order = score.sort(0, descending=True)
    pick = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            pick.append(i)
            break
        i = order[0].item()
        pick.append(i)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        zz1 = z1[order[1:]].clamp(max=z1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        zz2 = z2[order[1:]].clamp(max=z2[i])
        l = (xx2-xx1).clamp(min=0)
        w = (yy2-yy1).clamp(min=0)
        h = (zz2-zz1).clamp(min=0)
        if old_type:
            o = (l*w*h)/area[order[1:]]
        else:
            inter = l*w*h
            o = inter / (area[i] + area[order[1:]] - inter)
        cls1, cls2 = cls[i], cls[order[1:]]
        o = o * (cls1==cls2)
        idx = (o <= overlap_threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return pick

def nms_crnr_dist(boxes, conf, overlap_threshold):
        
    I = np.argsort(conf)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)        
        
        scores = []
        for ind in I[:-1]:
            scores.append(bbox_corner_dist_measure(boxes[i,:], boxes[ind, :]))

        I = np.delete(I, np.concatenate(([last-1], np.where(np.array(scores)>overlap_threshold)[0])))

    return pick

if __name__=='__main__':
    a = np.random.random((100,5))
    print(nms_2d(a,0.9))
    print(nms_2d_faster(a,0.9))
