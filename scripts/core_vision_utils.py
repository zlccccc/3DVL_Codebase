import numpy as np
import torch
import os

# for bounding box output (.obj)
# p: weight
def output_bounding_box(id_, point, color, heat, file_out, output_face=True, output_line=True): # color=(1,1,1),p=1: red
    # print(p, '<< heatmap')
    p = heat
    if point.shape[-1] == 6:  # convert center-offset to new_point
        pc, sz = point[:3], point[3:] / 2
        point = np.zeros([8, 3])
        off = [[1, 1, 1, 1, -1, -1, -1, -1], \
               [1, 1, -1, -1, 1, 1, -1, -1], \
               [1, -1, 1, -1, 1, -1, 1, -1]]
        for i in range(8):
            for j in range(3):
                point[i, j] = pc[j] + sz[j] * off[j][i]
    # vertex
    assert len(point.shape) == 2
    assert point.shape[0] == 8
    assert point.shape[1] == 3
    for i in range(8):
        print('v %f %f %f %f %f %f' % (point[i][0], point[i][1], point[i][2],
                 1-(1-color[0])*p, 1-(1-color[1])*p, 1-(1-color[2])*p), file=file_out)
    bs = 8 * id_

    # face
    if output_face:
        print('f %d %d %d %d' % (1 + bs, 2 + bs, 4 + bs, 3 + bs), file=file_out)
        print('f %d %d %d %d' % (5 + bs, 6 + bs, 8 + bs, 7 + bs), file=file_out)
        print('f %d %d %d %d' % (1 + bs, 2 + bs, 6 + bs, 5 + bs), file=file_out)
        print('f %d %d %d %d' % (3 + bs, 4 + bs, 8 + bs, 7 + bs), file=file_out)

        print('f %d %d %d %d' % (1 + bs, 3 + bs, 7 + bs, 5 + bs), file=file_out)
        print('f %d %d %d %d' % (2 + bs, 4 + bs, 8 + bs, 6 + bs), file=file_out)

    # line
    if output_line:
        print('l %d %d' % (1 + bs, 2 + bs), file=file_out)
        print('l %d %d' % (2 + bs, 4 + bs), file=file_out)
        print('l %d %d' % (4 + bs, 3 + bs), file=file_out)
        print('l %d %d' % (3 + bs, 1 + bs), file=file_out)

        print('l %d %d' % (5 + bs, 6 + bs), file=file_out)
        print('l %d %d' % (6 + bs, 8 + bs), file=file_out)
        print('l %d %d' % (8 + bs, 7 + bs), file=file_out)
        print('l %d %d' % (7 + bs, 5 + bs), file=file_out)

        print('l %d %d' % (2 + bs, 6 + bs), file=file_out)
        print('l %d %d' % (5 + bs, 1 + bs), file=file_out)
        print('l %d %d' % (4 + bs, 8 + bs), file=file_out)
        print('l %d %d' % (7 + bs, 3 + bs), file=file_out)


# visualize the bounding boxes
def save_bbox_heatmap(bboxes, heatmap, save_base = os.getcwd()+'/heatmap_result', save_name='', kth_input=None, color=[1, 0, 0]): # save_name: text
    #print("bboxes", bboxes.shape)
    #print("heatmap", heatmap.shape)
    #color = color.cpu().numpy()
    save_base = os.path.join(save_base, save_name)
    print(save_base, flush=True)
    if not os.path.exists(save_base):
        os.makedirs(save_base)
    if kth_input is not None:
        for idx in range(heatmap.shape[0]):  # idx: object
            kth = kth_input
            kth = int(kth)
            obj_name = str(idx) + '_' + str(int(kth)) + '.obj'
            real_save_path = os.path.join(save_base, obj_name)
            file_out = open(real_save_path, 'w')
            norm = heatmap[idx][kth].max()
            # print(real_save_path, '<< save path; norm=', norm)
            for _, point in enumerate(bboxes):
                p = (heatmap[idx][kth][_].cpu()*5).numpy()
                p = min(1, p)
                point = point[[0,4,1,5,3,7,2,6], :]
                output_bounding_box(_, point, color, p, file_out)
            file_out.close()
    else:
        raise NotImplementedError()

