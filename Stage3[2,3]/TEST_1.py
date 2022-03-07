import os
from collections import OrderedDict

from inference import Inference
from train_launcher import process_config
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# 关键点的颜色设置
RED = (0, 255, 255)
dataset_joint = ['r_b_paw', 'r_b_knee', 'r_b_elbow', 'l_b_elbow', 'l_b_knee', 'l_b_paw',
                 'tail', 'withers', 'head', 'nose', 'r_f_paw', 'r_f_knee', 'r_f_elbow', 'l_f_elbow', 'l_f_knee',
                 'l_f_paw']

point_pairs = [[0, 1], [1, 2], [2, 6], [3, 6], [4, 3], [5, 4], [6, 7], [10, 11], [11, 12], [12, 7], [7, 13], [13, 14],
               [14, 15], [8, 7], [9, 8], [15, 14]]


# 评估部分
def evaluate(pos_pred_src, pos_gt_src, w_gp, headboxes_src):
    # print('pos_pred:',pos_pred_src)

    # print('pos_gt:',pos_gt_src)

    SC_BIAS = 0.6  # 是一个系数
    threshold = 0.5  # 0.5
    jnt_visible = w_gp

    head = 8  # head=9
    withers = 7
    tail = 6
    nose = 9

    lfelbow = 13
    lfknee = 14
    lfpaw = 15
    lbelbow = 3
    lbknee = 4
    lbpaw = 5

    rfelbow = 12
    rfknee = 11
    rfpaw = 10
    rbelbow = 2
    rbknee = 1
    rbpaw = 0

    # 处理下预测得到的关键点坐标（16，2，n） #（4，n）
    '''
    pos_pred_src = np.transpose(pos_pred_src,(2,0,1))
    bbox= np.transpose(bbox,(1,0))
    
    for k in range(len(bbox)):
        for m in range(16):
            pos_pred_src[k,m,0] = (pos_pred_src[k,m,0] / 256 * (bbox[k][2]-bbox[k][0]))+bbox[k][0]
            pos_pred_src[k,m,1] = (pos_pred_src[k,m,1] / 256 * (bbox[k][3]-bbox[k][1]))+bbox[k][1]

    print('pos_pred_str:',pos_pred_src)
    #print('o:',pos_pred_src[0])
    i = 0
    img = cv2.imread(os.path.join('E:/jiayou',test_name[4]))

    for coord in pos_pred_src[4]:
        i += 1
        keypt = (int(coord[1]), int(coord[0]))
        # text_loc = (keypt[0] + 5, keypt[1] + 7)
        cv2.circle(img, keypt, 13, RED, -1)
    # cv2.putText(img, str(i), text_loc, cv2.FONT_HERSHEY_DUPLEX, 0.5, RED, 1, cv2.LINE_AA)
    plt.imshow(img / 255)
    plt.show()

    pos_pred_src = np.transpose(pos_pred_src, (1, 2 ,0))
    print('pos_pred:',np.array(pos_pred_src).shape)
    '''
    # print('jnt_visible:',jnt_visible)

    uv_error = pos_pred_src - pos_gt_src
    # print('ur_error:',uv_error)
    # (16, 2, 2958)

    uv_err = np.linalg.norm(uv_error, axis=1)
    # print('headboxes_src:',np.array(headboxes_src).shape)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]

    headsizes = np.linalg.norm(headsizes, axis=0)
    # print('headsize:', headsizes)
    headsizes *= SC_BIAS

    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    # print('scaled_uv_err:', scaled_uv_err)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)

    jnt_count = np.sum(jnt_visible, axis=1)  ##there are sum is : have different keypoint
    less_than_threshold = np.multiply((scaled_uv_err <= threshold), jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)
    print('PCKh:', PCKh)

    rng = np.arange(0, 0.5 + 0.01, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]  # start: 0.0, end: 0.5
        less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                          jnt_visible)
        pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1),
                                 jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:9] = False  # TRUE
    PCKh.mask[8] = False

    jnt_count = np.ma.array(jnt_count, mask=False)
    # print('jnt_count:',jnt_count)
    print('jnt_count:',
          jnt_count)  # jnt_count: [2115 2485 2889 2888 2478 2119 2878 2932 2932 2932 2928 2935 2944 2944 2932 2909]
    jnt_count.mask[6:9] = False  # true
    jnt_count.mask[8] = False

    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)
    '''
    print('jnt_ratio:',jnt_ratio)
    jnt_ratio: [0.05503512880562061 0.06466302367941712 0.07517564402810305
                0.0751496226906063 0.06448087431693988 0.055139214155607595 - - --
        0.07629456154046318 0.07629456154046318 0.0761904761904762
                0.07637262555295342 0.07660681759042415 0.07660681759042415
                0.07629456154046318 0.07569607077803799]
    '''

    name_value = [
        ('Head', PCKh[head]),
        ('withers', PCKh[withers]),
        ('nose', PCKh[nose]),
        ('tail', PCKh[tail]),
        ('Shoulder', 0.5 * (PCKh[lfelbow] + PCKh[rfelbow])),
        ('Elbow', 0.5 * (PCKh[rfknee] + PCKh[lfknee])),
        ('Wrist', 0.5 * (PCKh[rfpaw] + PCKh[lfpaw])),
        ('Hip', 0.5 * (PCKh[rbelbow] + PCKh[lbelbow])),
        ('Knee', 0.5 * (PCKh[lbknee] + PCKh[rbknee])),
        ('Ankle', 0.5 * (PCKh[rbpaw] + PCKh[lbpaw])),
        ('Mean', np.sum(PCKh * jnt_ratio)),
        ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
    ]
    name_value = OrderedDict(name_value)

    return name_value, name_value['Mean']


def show_prections(img, predictions, item):
    # 展示预测的内容：
    i = 0
    '''
    print('i:',i)
    print(predictions)
    '''
    print('predictions:', predictions)
    for coord in predictions:
        i += 1
        keypt = (int(coord[1]), int(coord[0]))
        print('coord[1]:', coord[1])
        # text_loc = (keypt[0] + 5, keypt[1] + 7)
        cv2.circle(img, keypt, 3, RED, -1)
    # cv2.putText(img, str(i), text_loc, cv2.FONT_HERSHEY_DUPLEX, 0.5, RED, 1, cv2.LINE_AA)
    # cv2.imwrite('F:/wenxin_/original_datasets/resize/' + str(item) + '.jpg', img)
    '''
    plt.imshow(img/255)
    plt.xlabel('show')
    plt.show()
    '''
    # cv2.imshow('img', imgsets[j])
    # cv2.waitKey(0)


def _crop_data(height, width, box, boxp=0.05):
    """ Automatically returns a padding vector and a bounding box given
    the size of the image and a list of joints.
    Args:
        height		: Original Height
        width		: Original Width
        box			: Bounding Box
        joints		: Array of joints
        boxp		: Box percentage (Use 20% to get a good bounding box)
    """

    padding = [[0, 0], [0, 0], [0, 0]]

    # 试图找到一个合适的初始的bbox外边界
    crop_box = [box[0] - int(boxp * (box[2] - box[0])), box[1] - int(boxp * (box[3] - box[1])),
                box[2] + int(boxp * (box[2] - box[0])), box[3] + int(boxp * (box[3] - box[1]))]
    if crop_box[0] < 0: crop_box[0] = 0
    if crop_box[1] < 0: crop_box[1] = 0
    if crop_box[2] > width - 1: crop_box[2] = width - 1
    if crop_box[3] > height - 1: crop_box[3] = height - 1

    # 新框的大小（新框的高和宽）
    new_h = int(crop_box[3] - crop_box[1])
    new_w = int(crop_box[2] - crop_box[0])

    crop_box = [crop_box[0] + new_w // 2, crop_box[1] + new_h // 2, new_w, new_h]
    # 通过以上操作，得到初始的一个还不错的crop_box

    if new_h > new_w:
        bounds = (crop_box[0] - new_h // 2, crop_box[0] + new_h // 2)
        if bounds[0] < 0:
            padding[1][0] = abs(bounds[0])
        if bounds[1] > width - 1:
            padding[1][1] = abs(width - bounds[1])
    elif new_h < new_w:
        bounds = (crop_box[1] - new_w // 2, crop_box[1] + new_w // 2)
        if bounds[0] < 0:
            padding[0][0] = abs(bounds[0])
        if bounds[1] > width - 1:
            padding[0][1] = abs(height - bounds[1])
    crop_box[0] += padding[1][0]
    crop_box[1] += padding[0][0]
    return padding, crop_box


def _crop_img(img, padding, crop_box, ground_points):
    """ Given a bounding box and padding values return cropped image
    Args:
        img			: Source Image
        padding	: Padding
        crop_box	: Bounding Box
    """
    tail_key = ground_points[6]
    # print('tail_key:',tail_key)
    # padding = [[0, 0], [0, 0], [0, 0]]
    img = np.pad(img, padding, mode='constant')  # 我个人认为：并未进行padding（我找了几个例子：是都没有进行padding的，都为0）
    img_pad = img

    img_shape = img.shape
    h = img_shape[0]
    w = img_shape[1]
    # print('img_shape:',img.shape)  #从这里可看出已进行填充了
    max_lenght = max(crop_box[2], crop_box[3])

    # img = img[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
    #    crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]  #这里就是实现了裁剪，一摸一样的

    # [a,b,c,d]
    img_bbox = [crop_box[0] - max_lenght // 2, crop_box[1] - max_lenght // 2, crop_box[0] + max_lenght // 2,
                crop_box[1] + max_lenght // 2]

    '''
    if (crop_box[1] + max_lenght // 2) > h :
        img_bbox = [ crop_box[0] - max_lenght // 2,crop_box[1] - max_lenght // 2 ,crop_box[0] + max_lenght // 2,h-50]

    if (crop_box[1] - max_lenght // 2 ) == 0 :
        img_bbox = [crop_box[0] - max_lenght // 2, crop_box[1] - max_lenght // 2 + 50, crop_box[0] + max_lenght // 2, h - 50]

    if w - (crop_box[0] + max_lenght // 2) <= 5:
        img_bbox = [crop_box[0] - max_lenght // 2, crop_box[1] - max_lenght // 2 + 50, crop_box[0] + max_lenght // 2 - 20,h -50]

    if  (tail_key[1] - (crop_box[1] - max_lenght // 2 ))<= 150:
        img_bbox = [crop_box[0] - max_lenght // 2, crop_box[1] - max_lenght // 2 , crop_box[0] + max_lenght // 2-20, h - 50]
    '''
    img = img[img_bbox[1]:img_bbox[3],
          img_bbox[0]:img_bbox[2]]

    '''
    cv2.rectangle(img_pad, (int(img_bbox[0]), int(img_bbox[1])), (int(img_bbox[2]), int(img_bbox[3])), (0, 255, 255),
                  thickness=12)
    #cv2.imwrite('E:/Expermin/rectage/' + str(i) + '.jpg', img_pad)  之前跑的内容
    cv2.imwrite('E:/Lab_6/rectage/' + str(i) + '.jpg', img_pad)
    '''
    return img, img_bbox, img_shape


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    print('| Arch ' + ' '.join(['| {}'.format(name) for name in names]) + ' |')
    print('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    print('| ' + full_arch_name + ' ' + ' '.join(['| {:.3f}'.format(value) for value in values]) + ' |')


test_amount = 315  # 46(在训练集抽的) #239  #88


def vis(img_cv2, points, point_pairs, epoch):
    img_cv3 = img_cv2
    img_cv3 = np.clip(img_cv3, 0, 255)
    img_cv2_copy = np.copy(img_cv3)

    img_cv2_copy_1 = img_cv3
    img_cv3_1 = np.copy(img_cv2_copy_1)
    for idx in range(len(points)):
        cv2.circle(img_cv2_copy_1,
                   tuple(points[idx]),
                   radius=3,
                   color=(255, 0, 255),
                   thickness=-1,
                   lineType=cv2.FILLED)
        # Draw Skeleton
    for pair in point_pairs:
        partA = pair[0]
        partB = pair[1]
        if list(points[partA])[0] != -1 and list(points[partA])[1] != -1 and list(points[partB])[0] != -1 and \
                list(points[partB])[1] != -1:
            cv2.line(img_cv3_1,
                     tuple(points[partA]),
                     tuple(points[partB]),
                     color=(0, 255, 255), thickness=2)
            cv2.circle(img_cv3_1,
                       tuple(points[partA]),
                       radius=3,
                       color=(0, 0, 255),
                       thickness=-1,
                       lineType=cv2.FILLED)

    plt.figure(figsize=[50, 50])
    plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(img_cv3, cv2.COLOR_BGR2RGB))
    plt.imshow(img_cv3_1)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_cv2_copy_1)
    # plt.imshow(cv2.cvtColor(img_cv2_copy, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig('E:/zwx/STAGE3_NEW/Stage3[2,3]/heatmap/' + str(epoch))
    # plt.savefig('D:/home_program/xiangmu_1/data_test/VIS_ZJ/key_vis/'+ str(epoch))
    # plt.show()


if __name__ == '__main__':
    print('--Parsing Config File')
    params = process_config('config.cfg')
    # 设置要测试多少数据

    valid_data_file = 'F:/zwx/STAGE3_NEW/Stage3[2,3]/dataset_val.txt'
    test_img = np.zeros((test_amount, 256, 256, 3), dtype=np.float32)

    i = 0
    bbox = []
    box_final = []

    groud_truth_joints = []
    pred = []
    w_ground = []
    test_name = []
    headboxes_src = []
    padding = []
    padding_size = []
    original_size = []

    # 从txt文件中读出这些信息(标准数据集)
    input_file = open(valid_data_file, 'r')
    print('READING VALID DATA')
    for line in input_file:
        headdist = []
        line = line.strip()
        line = line.split(' ')
        name = line[0]
        test_name.append(name)

        box = list(map(int, line[1:5]))
        bbox.append(box)

        joints = list(map(int, line[5:37]))  # 这里就是单纯的列表
        joints = np.reshape(joints, (-1, 2))

        # print('name:',name)
        # print('joints:',joints)
        headdist.append(joints[8])
        headdist.append(joints[9])
        headboxes_src.append(headdist)

        '''
        用框找到的头部的两个关键点（两者之间的距离）
        headlist = list(map(int, line[37:39]))
        headdist.append(headlist)
        headlist = list(map(int, line[39:]))
        headdist.append(headlist)
        headboxes_src.append(headdist)
        '''

        groud_truth_joints.append(joints)

        w = [1] * joints.shape[0]

        for j in range(joints.shape[0]):
            if np.array_equal(joints[j], [-1, -1]) or (joints[i][0] < 0 and joints[i][1] < 0):
                w[j] = 0
            if line[37:] != [] and len(line[37:]) == 16:
                w = list(map(int, line[37:]))

        w_ground.append(w)

    original_joints = groud_truth_joints
    # print('shape:',np.array(original_joints).shape) #(88,16,2)

    # 将得到的各种列表都转化为做pck要求的
    w_ground = np.transpose(w_ground, [1, 0])  # (16,12)
    groud_truth_joints = np.transpose(np.array(groud_truth_joints), (1, 2, 0))  # (16,2,12)
    headboxes_src = np.transpose(headboxes_src, (1, 2, 0))

    # 拿到要进行预测的图像数据
    while i in range(len(test_name)):
        img_load = 'F:/zwx/STAGE3_NEW/Stage3[2,3]/Images'
        img = cv2.imread(os.path.join(img_load, test_name[i]))
        img_or = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size.append(img_or.shape)

        padd, cbox = _crop_data(img.shape[0], img.shape[1], bbox[i], boxp=0.2)
        padding.append(padd[0])

        img, img_bbox, img_cropsize = _crop_img(img, padd, cbox, original_joints[i])
        # cv2.imwrite('G:/Expermin/cro/' + str(i) + '.jpg', img)
        padding_size.append(img_cropsize)

        box_final.append(img_bbox)  # 这里就是crop的bbox

        # print('i:',i)
        # print('test_name:',test_name[i])
        # print('img_crop:',img.shape)
        # print('img_or.shape:', np.array(img_or).shape)
        '''
        cv2.rectangle(img_or, (int(img_bbox[0]), int(img_bbox[1])), (int(img_bbox[2]), int(img_bbox[3])), (0, 255, 255), thickness=12)
        cv2.imwrite('E:/zwx/STAGE3_NEW/Stage3[2,3]/1' + str(i) + '.jpg', img_or)
        '''
        # print('img_box:',img_bbox)

        # print('box_final.shape:',np.array(box_final).shape)
        '''
        center1 = (img_bbox[2], img_bbox[0])
        cv2.circle(img_or, center1, radius=20, color=(0, 255, 255), thickness=-1)
        center2 = (img_bbox[3], img_bbox[1])
        cv2.circle(img_or, center2, radius=20, color=(0, 255, 0), thickness=-1)
        plt.imshow(img_or)
        plt.xlabel('crop_')
        plt.show()
        
        plt.imshow(img)
        plt.xlabel('crop_img')
        plt.show()
        '''

        # img = cv2.imread(os.path.join(params['img_directory'], '005808361.jpg'))
        # the bounding boxes used below are ground truth from dataset.txt

        # img1 = np.copy(img)[80:711, 798:1167]
        # img2 = np.copy(img)[66:651, 310:721]

        test_im = cv2.resize(img, (256, 256))
        '''
        if i == 0:
            cv2.imwrite('E:/Expermin/rectage/' + str(i) + 'jiayou' + '.jpg', test_img[i])
        '''

        test_img[i] = test_im

        i = i + 1

        # print('test_img.shape:',test_img.shape)  (12,256,256,3)

    # box_final = np.transpose(box_final, [1, 0])
    # print('box_final.shape:',np.array(box_final).shape)  #(4,12)

    model = Inference()
    for item in range(len(test_name)):
        predictions = model.predictJoints(test_img[item], mode='gpu')
        # show_prections(test_img[item], predictions,item)
        predictions_1 = predictions
        for m in range(16):
            # print('pead_1[m][1]:',predictions_1[m][1])

            # predictions_1[m][0] = predictions_1[m][0] - padding[item][0]
            predictions_1[m][0] = ((predictions_1[m][0] / 256) * (box_final[item][3] - box_final[item][1])) + \
                                  box_final[item][1] - padding[item][0]
            predictions_1[m][1] = ((predictions_1[m][1] / 256) * (box_final[item][2] - box_final[item][0])) + \
                                  box_final[item][0]

        i = 0
        x = cv2.imread(os.path.join('F:/zwx/STAGE3_NEW/Stage3[2,3]/Images', test_name[item]))
        # vis(x,predictions_1,point_pairs,str(item))
        '''
        for coord in predictions_1:
            i += 1
            keypt = (int(coord[1]), int(coord[0]))
            cv2.circle(x, keypt, 11, (0,255,255), -1)
            #text_loc = (keypt[0] + 5, keypt[1] + 7)
            cv2.putText(x, str(dataset_joint[i-1]), keypt,cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 1, cv2.LINE_AA)
        cv2.imwrite('E:/2020.7.18/aug_7/results/original/train_/' + str(item) + '.jpg', x )
        '''

        for m in range(16):
            c = predictions_1[m][0]
            predictions_1[m][0] = predictions_1[m][1]
            predictions_1[m][1] = c
        '''
            plt.imshow(x/255)
            plt.xlabel('jiayou')
            plt.show()
        '''
        # print('item:',item)
        # print('predictions_1:',predictions_1)
        pred.append(predictions_1)

    # print('pred.shape:',np.array(pred).shape) #(12,16,2)
    pred = np.transpose(np.array(pred), (1, 2, 0))
    '''
    predictions = model.predictJoints(test_img, mode='gpu')
    show_prections(test_img, predictions)
    '''

    # 下面写评估函数：用PCK来做：
    # print('pred.shape:',np.array(pred).shape)
    # print('pred:',pred)
    # print('grpund_shape:',np.array(groud_truth_joints).shape)
    # print('ground_joints:',groud_truth_joints)
    # print('head。shape：',np.array(headboxes_src).shape)
    # print('headbox:',headboxes_src)

    '''
    headboxes_src_1 = []
    for i in range(7):
        bbox1 = [2040, 727, 2477, 1206]
    headboxes_src_1.append(bbox1)
    bbox2 = [1389, 1351, 1653, 1742]
    headboxes_src_1.append(bbox2)
    bbox3 = [1316, 1115, 1571, 1430]
    headboxes_src_1.append(bbox3)
    bbox4 = [1128, 636, 1431, 1066]
    headboxes_src_1.append(bbox4)
    bbox5 = [874, 1324, 1137, 1676]
    headboxes_src_1.append(bbox5)
    bbox6 = [913, 1285, 1280, 1682]
    headboxes_src_1.append(bbox6)
    bbox7 = [1216, 1130, 1577, 1466]
    headboxes_src_1.append(bbox7)
    print('shape:',np.array(headboxes_src_1).shape)
    '''

    name_values, b = evaluate(pred, groud_truth_joints, w_ground, headboxes_src)

    model_name = 'pose_stack_hourglass'
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)
