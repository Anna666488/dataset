import os
import cv2
import time
import numpy as np


rootdir = './images_lip/'
images = os.listdir(rootdir)
count = 0
for image in images:
   count += 1
   t = time()
   img = cv2.imread(rootdir + str(image))
   sp = img.shape
img_hg = cv2.resize(img, (256,256))
img_res = cv2.resize(img, (256,256))
img_hg = cv2.cvtColor(img_hg, cv2.COLOR_BGR2RGB)
hg = self.HG.Session.run(self.HG.pred_sigmoid, feed_dict = {self.HG.img: np.expand_dims(img_hg/255, axis = 0)})
j = np.ones(shape = (self.params['num_joints'],2)) * -1
if plt_hm:
hm = np.sum(hg[0], axis = 2)
hm = np.repeat(np.expand_dims(hm, axis = 2), 3, axis = 2)
hm = cv2.resize(hm, (sp[0],sp[1]))
img_res = img_res / 255 + hm
for i in range(len(j)):
idx = np.unravel_index( hg[0,:,:,i].argmax(), (64,64))
j[i] = np.asarray(idx) * 256 / 64
cv2.circle(img_res, center = tuple(j[i].astype(np.int))[::-1], radius= 5, color= self.color[i][::-1], thickness= -1)
if plt_l:
for i in range(len(self.links)):
l = self.links[i]['link']
good_link = True
for p in l:
if np.array_equal(j[p], [-1,-1]):
good_link = False
if good_link:
pos = self.givePixel(l, j)
cv2.line(img_res, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.links[i]['color'][::-1], thickness = 5)
fps = 1/(time()-t)
if debug:
framerate.append(fps)
cv2.putText(img_res, 'FPS: ' + str(fps)[:4], (60, 40), 2, 2, (0,0,0), thickness = 2)
cv2.imwrite('./result/' + str(count) + '.jpg', img_res)
if count == 100:
break