#Time: 2020.8.6
#Autor: xiaoxin05
#Aim: 看看评估内容

joint_list = ['r_b_paw','r_b_knee','r_b_elbow','l_b_elbow','l_b_knee','l_b_paw',
             'tail','withers','head','nose','r_f_paw','r_f_knee','r_f_elbow','l_f_elbow','l_f_knee','l_f_paw']

joint_Li = [300,453,500,432,453,564]
joint_Li.mask[6:9] = False