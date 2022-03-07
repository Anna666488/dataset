# Stacked Hourglass model : TensorFlow implementation
Currently trained on our Pig DataSet.


# Dataset
All images are available in this link:

链接：https://pan.baidu.com/s/1XZKzOpcwcM2v51d-r7fLQg 

提取码：cifx

To create a dataset you need to put every images of your set on the 'img_directory'. 

Add information about your images into the 'training_txt_file':

EXAMPLE:

00003_4050.jpgA 552 73 1140 481 1102 466 1127 398 1140 314 1048 330 1071 404 1061 481 1140 73 720 182 631 285 552 430 901 460 869 446 898 386 749 411 746 465 802 462

00003_4175.jpgA 500 178 1198 468 -1 -1 -1 -1 -1 -1 1047 346 1096 392 1072 466 1198 178 756 252 625 282 500 400 -1 -1 -1 -1 -1 -1 716 410 690 450 745 468

00003_4450.jpgA 467 294 1301 720 -1 -1 -1 -1 -1 -1 1065 643 1036 717 943 720 1301 636 775 294 546 350 467 518 -1 -1 -1 -1 -1 -1 725 546 691 611 658 669

# training

To train a model, make sure to have a 'config.cfg' file in your main directory and a text file with regard to your dataset. Then run train_launcher.py. It will run the training.
