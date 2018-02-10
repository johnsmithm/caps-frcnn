# frcnn + capsule network implementation

[original code](https://github.com/endernewton/tf-faster-rcnn)
[caps layer](https://github.com/naturomics/CapsLayer)
[citypersons dataset](https://bitbucket.org/shanshanzhang/citypersons/src/f44d4e585d51d0c3fd7992c8fb913515b26d4b5a/evaluation/eval_script/?at=default)

### Execute code
* ./experiments/scripts/train_faster_rcnn.sh 0 citypersons vgg16
* ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc_0712 mdlstmvgg16
* ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc12 vgg16
* ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16