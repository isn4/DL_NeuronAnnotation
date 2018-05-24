# Neuron Annotation with UNet, SegNet, and varied Data Augmentation Techniques
Here is the code used to conduct research on neural pathway tracing through SEM zebrafish larvae data. This code was used in the papers "Understanding Neural Pathways in Zebrafish through Deep Learning and High Resolution Electron Microscope Data" and "A Simplified Approach to Deep Learning for Image Segmentation".

## Original Code
This work builds upon the work found in these places:
### UNet
The original TensorFlow/Keras UNet code was written by Marko Jocić and can be found here: https://github.com/jocicmarko/ultrasound-nerve-segmentation.
The paper it is based on is here: https://arxiv.org/abs/1505.04597.
### SegNet
The original TensorFlow code for SegNet was written by Tseng Kuan Lun and can be found here: https://github.com/tkuanlun350/Tensorflow-SegNet.
The paper it is based on is here: https://arxiv.org/abs/1511.00561.

### Flood-Fill & Region Growing Data Augmentation
Our flood-fill and region growing code is found here: https://github.com/kbushman/Image-Segmentation.
The original region growing code was written by someone with the username "DNeems" and can be found here: https://github.com/DNeems/NucNorm/blob/master/regiongrowing.m.
