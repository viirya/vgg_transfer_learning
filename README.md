# Transfer Learning on VGG19

This is a Tensorflow implemention for transfer learning on VGG19. The VGG19 implemention is based on [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg).

By replacing the last fully connected layer in VGG19 network, we can adjust the network output to the required number of classes. The final classification is learned based on the trained features of VGG19.

The trained model can be saved and used to predict images.

>To use the VGG networks, the npy files for [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) or [VGG19 NPY](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) has to be downloaded.

## Usage
Run `train.py` to perform transfer learning on VGG19.

    python train.py -p [base_path_to_images] -t [image_labels] -s [image_labels_list] -o [output_model_file]

The file `image_labels` is the list of image filename and label. E.g.,

    image1.jpg 0
    image2.jpg 1
    ...

The file `image_labels_list` is the full list of image labels.

Run `predict.py` to predict images.

    python predict.py -p [path_to_images] -s [image_labels_list] -m [model_filename]

