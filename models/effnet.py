from keras.applications import DenseNet201, ResNet152V2, InceptionV3, InceptionResNetV2
from efficientnet.tfkeras import EfficientNetB7 as effnetb7


# Initialize DenseNet201 base model
dense_net_model = DenseNet201(include_top=False,
                              weights=None,
                              input_shape=(224, 224, 3))


# Initialize ResNet152V2 base model
res_net_model = ResNet152V2(include_top=False,
                            weights=None,
                            input_shape=(224, 224, 3))


# Initialize EfficientNetB7 base model
eff_net_model = effnetb7(include_top=False,
                         weights=None,
                         input_shape=(224, 224, 3))


# Initialize InceptionV3 base model
inception_v3_model = InceptionV3(include_top=False,
                                 weights=None,
                                 input_shape=(224, 224, 3))


# Initialize InceptionResNetV2 base model
inception_resnet_v2_model = InceptionResNetV2(include_top=False,
                                               weights=None,
                                               input_shape=(224, 224, 3))

