from efficientnet.tfkeras import EfficientNetB7 as effnetb7

base_model = effnetb7(include_top=False,
                     weights = None,
                     input_shape=(224,224,3))