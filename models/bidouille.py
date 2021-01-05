
from keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D
from keras.layers import Lambda, MaxPooling2D, Dropout
#from keras.applications.resnet50 import ResNet50
from keras_resnet import models as resnet_models
from keras.regularizers import l2
from losses import loss
from keras.models import Model

def add_neck(resnet_model):
    C5 = resnet_model.outputs[-1]
    x = Dropout(rate=0.5)(C5)
    num_filters = 256
    for i in range(3):
        num_filters = num_filters // pow(2, i)
        x = Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, padding='same',
            kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        #print(num_filters)
    return x

def add_head(x):

    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

    return y1, y2, y3

def add_loss(y1, y2, y3, input_size=512, num_classes=80, max_objects=100):
    
    
    output_size = input_size // 4
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))


    loss_ = Lambda(loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    return model

if __name__=='__main__':
    input_size = 512
    output_size = input_size // 4
    image_input = Input(shape=(None,None,3))
    num_classes = 3
    max_objects = 100

    # get resnet model
    #image_input = Input((input_size,input_size,3))
    freeze_bn = True
    resnet = resnet_models.ResNet18(image_input, include_top=False, freeze_bn=freeze_bn)

    x = add_neck(resnet)
    y1, y2, y3 = add_head(x)

    model = add_loss(y1, y2, y3, input_size=input_size, num_classes=num_classes, max_objects=max_objects)

    # XXX = Model(input = resnet.input, output=x)
    print(model.summary())
    model.save("./centernet_resnet18_full.h5")
