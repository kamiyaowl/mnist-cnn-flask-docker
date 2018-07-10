#%% インポート関連

import tensorflow as tf
# tf.enable_eager_execution()
print(tf.__version__)
print(tf.test.is_built_with_cuda())


from tensorflow.python import keras
print(keras.__version__)
from tensorflow.python.keras.callbacks import EarlyStopping

import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
np.set_printoptions(threshold=100)


#%% データを読みこみ
(x_train_src, y_train_src), (x_test_src, y_test_src) = keras.datasets.mnist.load_data()
print(x_train_src.shape)
print(y_train_src.shape)
print(x_test_src.shape)
print(y_test_src.shape)

# channel last前提で処理
keras.backend.image_data_format()

#%% numpy配列に変換
input_shape =(28,28,1)
x_train = x_train_src.reshape(x_train_src.shape[0], 28, 28, 1)
x_test = x_test_src.reshape(x_test_src.shape[0], 28, 28, 1)

# テストデータを正規化
x_train = x_train / 255.0
x_test = x_test / 255.0
# 分類問題なのでone-hot enc
y_train = keras.utils.to_categorical(y_train_src, 10)
y_test = keras.utils.to_categorical(y_test_src, 10)

print(x_train.shape)
print(x_test.shape)

# 画像を表示、arrは28x28x1の正規化されたもの
def convert_image(arr, show=True, title="", w=28, h=28):
    img = Image.fromarray(arr.reshape(w,h) * 255.0)
    if show:
        plt.imshow(img)
        plt.title(title)
    return img

def convert_images(srcs, length, show=True, cols=5, w=28, h=28):
    rows = int(length / cols + 1)
    dst = Image.new('1', (w * cols, h * rows))
    for j in range(rows):
        for i in range(cols):
            ptr = i + j * cols
            img = convert_image(srcs[ptr], show=False, w=w, h=h)
            dst.paste(img, (i * w, j * h))
    if show:
        plt.imshow(dst)
    return dst

plt.subplot(1,2,1)
convert_images(x_train, 50,)
plt.subplot(1,2,2)
convert_images(x_test, 50,)
plt.show()

#%%　モデル構築・学習
def MNISTConvModel(input_shape, predicates_class_n):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')(inputs)
    x = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten()(x) # 2D(12*12*64) -> 1d(9216)
    x = keras.layers.Dense(120, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    predicates = keras.layers.Dense(predicates_class_n, activation='softmax')(x)
    return keras.models.Model(inputs=inputs, outputs=predicates)

model = MNISTConvModel(input_shape=input_shape, predicates_class_n=10)
model.summary()

# モデルをコンパイルして実行
batch_size = 128
epochs = 20

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adadelta',
    metrics=['accuracy']
)
tensorboard_cb = keras.callbacks.TensorBoard(log_dir="./tflogs/", histogram_freq=1)
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_cb],
)

#%% 学習結果の確認
plt.subplot(2,1,1)
plt.plot(range(epochs), history.history['acc'], label='acc')
plt.plot(range(epochs), history.history['val_acc'], label='val_acc')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.subplot(2,1,2)
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#%% 性能
scores = model.evaluate(x_test, y_test, verbose=2)
print('loss', scores[0], 'accuracy', scores[1])

#%% モデルの保存
model.save('MNISTConv.h5')
