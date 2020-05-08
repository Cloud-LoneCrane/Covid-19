from Utils.parse import tfrecord_reader_dataset
import os
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    epoch = 1
    # 1.获取所有TFRecord文件的列表
    dir = "D:\\data\\COVID19\\TFRecord"
    files_name = os.listdir(dir)
    train_files_name = [os.path.join(dir, name) for name in files_name if name.startswith("train")]
    test_files_name = [os.path.join(dir, name) for name in files_name if name.startswith("test")]

    train_dataset = tfrecord_reader_dataset(train_files_name, batch_size=32)
    test_dataset = tfrecord_reader_dataset(test_files_name, batch_size=32).repeat(epoch)

    # 如果是标准tf，那么使用方式是创建一个dataset迭代器，并且在sess中每次run迭代器即可
    iterator = test_dataset.make_one_shot_iterator()

    # 如果是keras，那么直接在fit中直接fit(train_dataset, test_dataset)即可
    # keras 的可选方式：是通过传入dataset的迭代器
    # model.fit(dataset.make_one_shot_iterator(), epochs=10, steps_per_epoch=10)
    next_element = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                features = sess.run(next_element)
                imgs = features["image"]
                masks = features["mask"]

                plt.imshow(imgs[0], cmap="gray")
                plt.show()

                plt.imshow(masks[0], cmap="gray")
                plt.show()
            except tf.errors.OutOfRangeError:
                break
