"""
测试读取tfrecord.zip文件
"""
import tensorflow as tf
from matplotlib import pyplot as plt
import os


def _parse_record(example_photo):
    """
    :param example_photo:  是序列化后的数据
    :return: 反序列化的数据
    """
    # 定义一个解析序列的features
    expected_features = {
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "lung_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "shape": tf.io.FixedLenFeature(shape=[2], dtype=tf.int64)
    }
    # 反序列化
    parsed_features = tf.parse_single_example(example_photo, features=expected_features)
    # 将数据图片从字符串解析还原
    feature = {
        "image": tf.reshape(tf.decode_raw(parsed_features["image"], tf.float32), shape=parsed_features["shape"]),
        "mask": tf.reshape(tf.decode_raw(parsed_features["mask"], tf.int16), shape=parsed_features["shape"]),
        # "lung_mask": tf.reshape(tf.decode_raw(parsed_features["lung_mask"], tf.int16), shape=parsed_features["shape"]),
        # "shape": parsed_features["shape"]
    }
    return feature


def tfrecord_reader_dataset(filenames, n_readers=6, batch_size=32, n_parse_threads=6, shuffle_buffer_size=100):
    # 1.构建filenames的dataset
    dataset_filenames = tf.data.Dataset.list_files(filenames).repeat()
    # 2.构建全部文件内容的dataset
    dataset_filecontent = dataset_filenames.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type="GZIP"),
        cycle_length=n_readers  # 读取文件的并行数
    )
    dataset_filecontent = dataset_filecontent.shuffle(shuffle_buffer_size)
    # 3.构建样本的dataset
    dataset = dataset_filecontent.map(_parse_record,  # 负责将example解析并反序列化处理的函数
                                      num_parallel_calls=n_parse_threads  # 处理样本的并行线程数量
                                      )
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    return dataset


if __name__ == "__main__":
    # 1.获取所有TFRecord文件的列表
    dir = "D:\\data\\COVID19\\TFRecord"
    files_name = os.listdir(dir)
    train_files_name = [os.path.join(dir, name) for name in files_name if name.startswith("train")]

    # dataset = dataset.map(_parse_record)
    dataset = tfrecord_reader_dataset(train_files_name, batch_size=32)
    iterator = dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        features = sess.run(iterator.get_next())
        imgs = features["image"]
        masks = features["mask"]
        lung_masks = features["lung_mask"]

        plt.imshow(imgs[0], cmap="gray")
        plt.show()

        plt.imshow(masks[0], cmap="gray")
        plt.show()

        plt.imshow(lung_masks[0], cmap="gray")
        plt.show()

