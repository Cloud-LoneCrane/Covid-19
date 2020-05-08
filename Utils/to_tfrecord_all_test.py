"""
制作测试集的tfrecord文件
使用COVID19类，获取数据，之后按照将获取到的数据处理成tfrecord的存储格式，写入到tfrecord，保存到磁盘
"""
import os
import tensorflow as tf
from Utils import COVID19


def serialize_example(x, mask, lung_mask):
    """converts x, mask, lung_mask to tf.train.Example and serialize"""
    # 1.构建用于创建Feature对象的 list
    # 注意，value只接受一维数组，所以要将二维的图片x，reshape成一维，或者将二维图片转成字符串
    # 同理，mask和lung_mask同上
    input_features = tf.train.BytesList(value=[x.tobytes()])
    mask_ = tf.train.BytesList(value=[mask.tobytes()])
    lung_mask_ = tf.train.BytesList(value=[lung_mask.tobytes()])
    shape = tf.train.Int64List(value=list(x.shape))
    # 2.构建Feature对象
    features = tf.train.Features(
        feature={
            "image": tf.train.Feature(bytes_list=input_features),
            "mask": tf.train.Feature(bytes_list=mask_),
            "lung_mask": tf.train.Feature(bytes_list=lung_mask_),
            "shape": tf.train.Feature(int64_list=shape)
        }
    )
    # 3.Features组建Example
    example = tf.train.Example(features=features)
    # 4.将Example序列化
    return example.SerializeToString()


def main(Covid19_dir, TFRecord_save_dir):
    # 存储到一个多个tfrecord文件中
    batch_size = 10
    covid19 = COVID19.Covid19(Covid19_dir, batch_size=batch_size)

    # 定义存储的tfrecord文件名，压缩格式
    # 定义存储路径 dir
    dir = TFRecord_save_dir
    if not os.path.exists(dir):
        os.mkdir(dir)

    filename_zip = "test.zip"

    # 定义压缩的可选项options
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    # 打开文件进行写入
    name = os.path.join(dir, filename_zip)
    with tf.io.TFRecordWriter(name, options=options) as writer:
        batch_x, batch_mask, batch_lung_mask = covid19.test_data
        # 1.创建用于Feature对象的List
        # 将batch中的样本逐个取出
        for x, mask, lung_mask in zip(batch_x, batch_mask, batch_lung_mask):
            # 每次写入一个样本的
            writer.write(serialize_example(x, mask, lung_mask))


if __name__ == "__main__":
    # 测试集
    Covid19_dir = "D:\\data\\COVID19\\"
    TFRecord_save_dir = "D:\\data\\COVID19\\TFRecord"
    # 调用main传入两个dir即可自动完成数据转换成TFRecord格式
    main(Covid19_dir, TFRecord_save_dir)

