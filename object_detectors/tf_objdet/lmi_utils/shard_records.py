import tensorflow as tf
import argparse
import os


def main(s, r, w):
    raw_dataset = tf.data.TFRecordDataset(r)

    for i in range(s):
        path = os.path.join(w, f"shard-{i}.tfrecord")
        print(path)
        writer = tf.data.experimental.TFRecordWriter(path)
        writer.write(raw_dataset.shard(s, i))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--shards", required=True, help="number of shards")
    ap.add_argument("-r", "--record_path", required=True, help="path of record file")
    ap.add_argument("-w", "--write_path", required=True, help="where to write shards")

    args = vars(ap.parse_args())
    s = int(args["shards"])
    r = args["record_path"]
    w = args["write_path"]
    main(s, r, w)