import motmetrics as mm
import pandas as pd

# 读取ground truth和结果文件
gt = pd.read_csv('/home/suml/docker/su/Yolov5-Deepsort/gt.txt', header=None)
res = pd.read_csv('/home/suml/docker/su/Yolov5-Deepsort/gt.txt', header=None)

# 创建一个accumulator对象
acc = mm.MOTAccumulator(auto_id=True)

# 假设文件格式为：<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
for frame in gt[0].unique():
    gt_frame = gt[gt[0] == frame]
    res_frame = res[res[0] == frame]

    # 提取ID和位置
    gt_ids = gt_frame[1].values
    gt_positions = gt_frame[[2, 3, 4, 5]].values

    res_ids = res_frame[1].values
    res_positions = res_frame[[2, 3, 4, 5]].values

    # 更新accumulator
    acc.update(
        gt_ids, res_ids,
        mm.distances.iou_matrix(gt_positions, res_positions, max_iou=0.5)
    )

# 计算指标
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['idf1', 'mota', 'motp'], name='DeepSORT')

print(summary)
