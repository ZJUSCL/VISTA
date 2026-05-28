import random
import math
from PIL import Image


def generate_weighted_number(seed_value=None):
    # 1. 设置 Seed (种子)
    # 注意：通常 Seed 只需在程序开始时设置一次。
    # 如果每次调用函数都重置相同的 Seed，你将永远得到相同的结果。
    if seed_value is not None:
        random.seed(seed_value)

    # 2. 生成 0.0 到 1.0 之间的随机浮点数
    rand_val = random.random()

    # 3. 判断概率
    # 小于 0.2 (20%) -> 0
    # 大于等于 0.2 (80%) -> 1
    if rand_val < 0.2:
        return 0
    else:
        return 1

def dynamic_crop_with_gt(image: Image.Image, gt_bbox: list, target_wh: list):
    """
    动态裁剪图像。
    改进点：如果 target_wh 小于物体尺寸，会自动扩大裁剪框以包含物体。
    """

    W_img, H_img = image.size
    target_w, target_h = target_wh
    w1, h1, w2, h2 = gt_bbox

    # 1. 将 GT Box 转换为绝对像素坐标
    x_min = max(0, w1 * W_img)
    y_min = max(0, h1 * H_img)
    x_max = min(W_img, w2 * W_img)
    y_max = min(H_img, h2 * H_img)

    obj_w = x_max - x_min
    obj_h = y_max - y_min

    # --- Corner Case 处理逻辑 ---
    # 定义实际使用的裁剪尺寸 (actual_crop_w/h)
    # 如果目标尺寸小于物体，强制使用物体尺寸（向上取整以防舍入误差切掉边缘）
    actual_crop_w = target_w
    actual_crop_h = target_h

    if target_w < obj_w:
        actual_crop_w = int(math.ceil(obj_w))

    if target_h < obj_h:
        actual_crop_h = int(math.ceil(obj_h))

    # 再次检查：防止物体本身就比原图还大（虽然理论上 GT 0-1 不会越界，但为了健壮性）
    actual_crop_w = min(actual_crop_w, W_img)
    actual_crop_h = min(actual_crop_h, H_img)
    # ---------------------------

    # 2. 计算裁剪框左上角 (crop_x, crop_y) 的合法范围
    # 逻辑：
    # crop_x <= x_min (必须包含左边)
    # crop_x >= x_max - actual_crop_w (必须包含右边)
    # 且不越出图片边界

    min_crop_x = int(max(0, x_max - actual_crop_w))
    max_crop_x = int(min(W_img - actual_crop_w, x_min))

    min_crop_y = int(max(0, y_max - actual_crop_h))
    max_crop_y = int(min(H_img - actual_crop_h, y_min))

    # 这里的 min 可能大于 max 的唯一情况是：物体尺寸 > 图片尺寸 (前面已做截断处理)
    # 所以这里为了安全，做一个修正，如果 min > max，说明无法完全包含，强行取 min
    if min_crop_x > max_crop_x: max_crop_x = min_crop_x
    if min_crop_y > max_crop_y: max_crop_y = min_crop_y

    cropped_images = []
    new_bboxes = []

    for _ in range(8):
        # 3. 随机采样
        # 如果 min == max (即物体刚好卡住裁剪框，或没有移动空间)，则只有一种结果
        crop_x = random.randint(min_crop_x, max_crop_x)
        crop_y = random.randint(min_crop_y, max_crop_y)

        # 裁剪
        crop_box = (crop_x, crop_y, crop_x + actual_crop_w, crop_y + actual_crop_h)
        img_crop = image.crop(crop_box)

        # 4. 计算相对坐标
        # 注意分母要是 actual_crop_w，因为裁剪尺寸变了
        new_w1 = (x_min - crop_x) / actual_crop_w
        new_h1 = (y_min - crop_y) / actual_crop_h
        new_w2 = (x_max - crop_x) / actual_crop_w
        new_h2 = (y_max - crop_y) / actual_crop_h

        new_bbox = [
            max(0.0, min(1.0, new_w1)),
            max(0.0, min(1.0, new_h1)),
            max(0.0, min(1.0, new_w2)),
            max(0.0, min(1.0, new_h2))
        ]

        cropped_images.append(img_crop)
        new_bboxes.append(new_bbox)

    return cropped_images, new_bboxes


# --- 测试代码 ---
if __name__ == "__main__":
    # 创建一个简单的 dummy 图片 (500x500)
    img = Image.new('RGB', (1280, 720), color='white')

    # 假设原图中有一个物体在中心附近 (相对坐标)
    # [w1, h1, w2, h2] -> 绝对坐标约为 [200, 200, 300, 300]
    gt_box = [0.3, 0.3, 0.4, 0.4]

    # 目标裁剪尺寸
    target_wh = [1008, 568]

    try:
        images, bboxes = dynamic_crop_with_gt(img, gt_box, target_wh)

        print(f"Original GT: {gt_box}")
        print("-" * 30)
        for i, bbox in enumerate(bboxes):
            # 格式化输出，保留4位小数
            bbox_str = [f"{x:.4f}" for x in bbox]
            print(f"Crop {i+1} GT: {bbox_str}")
            # images[i].show() # 如果你想查看图片，可以取消注释

    except ValueError as e:
        print(f"Error: {e}")