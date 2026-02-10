import os
import shutil
from pathlib import Path
from PIL import Image, ImageOps
import yaml
import numpy as np
import cv2


class YOLOExporter:
    """YOLO格式导出器"""

    def __init__(self):
        # 平滑参数配置（在mask级别进行形态学平滑）
        self.smooth_params = {
            'none': {'kernel_size': 0, 'simplify_epsilon': 0.002},
            'low': {'kernel_size': 3, 'simplify_epsilon': 0.0015},
            'medium': {'kernel_size': 5, 'simplify_epsilon': 0.001},
            'high': {'kernel_size': 7, 'simplify_epsilon': 0.0008},
            'ultra': {'kernel_size': 11, 'simplify_epsilon': 0.0005},
        }
        self.format_type = 'segment'  # 默认导出类型

    def _smooth_polygon_via_mask(self, polygon: list, kernel_size: int) -> np.ndarray:
        """通过渲染到mask再提取的方式平滑多边形（最有效的方法）

        原理：
        1. 将多边形渲染到临时mask
        2. 对mask进行形态学平滑
        3. 从平滑后的mask提取新轮廓
        """
        if kernel_size == 0:
            return np.array(polygon, dtype=np.float64)

        points = np.array(polygon, dtype=np.float64)

        # 计算边界框，创建合适大小的mask
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        # 添加边距
        margin = kernel_size * 2 + 10
        x_min = max(0, int(x_min) - margin)
        y_min = max(0, int(y_min) - margin)
        x_max = int(x_max) + margin
        y_max = int(y_max) + margin

        width = x_max - x_min
        height = y_max - y_min

        # 创建mask并绘制多边形
        mask = np.zeros((height, width), dtype=np.uint8)
        shifted_points = points - np.array([x_min, y_min])
        cv2.fillPoly(mask, [shifted_points.astype(np.int32)], 255)

        # 形态学平滑
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

        # 高斯模糊 + 阈值
        if kernel_size >= 5:
            blur_size = kernel_size | 1
            smoothed = cv2.GaussianBlur(smoothed, (blur_size, blur_size), 0)
            _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

        # 提取新轮廓
        contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if not contours:
            return points

        largest = max(contours, key=cv2.contourArea)
        new_points = largest.reshape(-1, 2).astype(np.float64)

        # 还原坐标偏移
        new_points += np.array([x_min, y_min])

        return new_points

    def _adaptive_simplify(self, points: np.ndarray, epsilon_factor: float) -> np.ndarray:
        """自适应简化多边形"""
        if len(points) < 3:
            return points

        contour = points.reshape(-1, 1, 2).astype(np.float32)
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        return approx.reshape(-1, 2)

    def smooth_polygon(self, polygon: list, smooth_level: str = 'medium') -> list:
        """对多边形进行平滑处理（通过mask级别的形态学操作）

        Args:
            polygon: 多边形点列表 [[x, y], ...]
            smooth_level: 平滑级别 'none', 'low', 'medium', 'high', 'ultra'

        Returns:
            平滑后的多边形点列表
        """
        if not polygon or len(polygon) < 3:
            return polygon

        params = self.smooth_params.get(smooth_level, self.smooth_params['medium'])

        # 通过mask渲染的方式平滑（最有效）
        smoothed = self._smooth_polygon_via_mask(polygon, params['kernel_size'])

        # 简化多边形
        result = self._adaptive_simplify(smoothed, params['simplify_epsilon'])

        return result.tolist()

    def export(self, project: dict, output_dir: str,
               format_type: str = 'segment',
               split_ratio: tuple = (0.8, 0.1, 0.1),
               smooth_level: str = 'medium') -> dict:
        """
        导出为YOLO格式

        Args:
            project: 项目数据
            output_dir: 输出目录
            format_type: 'detect' 或 'segment'
            split_ratio: (train, val, test) 比例
            smooth_level: 多边形平滑级别 'none', 'low', 'medium', 'high', 'ultra'

        Returns:
            导出结果统计
        """
        self.current_smooth_level = smooth_level
        self.format_type = format_type  # 保存导出类型
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 创建目录结构
        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

        # 获取类别映射
        classes = project.get('classes', [])
        if not classes:
            # 从标注中提取类别
            classes = self._extract_classes(project)

        class_to_id = {cls: i for i, cls in enumerate(classes)}

        # 分割数据集
        images = [img for img in project.get('images', []) if img.get('annotated', False)]
        train_end = int(len(images) * split_ratio[0])
        val_end = train_end + int(len(images) * split_ratio[1])

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        stats = {'train': 0, 'val': 0, 'test': 0, 'total_annotations': 0}

        for split_name, split_images in splits.items():
            for img_info in split_images:
                result = self._export_image(
                    img_info, output_path, split_name,
                    class_to_id
                )
                if result:
                    stats[split_name] += 1
                    stats['total_annotations'] += result

        # 生成data.yaml
        self._generate_yaml(output_path, classes, project.get('name', 'dataset'))

        stats['classes'] = classes
        stats['output_dir'] = str(output_path)

        return stats

    def _extract_classes(self, project: dict) -> list:
        """从标注中提取类别"""
        classes = set()
        for img in project.get('images', []):
            for ann in img.get('annotations', []):
                class_name = ann.get('class_name') or ann.get('label', 'object')
                classes.add(class_name)
        return sorted(list(classes))

    def _export_image(self, img_info: dict, output_path: Path,
                      split: str, class_to_id: dict) -> int:
        """导出单张图片"""
        src_path = img_info.get('path')
        if not src_path or not os.path.exists(src_path):
            return 0

        filename = img_info.get('filename')
        name_without_ext = Path(filename).stem

        # 复制图片
        dst_image = output_path / 'images' / split / filename
        shutil.copy2(src_path, dst_image)

        # 获取图片尺寸
        with Image.open(src_path) as img:
            # 处理 EXIF 旋转信息
            img = ImageOps.exif_transpose(img)
            img_width, img_height = img.size

        # 生成标签文件
        annotations = img_info.get('annotations', [])
        if not annotations:
            return 0

        label_file = output_path / 'labels' / split / f"{name_without_ext}.txt"
        lines = []

        for ann in annotations:
            class_name = ann.get('class_name') or ann.get('label', 'object')
            class_id = class_to_id.get(class_name, 0)

            if self.format_type == 'segment' and ann.get('polygon'):
                # 分割格式: class_id x1 y1 x2 y2 ... xn yn (归一化)
                polygon = ann['polygon']
                if len(polygon) >= 3:
                    # 应用平滑处理
                    smoothed_polygon = self.smooth_polygon(polygon, self.current_smooth_level)
                    coords = []
                    for point in smoothed_polygon:
                        x_norm = point[0] / img_width
                        y_norm = point[1] / img_height
                        coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                    line = f"{class_id} " + " ".join(coords)
                    lines.append(line)
            else:
                # 检测格式: class_id x_center y_center width height (归一化)
                bbox = ann.get('bbox', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    lines.append(line)

        with open(label_file, 'w') as f:
            f.write('\n'.join(lines))

        return len(lines)

    def _generate_yaml(self, output_path: Path, classes: list, dataset_name: str):
        """生成YOLO data.yaml配置文件"""
        data = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {i: name for i, name in enumerate(classes)},
            'nc': len(classes)
        }

        yaml_path = output_path / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        # 同时生成classes.txt
        classes_path = output_path / 'classes.txt'
        with open(classes_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(classes))
