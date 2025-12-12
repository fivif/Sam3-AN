"""
YOLO格式导出器
支持YOLOv5/v8检测和分割格式
"""
import os
import shutil
from pathlib import Path
from PIL import Image
import yaml


class YOLOExporter:
    """YOLO格式导出器"""

    def export(self, project: dict, output_dir: str,
               format_type: str = 'segment',
               split_ratio: tuple = (0.8, 0.1, 0.1)) -> dict:
        """
        导出为YOLO格式

        Args:
            project: 项目数据
            output_dir: 输出目录
            format_type: 'detect' 或 'segment'
            split_ratio: (train, val, test) 比例

        Returns:
            导出结果统计
        """
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
                    class_to_id, format_type
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
                      split: str, class_to_id: dict, format_type: str) -> int:
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

            if format_type == 'segment' and ann.get('polygon'):
                # 分割格式: class_id x1 y1 x2 y2 ... xn yn (归一化)
                polygon = ann['polygon']
                if len(polygon) >= 3:
                    coords = []
                    for point in polygon:
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
