"""
COCO格式导出器
支持标准COCO实例分割格式
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image


class COCOExporter:
    """COCO格式导出器"""

    def export(self, project: dict, output_dir: str,
               split_ratio: tuple = (0.8, 0.1, 0.1)) -> dict:
        """
        导出为COCO格式

        Args:
            project: 项目数据
            output_dir: 输出目录
            split_ratio: (train, val, test) 比例

        Returns:
            导出结果统计
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 创建目录结构
        for split in ['train', 'val', 'test']:
            (output_path / split).mkdir(exist_ok=True)

        annotations_dir = output_path / 'annotations'
        annotations_dir.mkdir(exist_ok=True)

        # 获取类别
        classes = project.get('classes', [])
        if not classes:
            classes = self._extract_classes(project)

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
            coco_data = self._create_coco_structure(project, classes)
            ann_count = self._export_split(
                split_images, output_path, split_name, coco_data, classes
            )

            # 保存COCO JSON
            json_path = annotations_dir / f'instances_{split_name}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, ensure_ascii=False, indent=2)

            stats[split_name] = len(split_images)
            stats['total_annotations'] += ann_count

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

    def _create_coco_structure(self, project: dict, classes: list) -> dict:
        """创建COCO基础结构"""
        return {
            'info': {
                'description': project.get('name', 'SAM3 Annotation Dataset'),
                'url': '',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'SAM3 Annotation Tool',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [{
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }],
            'categories': [
                {'id': i + 1, 'name': name, 'supercategory': 'object'}
                for i, name in enumerate(classes)
            ],
            'images': [],
            'annotations': []
        }

    def _export_split(self, images: list, output_path: Path,
                      split: str, coco_data: dict, classes: list) -> int:
        """导出单个数据集分割"""
        class_to_id = {cls: i + 1 for i, cls in enumerate(classes)}
        annotation_id = 1
        total_annotations = 0

        for img_idx, img_info in enumerate(images):
            src_path = img_info.get('path')
            if not src_path or not os.path.exists(src_path):
                continue

            filename = img_info.get('filename')
            image_id = img_idx + 1

            # 复制图片
            dst_path = output_path / split / filename
            shutil.copy2(src_path, dst_path)

            # 获取图片信息
            with Image.open(src_path) as img:
                img_width, img_height = img.size

            # 添加图片信息
            coco_data['images'].append({
                'id': image_id,
                'file_name': filename,
                'width': img_width,
                'height': img_height,
                'license': 1
            })

            # 添加标注
            for ann in img_info.get('annotations', []):
                class_name = ann.get('class_name') or ann.get('label', 'object')
                category_id = class_to_id.get(class_name, 1)

                coco_ann = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'iscrowd': 0
                }

                # 处理分割
                polygon = ann.get('polygon', [])
                if polygon and len(polygon) >= 3:
                    # 转换为COCO格式 [x1, y1, x2, y2, ...]
                    segmentation = []
                    for point in polygon:
                        segmentation.extend([float(point[0]), float(point[1])])
                    coco_ann['segmentation'] = [segmentation]

                    # 计算bbox
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    coco_ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                    coco_ann['area'] = (x_max - x_min) * (y_max - y_min)
                elif ann.get('bbox'):
                    # 只有bbox
                    bbox = ann['bbox']
                    x1, y1, x2, y2 = bbox[:4]
                    coco_ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    coco_ann['area'] = (x2 - x1) * (y2 - y1)
                    coco_ann['segmentation'] = []
                else:
                    continue

                coco_data['annotations'].append(coco_ann)
                annotation_id += 1
                total_annotations += 1

        return total_annotations
