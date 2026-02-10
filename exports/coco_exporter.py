"""
COCO格式导出器
支持标准COCO实例分割格式
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
import cv2


class COCOExporter:
    """COCO格式导出器"""

    def __init__(self):
        # 平滑参数配置（在mask级别进行形态学平滑）
        self.smooth_params = {
            'none': {'kernel_size': 0, 'simplify_epsilon': 0.002},
            'low': {'kernel_size': 3, 'simplify_epsilon': 0.0015},
            'medium': {'kernel_size': 5, 'simplify_epsilon': 0.001},
            'high': {'kernel_size': 7, 'simplify_epsilon': 0.0008},
            'ultra': {'kernel_size': 11, 'simplify_epsilon': 0.0005},
        }
        self.export_type = 'segment'  # 默认导出类型

    def _smooth_polygon_via_mask(self, polygon: list, kernel_size: int) -> np.ndarray:
        """通过渲染到mask再提取的方式平滑多边形"""
        if kernel_size == 0:
            return np.array(polygon, dtype=np.float64)

        points = np.array(polygon, dtype=np.float64)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        margin = kernel_size * 2 + 10
        x_min = max(0, int(x_min) - margin)
        y_min = max(0, int(y_min) - margin)
        x_max = int(x_max) + margin
        y_max = int(y_max) + margin

        width = x_max - x_min
        height = y_max - y_min

        mask = np.zeros((height, width), dtype=np.uint8)
        shifted_points = points - np.array([x_min, y_min])
        cv2.fillPoly(mask, [shifted_points.astype(np.int32)], 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

        if kernel_size >= 5:
            blur_size = kernel_size | 1
            smoothed = cv2.GaussianBlur(smoothed, (blur_size, blur_size), 0)
            _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if not contours:
            return points

        largest = max(contours, key=cv2.contourArea)
        new_points = largest.reshape(-1, 2).astype(np.float64)
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
        """对多边形进行平滑处理（通过mask级别的形态学操作）"""
        if not polygon or len(polygon) < 3:
            return polygon

        params = self.smooth_params.get(smooth_level, self.smooth_params['medium'])
        smoothed = self._smooth_polygon_via_mask(polygon, params['kernel_size'])
        result = self._adaptive_simplify(smoothed, params['simplify_epsilon'])

        return result.tolist()

    def export(self, project: dict, output_dir: str,
               export_type: str = 'segment',
               split_ratio: tuple = (0.8, 0.1, 0.1),
               smooth_level: str = 'medium') -> dict:
        """
        导出为COCO格式

        Args:
            project: 项目数据
            output_dir: 输出目录
            export_type: 'detect' 仅边界框 或 'segment' 实例分割
            split_ratio: (train, val, test) 比例
            smooth_level: 多边形平滑级别 'none', 'low', 'medium', 'high', 'ultra'

        Returns:
            导出结果统计
        """
        self.current_smooth_level = smooth_level
        self.export_type = export_type
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
                split_images, output_path, split_name, coco_data, classes, export_type
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
                      split: str, coco_data: dict, classes: list,
                      export_type: str = 'segment') -> int:
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
                # 处理 EXIF 旋转信息
                img = ImageOps.exif_transpose(img)
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

                # 根据导出类型处理分割或检测
                if export_type == 'segment' and ann.get('polygon'):
                    # 分割格式：包含完整多边形轮廓
                    polygon = ann['polygon']
                    if len(polygon) >= 3:
                        # 应用平滑处理
                        smoothed_polygon = self.smooth_polygon(polygon, self.current_smooth_level)
                        # 转换为COCO格式 [x1, y1, x2, y2, ...]
                        segmentation = []
                        for point in smoothed_polygon:
                            segmentation.extend([float(point[0]), float(point[1])])
                        coco_ann['segmentation'] = [segmentation]

                        # 计算bbox
                        xs = [p[0] for p in polygon]
                        ys = [p[1] for p in polygon]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        coco_ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                        coco_ann['area'] = (x_max - x_min) * (y_max - y_min)
                    else:
                        continue
                else:
                    # 检测格式：仅包含边界框
                    bbox = ann.get('bbox')
                    if bbox and len(bbox) >= 4:
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
