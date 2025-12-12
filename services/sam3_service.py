"""
SAM3模型服务封装 - 修正版
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import uuid
import traceback

# 使用本地 SAM_src 目录
sam3_src = Path(__file__).parent.parent / "SAM_src"
sys.path.insert(0, str(sam3_src))


class SAM3Service:
    """SAM3模型服务"""

    def __init__(self):
        self.image_model = None
        self.image_processor = None
        self.video_predictor = None
        self.current_image_path = None
        self.inference_state = None
        self.video_sessions = {}
        self._image_size = None

    def _init_image_model(self):
        """初始化图像分割模型"""
        if self.image_model is not None:
            return

        print("正在加载SAM3图像模型...")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        bpe_path = sam3_src / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        self.image_model = build_sam3_image_model(bpe_path=str(bpe_path))
        self.image_processor = Sam3Processor(self.image_model)

        print("SAM3图像模型加载完成")

    def _load_image(self, image_path: str):
        """加载图像"""
        if self.current_image_path != image_path or self.inference_state is None:
            print(f"[DEBUG] 加载图像: {image_path}")
            image = Image.open(image_path).convert('RGB')
            self._image_size = image.size
            self.inference_state = self.image_processor.set_image(image)
            self.current_image_path = image_path
            print(f"[DEBUG] 图像尺寸: {self._image_size}")

    def _mask_to_polygon(self, mask: np.ndarray) -> list:
        """将mask转换为多边形"""
        import cv2
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        return approx.reshape(-1, 2).tolist()

    def segment_by_text(self, image_path: str, prompt: str, confidence: float = 0.5) -> list:
        """文本提示分割"""
        try:
            self._init_image_model()
            self._load_image(image_path)

            print(f"[DEBUG] 文本分割: prompt='{prompt}', confidence={confidence}")

            # 设置置信度
            self.image_processor.confidence_threshold = confidence

            # 执行文本分割 - 直接使用 set_text_prompt，它会返回结果
            output = self.image_processor.set_text_prompt(
                state=self.inference_state,
                prompt=prompt
            )

            print(f"[DEBUG] 输出keys: {list(output.keys()) if output else 'None'}")

            return self._extract_results(output, prompt)

        except Exception as e:
            print(f"[ERROR] segment_by_text: {e}")
            traceback.print_exc()
            return []

    def segment_by_points(self, image_path: str, points: list) -> list:
        """点击分割 - 支持正负样本点

        正样本点：指示要分割的对象位置
        负样本点：指示不想要的区域（用于排除）

        策略：
        1. 正样本点转换为小框，用于触发分割
        2. 负样本点转换为小框，用于过滤结果
        """
        try:
            self._init_image_model()
            self._load_image(image_path)

            # 分离正负样本点
            positive_points = []
            negative_points = []

            print(f"[DEBUG] 点击分割: 共 {len(points)} 个点")
            for i, p in enumerate(points):
                x, y, label = p
                is_positive = label == 1
                label_str = "正样本(+)" if is_positive else "负样本(-)"
                print(f"[DEBUG]   点{i}: ({x:.1f}, {y:.1f}) {label_str}")

                if is_positive:
                    positive_points.append([x, y])
                else:
                    negative_points.append([x, y])

            print(f"[DEBUG] 正样本点: {len(positive_points)}, 负样本点: {len(negative_points)}")

            # 如果没有正样本点，无法分割
            if not positive_points:
                print("[DEBUG] 没有正样本点，无法分割")
                return []

            width, height = self._image_size

            # 将点转换为框
            boxes = []

            # 正样本点 -> 正样本框
            for x, y in positive_points:
                box_size = 15  # 稍小的框，更精确
                x1 = max(0, x - box_size)
                y1 = max(0, y - box_size)
                x2 = min(width, x + box_size)
                y2 = min(height, y + box_size)
                boxes.append([x1, y1, x2, y2, 1])  # label=1 正样本

            # 负样本点 -> 负样本框（用于过滤）
            for x, y in negative_points:
                box_size = 30  # 稍大的框，确保覆盖不想要的区域
                x1 = max(0, x - box_size)
                y1 = max(0, y - box_size)
                x2 = min(width, x + box_size)
                y2 = min(height, y + box_size)
                boxes.append([x1, y1, x2, y2, 0])  # label=0 负样本

            return self.segment_by_boxes(image_path, boxes)

        except Exception as e:
            print(f"[ERROR] segment_by_points: {e}")
            traceback.print_exc()
            return []

    def _mask_in_negative_region(self, mask: np.ndarray, negative_boxes: list, threshold: float = 0.5) -> bool:
        """检查 mask 是否主要位于负样本区域内

        Args:
            mask: 二值 mask (H, W)
            negative_boxes: 负样本框列表 [[x1, y1, x2, y2], ...]
            threshold: mask 在负样本区域内的比例阈值

        Returns:
            True 如果 mask 主要在负样本区域内，应该被排除
        """
        if not negative_boxes:
            return False

        mask_area = mask.sum()
        if mask_area == 0:
            return True  # 空 mask 直接排除

        # 创建负样本区域的 mask
        h, w = mask.shape
        negative_region = np.zeros((h, w), dtype=np.uint8)

        for box in negative_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            negative_region[y1:y2, x1:x2] = 1

        # 计算 mask 与负样本区域的重叠
        overlap = (mask > 0) & (negative_region > 0)
        overlap_area = overlap.sum()

        # 计算重叠比例（相对于 mask 面积）
        overlap_ratio = overlap_area / mask_area

        return overlap_ratio > threshold

    def _boxes_overlap(self, box1: list, box2: list, threshold: float = 0.3) -> bool:
        """检查两个框是否重叠
        box 格式: [x1, y1, x2, y2]
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return False

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        min_area = min(box1_area, box2_area)
        if min_area <= 0:
            return False

        overlap_ratio = inter_area / min_area
        return overlap_ratio > threshold

    def segment_by_boxes(self, image_path: str, boxes: list) -> list:
        """框选分割 - 支持正负样本

        正样本框：用于指示要分割的区域
        负样本框：用于排除不想要的分割结果

        策略：
        1. 同时将正样本和负样本框传递给 SAM3（利用原生支持）
        2. 使用 mask 级别的后处理过滤（更精确）
        """
        try:
            self._init_image_model()
            self._load_image(image_path)

            # 分离正样本框和负样本框（原始像素坐标）
            positive_boxes_px = []
            negative_boxes_px = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box[:4]
                label = box[4] if len(box) > 4 else 1
                is_positive = bool(label)

                label_str = "正样本(+)" if is_positive else "负样本(-)"
                print(f"[DEBUG] 框{i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] {label_str}")

                if is_positive:
                    positive_boxes_px.append([x1, y1, x2, y2])
                else:
                    negative_boxes_px.append([x1, y1, x2, y2])

            print(f"[DEBUG] 正样本框: {len(positive_boxes_px)}, 负样本框: {len(negative_boxes_px)}")

            # 如果没有正样本框，无法分割
            if not positive_boxes_px:
                print("[DEBUG] 没有正样本框，无法分割")
                return []

            # 重置 geometric_prompt
            if "geometric_prompt" in self.inference_state:
                del self.inference_state["geometric_prompt"]
            print("[DEBUG] 已重置 geometric_prompt")

            width, height = self._image_size
            output = None

            # 先添加所有正样本框
            for i, box in enumerate(positive_boxes_px):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2 / width
                cy = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                norm_box = [cx, cy, w, h]
                print(f"[DEBUG] 添加正样本框{i}: {norm_box}")
                output = self.image_processor.add_geometric_prompt(
                    norm_box, True, self.inference_state
                )

            # 再添加所有负样本框（SAM3 原生支持）
            for i, box in enumerate(negative_boxes_px):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2 / width
                cy = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                norm_box = [cx, cy, w, h]
                print(f"[DEBUG] 添加负样本框{i}: {norm_box}")
                output = self.image_processor.add_geometric_prompt(
                    norm_box, False, self.inference_state
                )

            if output is None:
                return []

            print(f"[DEBUG] 输出keys: {list(output.keys()) if output else 'None'}")

            # 提取结果（带 mask 数据用于后处理）
            results = self._extract_results_with_mask(output, "box_prompt", negative_boxes_px)

            return results

        except Exception as e:
            print(f"[ERROR] segment_by_boxes: {e}")
            traceback.print_exc()
            return []

    def _extract_results_with_mask(self, output: dict, label: str, negative_boxes: list) -> list:
        """从输出提取结果，并使用 mask 级别过滤负样本区域"""
        results = []

        if output is None:
            print("[DEBUG] output is None")
            return results

        masks = output.get('masks', [])
        boxes = output.get('boxes', [])
        scores = output.get('scores', [])

        print(f"[DEBUG] 原始结果: masks={len(masks)}, boxes={len(boxes)}, scores={len(scores)}")

        filtered_count = 0
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            try:
                mask_np = mask[0].cpu().numpy()
                box_np = box.cpu().numpy().tolist()

                # 使用 mask 级别的负样本过滤
                if negative_boxes and self._mask_in_negative_region(mask_np, negative_boxes, threshold=0.4):
                    print(f"[DEBUG] 结果{i}: score={float(score):.4f} - 被负样本区域过滤")
                    filtered_count += 1
                    continue

                print(f"[DEBUG] 结果{i}: score={float(score):.4f}, bbox={[f'{v:.1f}' for v in box_np]}")

                polygon = self._mask_to_polygon(mask_np)

                results.append({
                    'id': str(uuid.uuid4())[:8],
                    'label': label,
                    'score': float(score),
                    'bbox': box_np,
                    'polygon': polygon,
                    'area': float(mask_np.sum()),
                })
            except Exception as e:
                print(f"[ERROR] 提取结果{i}失败: {e}")

        if filtered_count > 0:
            print(f"[DEBUG] 负样本过滤: 排除了 {filtered_count} 个结果")

        return results

    def _extract_results(self, output: dict, label: str) -> list:
        """从输出提取结果"""
        results = []

        if output is None:
            print("[DEBUG] output is None")
            return results

        masks = output.get('masks', [])
        boxes = output.get('boxes', [])
        scores = output.get('scores', [])

        print(f"[DEBUG] 结果: masks={len(masks)}, boxes={len(boxes)}, scores={len(scores)}")

        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            try:
                mask_np = mask[0].cpu().numpy()
                box_np = box.cpu().numpy().tolist()

                print(f"[DEBUG] 结果{i}: score={float(score):.4f}, bbox={box_np}")

                polygon = self._mask_to_polygon(mask_np)

                results.append({
                    'id': str(uuid.uuid4())[:8],
                    'label': label,
                    'score': float(score),
                    'bbox': box_np,
                    'polygon': polygon,
                    'area': float(mask_np.sum()),
                })
            except Exception as e:
                print(f"[ERROR] 提取结果{i}失败: {e}")

        return results

    # ==================== 视频分割 ====================

    def _init_video_model(self):
        if self.video_predictor is not None:
            return

        print("正在加载SAM3视频模型...")
        from sam3.model_builder import build_sam3_video_predictor

        gpus = range(torch.cuda.device_count()) if torch.cuda.is_available() else []
        self.video_predictor = build_sam3_video_predictor(gpus_to_use=gpus)
        print("SAM3视频模型加载完成")

    def start_video_session(self, video_path: str) -> str:
        self._init_video_model()
        response = self.video_predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = response["session_id"]
        self.video_sessions[session_id] = {'video_path': video_path, 'outputs': {}}
        return session_id

    def add_video_prompt(self, session_id: str, frame_index: int,
                         prompt_type: str, prompt_data) -> dict:
        self._init_video_model()

        request = {
            'type': 'add_prompt',
            'session_id': session_id,
            'frame_index': frame_index,
        }

        if prompt_type == 'text':
            request['text'] = prompt_data
        elif prompt_type == 'points':
            points = torch.tensor(prompt_data['points'], dtype=torch.float32)
            labels = torch.tensor(prompt_data['labels'], dtype=torch.int32)
            request['points'] = points
            request['point_labels'] = labels
            if 'obj_id' in prompt_data:
                request['obj_id'] = prompt_data['obj_id']

        response = self.video_predictor.handle_request(request=request)
        return response.get('outputs', {})

    def propagate_video(self, session_id: str) -> dict:
        self._init_video_model()
        outputs = {}
        for response in self.video_predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            outputs[response["frame_index"]] = response["outputs"]
        return outputs

    def close_video_session(self, session_id: str):
        if self.video_predictor:
            self.video_predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
        self.video_sessions.pop(session_id, None)

    def shutdown(self):
        if self.video_predictor:
            self.video_predictor.shutdown()
        self.video_predictor = None
        self.image_model = None
        self.image_processor = None
