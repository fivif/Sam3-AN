import os
import sys
import subprocess
import threading
import webbrowser
from pathlib import Path

# 添加SAM3到路径 (使用本地 SAM_src 目录)
sam3_src = Path(__file__).parent / "SAM_src"
sys.path.insert(0, str(sam3_src))

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import json
import uuid
import requests
from datetime import datetime

from services.sam3_service import SAM3Service
from services.annotation_manager import AnnotationManager
from exports.yolo_exporter import YOLOExporter
from exports.coco_exporter import COCOExporter

app = Flask(__name__)
CORS(app)

# 全局配置
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# 全局服务实例
sam3_service = None
annotation_manager = AnnotationManager()


def get_sam3_service():
    """延迟加载SAM3服务"""
    global sam3_service
    if sam3_service is None:
        sam3_service = SAM3Service()
    return sam3_service


# ==================== 页面路由 ====================

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/video')
def video_page():
    """视频标注页面"""
    return render_template('video.html')


# ==================== 项目管理API ====================

@app.route('/api/project/create', methods=['POST'])
def create_project():
    """创建新项目"""
    data = request.json
    project_id = str(uuid.uuid4())[:8]
    project = {
        'id': project_id,
        'name': data.get('name', f'项目_{project_id}'),
        'image_dir': data.get('image_dir', ''),
        'output_dir': data.get('output_dir', ''),
        'export_format': data.get('export_format', 'yolo'),
        'classes': data.get('classes', []),
        'created_at': datetime.now().isoformat(),
        'images': [],
        'current_index': 0
    }
    annotation_manager.create_project(project)
    return jsonify({'success': True, 'project': project})


@app.route('/api/project/<project_id>', methods=['GET'])
def get_project(project_id):
    """获取项目信息"""
    project = annotation_manager.get_project(project_id)
    if project:
        return jsonify({'success': True, 'project': project})
    return jsonify({'success': False, 'error': '项目不存在'})


@app.route('/api/project/<project_id>/update', methods=['POST'])
def update_project(project_id):
    """更新项目信息"""
    data = request.json
    project = annotation_manager.get_project(project_id)

    if not project:
        return jsonify({'success': False, 'error': '项目不存在'})

    # 构建更新字段
    updates = {}
    if 'name' in data:
        updates['name'] = data['name']
    if 'image_dir' in data:
        updates['image_dir'] = data['image_dir']
    if 'output_dir' in data:
        updates['output_dir'] = data['output_dir']
    if 'classes' in data:
        updates['classes'] = data['classes']

    try:
        updated_project = annotation_manager.update_project(project_id, updates)
        return jsonify({'success': True, 'project': updated_project})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/project/<project_id>/delete', methods=['POST'])
def delete_project(project_id):
    """删除项目"""
    project = annotation_manager.get_project(project_id)

    if not project:
        return jsonify({'success': False, 'error': '项目不存在'})

    try:
        annotation_manager.delete_project(project_id)
        return jsonify({'success': True, 'message': '项目已删除'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/project/<project_id>/load_images', methods=['POST'])
def load_project_images(project_id):
    """加载项目图片目录"""
    data = request.json
    image_dir = data.get('image_dir', '')

    if not os.path.isdir(image_dir):
        return jsonify({'success': False, 'error': '目录不存在'})

    # 支持的图片格式
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    images = []

    for f in sorted(os.listdir(image_dir)):
        if Path(f).suffix.lower() in extensions:
            images.append({
                'filename': f,
                'path': os.path.join(image_dir, f),
                'annotated': False,
                'annotations': []
            })

    annotation_manager.update_project_images(project_id, images, image_dir)
    return jsonify({
        'success': True,
        'count': len(images),
        'images': images
    })


@app.route('/api/project/list', methods=['GET'])
def list_projects():
    """列出所有项目"""
    projects = annotation_manager.list_projects()
    return jsonify({'success': True, 'projects': projects})


# ==================== 图片服务API ====================

@app.route('/api/image/serve')
def serve_image():
    """提供图片文件"""
    path = request.args.get('path', '')
    if not path:
        return jsonify({'error': '缺少文件路径'}), 400

    abs_path = Path(path).resolve()

    # 仅允许读取已知项目图片目录或上传目录，避免任意文件读取
    allowed_dirs = {app.config['UPLOAD_FOLDER'].resolve()}
    for project in annotation_manager.list_projects():
        image_dir = project.get('image_dir')
        if image_dir:
            allowed_dirs.add(Path(image_dir).resolve())

    is_allowed = any(abs_path == d or d in abs_path.parents for d in allowed_dirs)
    if not (abs_path.is_file() and is_allowed):
        return jsonify({'error': '文件不存在或无权访问'}), 404

    return send_file(abs_path)


# ==================== SAM3分割API ====================

@app.route('/api/segment/text', methods=['POST'])
def segment_by_text():
    """文本提示分割"""
    data = request.json
    image_path = data.get('image_path')
    prompt = data.get('prompt', '')
    confidence = data.get('confidence', 0.5)

    if not image_path or not prompt:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    try:
        service = get_sam3_service()
        results = service.segment_by_text(image_path, prompt, confidence)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/segment/point', methods=['POST'])
def segment_by_point():
    """点击分割"""
    data = request.json
    image_path = data.get('image_path')
    points = data.get('points', [])  # [[x, y, label], ...]

    if not image_path or not points:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    try:
        service = get_sam3_service()
        results = service.segment_by_points(image_path, points)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/segment/box', methods=['POST'])
def segment_by_box():
    """框选分割"""
    data = request.json
    image_path = data.get('image_path')
    boxes = data.get('boxes', [])  # [[x1, y1, x2, y2, label], ...]

    if not image_path or not boxes:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    try:
        service = get_sam3_service()
        results = service.segment_by_boxes(image_path, boxes)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/segment/batch', methods=['POST'])
def batch_segment():
    """批量分割"""
    data = request.json
    project_id = data.get('project_id')
    prompt = data.get('prompt', '')
    class_name = data.get('class_name', prompt)  # 使用传入的类名，默认为prompt
    start_index = data.get('start_index', 0)
    end_index = data.get('end_index', -1)
    skip_annotated = data.get('skip_annotated', True)
    confidence = data.get('confidence', 0.5)

    try:
        service = get_sam3_service()
        project = annotation_manager.get_project(project_id)

        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        images = project['images']
        if end_index == -1:
            end_index = len(images)

        processed = 0
        failed = 0
        total_detections = 0
        results = []

        for i in range(start_index, min(end_index, len(images))):
            img_info = images[i]

            # 跳过已标注的图片
            if skip_annotated and img_info.get('annotated', False):
                continue

            try:
                seg_results = service.segment_by_text(
                    img_info['path'], prompt, confidence
                )

                if seg_results:
                    # 使用传入的类名，而不是 prompt
                    for r in seg_results:
                        r['class_name'] = class_name

                    annotation_manager.add_annotations(
                        project_id, i, seg_results, class_name
                    )
                    processed += 1
                    total_detections += len(seg_results)
                    results.append({
                        'index': i,
                        'filename': img_info['filename'],
                        'count': len(seg_results)
                    })
                else:
                    # 没有检测到对象，也算处理过
                    processed += 1

            except Exception as e:
                print(f"[ERROR] 批量分割图片 {img_info['filename']} 失败: {e}")
                failed += 1
                continue

        return jsonify({
            'success': True,
            'processed': processed,
            'failed': failed,
            'total_detections': total_detections,
            'results': results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ==================== 标注管理API ====================

@app.route('/api/annotation/save', methods=['POST'])
def save_annotation():
    """保存标注"""
    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index')
    annotations = data.get('annotations', [])

    try:
        annotation_manager.save_annotations(project_id, image_index, annotations)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/annotation/get', methods=['GET'])
def get_annotation():
    """获取标注"""
    project_id = request.args.get('project_id')
    image_index = int(request.args.get('image_index', 0))

    annotations = annotation_manager.get_annotations(project_id, image_index)
    return jsonify({'success': True, 'annotations': annotations})


@app.route('/api/annotation/update', methods=['POST'])
def update_annotation():
    """更新单个标注（手动调整）"""
    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index')
    annotation_id = data.get('annotation_id')
    updates = data.get('updates', {})

    try:
        annotation_manager.update_annotation(
            project_id, image_index, annotation_id, updates
        )
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/annotation/delete', methods=['POST'])
def delete_annotation():
    """删除标注"""
    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index')
    annotation_id = data.get('annotation_id')

    try:
        annotation_manager.delete_annotation(project_id, image_index, annotation_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 类别管理API ====================

@app.route('/api/classes/update', methods=['POST'])
def update_classes():
    """更新类别列表"""
    data = request.json
    project_id = data.get('project_id')
    classes = data.get('classes', [])

    try:
        annotation_manager.update_classes(project_id, classes)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 导出API ====================

@app.route('/api/export/yolo', methods=['POST'])
def export_yolo():
    """导出YOLO格式"""
    data = request.json
    project_id = data.get('project_id')
    output_dir = data.get('output_dir', '')
    smooth_level = data.get('smooth_level', 'medium')
    export_type = data.get('export_type', 'segment')  # 'detect' 或 'segment'

    try:
        project = annotation_manager.get_project(project_id)
        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        exporter = YOLOExporter()
        result = exporter.export(
            project, output_dir, 
            format_type=export_type,
            smooth_level=smooth_level
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/export/coco', methods=['POST'])
def export_coco():
    """导出COCO格式"""
    data = request.json
    project_id = data.get('project_id')
    output_dir = data.get('output_dir', '')
    smooth_level = data.get('smooth_level', 'medium')
    export_type = data.get('export_type', 'segment')  # 'detect' 或 'segment'

    try:
        project = annotation_manager.get_project(project_id)
        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        exporter = COCOExporter()
        result = exporter.export(
            project, output_dir,
            export_type=export_type,
            smooth_level=smooth_level
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 导出预览API ====================

@app.route('/api/export/preview', methods=['POST'])
def export_preview():
    """生成导出预览图片，显示平滑后的分割覆盖效果"""
    import cv2
    import numpy as np
    import base64
    from io import BytesIO

    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index', 0)
    smooth_level = data.get('smooth_level', 'medium')
    show_polygon = data.get('show_polygon', True)
    show_fill = data.get('show_fill', True)
    opacity = data.get('opacity', 0.4)

    try:
        project = annotation_manager.get_project(project_id)
        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        images = project.get('images', [])
        if image_index >= len(images):
            return jsonify({'success': False, 'error': '图片索引超出范围'})

        img_info = images[image_index]
        image_path = img_info.get('path')

        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': '图片文件不存在'})

        # 读取原始图片
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({'success': False, 'error': '无法读取图片'})

        overlay = img.copy()
        annotations = img_info.get('annotations', [])

        # 使用导出器的平滑方法
        exporter = YOLOExporter()

        # 颜色列表（BGR格式）
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
            (128, 0, 255),  # 紫色
            (255, 128, 0),  # 橙色
        ]

        for i, ann in enumerate(annotations):
            polygon = ann.get('polygon', [])
            if not polygon or len(polygon) < 3:
                continue

            # 应用平滑处理
            smoothed_polygon = exporter.smooth_polygon(polygon, smooth_level)

            # 转换为numpy数组
            pts = np.array(smoothed_polygon, dtype=np.int32)
            color = colors[i % len(colors)]

            # 绘制填充
            if show_fill:
                cv2.fillPoly(overlay, [pts], color)

            # 绘制轮廓线
            if show_polygon:
                cv2.polylines(img, [pts], True, color, 2)

        # 混合原图和覆盖层
        if show_fill:
            img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)

        # 添加标注信息文字
        for i, ann in enumerate(annotations):
            polygon = ann.get('polygon', [])
            if not polygon:
                continue

            smoothed_polygon = exporter.smooth_polygon(polygon, smooth_level)
            if smoothed_polygon:
                # 计算中心点
                pts = np.array(smoothed_polygon)
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())

                label = ann.get('class_name') or ann.get('label', '')
                color = colors[i % len(colors)]

                # 绘制标签背景
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (cx - 2, cy - text_h - 4), (cx + text_w + 2, cy + 2), color, -1)
                cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 转换为base64
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 统计信息
        stats = {
            'total_annotations': len(annotations),
            'smooth_level': smooth_level,
            'image_size': [img.shape[1], img.shape[0]],
            'filename': img_info.get('filename', '')
        }

        return jsonify({
            'success': True,
            'preview': f'data:image/jpeg;base64,{img_base64}',
            'stats': stats
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/export/preview_compare', methods=['POST'])
def export_preview_compare():
    """生成多个平滑级别的对比预览"""
    import cv2
    import numpy as np
    import base64

    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index', 0)
    annotation_index = data.get('annotation_index', 0)  # 指定要预览的标注索引

    try:
        project = annotation_manager.get_project(project_id)
        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        images = project.get('images', [])
        if image_index >= len(images):
            return jsonify({'success': False, 'error': '图片索引超出范围'})

        img_info = images[image_index]
        image_path = img_info.get('path')
        annotations = img_info.get('annotations', [])

        if annotation_index >= len(annotations):
            return jsonify({'success': False, 'error': '标注索引超出范围'})

        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': '图片文件不存在'})

        # 读取原始图片
        original_img = cv2.imread(image_path)
        if original_img is None:
            return jsonify({'success': False, 'error': '无法读取图片'})

        exporter = YOLOExporter()
        polygon = annotations[annotation_index].get('polygon', [])

        if not polygon or len(polygon) < 3:
            return jsonify({'success': False, 'error': '标注没有有效的多边形数据'})

        # 生成不同平滑级别的预览
        levels = ['none', 'low', 'medium', 'high', 'ultra']
        previews = {}

        for level in levels:
            img = original_img.copy()
            smoothed_polygon = exporter.smooth_polygon(polygon, level)
            pts = np.array(smoothed_polygon, dtype=np.int32)

            # 绘制填充和轮廓
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

            # 添加级别标签
            cv2.putText(img, f'{level} ({len(smoothed_polygon)} pts)',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 转换为base64
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            previews[level] = f'data:image/jpeg;base64,{base64.b64encode(buffer).decode("utf-8")}'

        return jsonify({
            'success': True,
            'previews': previews,
            'original_points': len(polygon),
            'annotation_label': annotations[annotation_index].get('class_name', '')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ==================== 视频分割API ====================

@app.route('/api/video/start_session', methods=['POST'])
def video_start_session():
    """开始视频分割会话"""
    data = request.json
    video_path = data.get('video_path')

    try:
        service = get_sam3_service()
        session_id = service.start_video_session(video_path)
        return jsonify({'success': True, 'session_id': session_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/video/add_prompt', methods=['POST'])
def video_add_prompt():
    """添加视频分割提示"""
    data = request.json
    session_id = data.get('session_id')
    frame_index = data.get('frame_index', 0)
    prompt_type = data.get('prompt_type', 'text')
    prompt_data = data.get('prompt_data')

    try:
        service = get_sam3_service()
        results = service.add_video_prompt(
            session_id, frame_index, prompt_type, prompt_data
        )
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/video/propagate', methods=['POST'])
def video_propagate():
    """传播视频分割"""
    data = request.json
    session_id = data.get('session_id')

    try:
        service = get_sam3_service()
        results = service.propagate_video(session_id)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/video/close_session', methods=['POST'])
def video_close_session():
    """关闭视频会话"""
    data = request.json
    session_id = data.get('session_id')

    try:
        service = get_sam3_service()
        service.close_video_session(session_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== AI翻译API ====================

@app.route('/api/ai/translate', methods=['POST'])
def ai_translate():
    """
    使用OpenAI格式的API将中文翻译成简短的英文
    支持第三方API（如DeepSeek、通义千问、Moonshot等）
    """
    data = request.json
    text = data.get('text', '').strip()
    api_url = data.get('api_url', '').strip()
    api_key = data.get('api_key', '').strip()
    model = data.get('model', 'gpt-3.5-turbo').strip()

    if not text:
        return jsonify({'success': False, 'error': '文本为空'})

    if not api_url or not api_key:
        return jsonify({'success': False, 'error': 'API未配置'})

    # 确保API URL以/v1/chat/completions结尾
    if not api_url.endswith('/v1/chat/completions'):
        api_url = api_url.rstrip('/')
        if not api_url.endswith('/v1'):
            api_url += '/v1'
        api_url += '/chat/completions'

    try:
        # 构建翻译提示
        system_prompt = """You are a translation assistant for image segmentation tasks.
Translate the user's Chinese text into simple, concise English words or short phrases that can be used as object detection prompts.
Rules:
1. Output ONLY the English translation, nothing else
2. Keep it as short as possible (1-3 words preferred)
3. Use common object names (e.g., "apple", "car", "person", "red ball")
4. If multiple objects, separate with comma
5. No explanations, no quotes, just the words"""

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text}
            ],
            'max_tokens': 100,
            'temperature': 0.3
        }

        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        print(f"[AI翻译] 正在连接: {api_url}")

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30,
            verify=False  # 跳过SSL证书验证，解决WSL环境下的证书问题
        )

        print(f"[AI翻译] 响应状态码: {response.status_code}")

        if response.status_code != 200:
            error_msg = f'API请求失败: {response.status_code}'
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg = error_data['error'].get('message', error_msg)
            except:
                pass
            return jsonify({'success': False, 'error': error_msg})

        result = response.json()
        translated = result['choices'][0]['message']['content'].strip()

        print(f"[AI翻译] {text} -> {translated}")

        return jsonify({
            'success': True,
            'original': text,
            'translated': translated
        })

    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'API请求超时 (30秒)'})
    except requests.exceptions.SSLError as e:
        print(f"[AI翻译] SSL错误: {e}")
        return jsonify({'success': False, 'error': f'SSL证书错误'})
    except requests.exceptions.ConnectionError as e:
        print(f"[AI翻译] 连接错误: {e}")
        return jsonify({'success': False, 'error': '无法连接到API服务器'})
    except Exception as e:
        print(f"[AI翻译错误] {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/ai/test', methods=['POST'])
def ai_test():
    """测试AI API配置是否有效"""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    data = request.json
    api_url = data.get('api_url', '').strip()
    api_key = data.get('api_key', '').strip()
    model = data.get('model', 'gpt-3.5-turbo').strip()

    if not api_url or not api_key:
        return jsonify({'success': False, 'error': 'API地址和密钥不能为空'})

    # 确保API URL格式正确
    if not api_url.endswith('/v1/chat/completions'):
        api_url = api_url.rstrip('/')
        if not api_url.endswith('/v1'):
            api_url += '/v1'
        api_url += '/chat/completions'

    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        payload = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': 'Hello'}
            ],
            'max_tokens': 10
        }

        print(f"[AI测试] 正在连接: {api_url}")

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30,
            verify=False  # 跳过SSL证书验证，解决WSL环境下的证书问题
        )

        print(f"[AI测试] 响应状态码: {response.status_code}")

        if response.status_code == 200:
            return jsonify({'success': True, 'message': 'API连接成功'})
        else:
            error_msg = f'状态码: {response.status_code}'
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg = error_data['error'].get('message', error_msg)
            except:
                pass
            return jsonify({'success': False, 'error': error_msg})

    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': '连接超时 (30秒)'})
    except requests.exceptions.SSLError as e:
        print(f"[AI测试] SSL错误: {e}")
        return jsonify({'success': False, 'error': f'SSL证书错误: {str(e)[:100]}'})
    except requests.exceptions.ConnectionError as e:
        print(f"[AI测试] 连接错误: {e}")
        return jsonify({'success': False, 'error': f'无法连接到API服务器，请检查网络或API地址是否正确'})
    except Exception as e:
        print(f"[AI测试] 未知错误: {e}")
        return jsonify({'success': False, 'error': str(e)})


def wait_for_server(url, timeout=30):
    """等待服务器启动就绪"""
    import time
    import urllib.request
    import urllib.error

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(0.3)
    return False


def open_browser(url):
    """等待服务就绪后打开浏览器（独立窗口模式）"""
    print("[INFO] 等待服务启动...")

    # 等待服务就绪
    if not wait_for_server(url):
        print("[ERROR] 服务启动超时，请手动打开浏览器访问:", url)
        return

    print("[INFO] 服务已就绪，正在打开浏览器...")

    # 尝试不同的浏览器路径
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]

    browser_path = None
    for path in chrome_paths:
        if os.path.exists(path):
            browser_path = path
            break

    if browser_path:
        # 使用 --app 模式打开，类似独立应用（无地址栏）
        subprocess.Popen([
            browser_path,
            f'--app={url}',
            '--disable-infobars',
            '--no-first-run',
            '--force-device-scale-factor=1',  # 强制缩放比例为1，避免字体变小
        ])
        print(f"[INFO] 已在独立窗口中打开: {url}")
    else:
        webbrowser.open(url)
        print(f"[INFO] 已在默认浏览器中打开: {url}")


# 退出程序的API
@app.route('/api/app/exit', methods=['POST'])
def exit_app():
    """退出程序"""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        # 强制退出
        os._exit(0)
    func()
    return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 50)
    print("SAM3 AN - 数据标注工具")
    print("=" * 50)

    # 在后台线程中等待服务就绪后打开浏览器
    url = "http://localhost:5000"
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    print(f"[INFO] 正在启动服务器...")
    print(f"[INFO] 服务就绪后将自动打开浏览器")
    print("=" * 50)

    # 启动Flask服务器（关闭debug模式以避免重复打开浏览器）
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
