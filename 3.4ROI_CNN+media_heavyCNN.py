import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import glob
import random

# ================= 配置区域 =================
# 输入视频根目录 (确保路径正确)
DATA_ROOT = r"D:\Course Data\STL\CE-CSL\video"
# 输出特征保存目录
OUTPUT_ROOT = r"D:\macang\py\sign_matrices_fusion\3.4"
# 处理的分组顺序
SPLITS = ['dev', 'test', 'train']
# 临时文件保存间隔 (每多少个视频存一次盘，防止内存溢出)
BATCH_SIZE = 10 
# ===========================================

# --- 1. 定义最强特征提取器 (ConvNeXt-L + LayerNorm) ---
class ConvNeXtUltimateExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading ConvNeXt-Large on {self.device} (First time run will download ~800MB)...")
        # 加载 ConvNeXt-Large
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        self.backbone = models.convnext_large(weights=weights)
        
        # 移除分类头 (Classifier)，使其输出 1536 维特征
        # ConvNeXt 的结构通常以 AdaptiveAvgPool 结束，接入 Classifier
        # 我们把 Classifier 替换为 Identity，保留 Pooling 后的特征
        self.backbone.classifier = nn.Identity()
        
        # *** 关键点：LayerNorm 实现量级对齐 ***
        # 将 1536 维特征强制拉回 (Mean=0, Std=1) 的分布
        self.norm = nn.LayerNorm(1536)
        
        self.to(self.device)
        self.eval()
        
        # 官方推荐的预处理 (Resize=232, Crop=224)
        self.preprocess = transforms.Compose([
            transforms.Resize(232, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, cv2_frame):
        # 转为 PIL 并预处理
        img_rgb = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 1. 提取原始特征 [1, 1536]
            features = self.backbone(input_tensor)
            # 2. 执行 LayerNorm 标准化
            features = self.norm(features)
            
        return features.cpu()

# --- 2. 平滑 ROI 裁剪器 (带记忆功能) ---
class SmoothedROICropper:
    def __init__(self, target_size=224, max_memory=5):
        self.target_size = target_size
        self.last_bboxes = {'face': None, 'lh': None, 'rh': None}
        self.lost_count = {'face': 0, 'lh': 0, 'rh': 0}
        self.max_memory = max_memory

    def get_bbox(self, landmarks, w, h, key, padding=0.2):
        if landmarks:
            # 正常检测到
            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]
            x1, x2 = min(xs)*w, max(xs)*w
            y1, y2 = min(ys)*h, max(ys)*h
            bw, bh = x2-x1, y2-y1
            # 增加 Padding
            pad_w, pad_h = bw * padding, bh * padding
            x1 = max(0, int(x1 - pad_w))
            y1 = max(0, int(y1 - pad_h))
            x2 = min(w, int(x2 + pad_w))
            y2 = min(h, int(y2 + pad_h))
            
            bbox = (x1, y1, x2, y2)
            self.last_bboxes[key] = bbox
            self.lost_count[key] = 0
            return bbox
        
        # 未检测到，尝试记忆回溯
        self.lost_count[key] += 1
        if self.last_bboxes[key] and self.lost_count[key] <= self.max_memory:
            return self.last_bboxes[key]
        
        return None # 彻底丢失

    def crop(self, frame, bbox, out_w, out_h):
        if bbox is None: return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        if x2<=x1 or y2<=y1: return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        try:
            return cv2.resize(frame[y1:y2, x1:x2], (out_w, out_h))
        except:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    def process(self, frame, results):
        h, w, _ = frame.shape
        # 获取 BBox (带平滑)
        bb_face = self.get_bbox(results.face_landmarks, w, h, 'face', 0.1)
        bb_lh = self.get_bbox(results.left_hand_landmarks, w, h, 'lh', 0.2)
        bb_rh = self.get_bbox(results.right_hand_landmarks, w, h, 'rh', 0.2)
        
        # 拼图: 上面是脸(224x112)，下面左右是手(112x112)
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        canvas[0:112, :] = self.crop(frame, bb_face, 224, 112)
        canvas[112:, 0:112] = self.crop(frame, bb_lh, 112, 112)
        canvas[112:, 112:] = self.crop(frame, bb_rh, 112, 112)
        return canvas

# --- 3. 计算 531 维 MediaPipe 特征 (保持 0 值) ---
def calc_mp_features(raw_data):
    # raw_data: List of [75 points * 4 dims]
    v_feat = np.array(raw_data)
    if v_feat.ndim < 3: return None
    frames_count = v_feat.shape[0]
    
    # 空间归一化 (肩膀为中心)
    # 注意：如果肩膀没检测到(0,0,0)，这一步计算结果仍为0，不影响 Mask 逻辑
    shoulder_center = (v_feat[:, 11, :3] + v_feat[:, 12, :3]) / 2
    v_feat[:, :, :3] -= shoulder_center[:, np.newaxis, :]
    
    shoulder_dist = np.linalg.norm(v_feat[:, 11, :3] - v_feat[:, 12, :3], axis=1)
    scale = np.mean(shoulder_dist) + 1e-6
    v_feat[:, :, :3] /= scale
    
    # 提取特征
    nose = v_feat[:, 0, :3]
    lw = v_feat[:, 15, :3]
    rw = v_feat[:, 16, :3]
    
    # 展平 [T, 300]
    flat_feat = v_feat.reshape(frames_count, -1)
    # 坐标 [T, 225]
    coord_feat = v_feat[:, :, :3].reshape(frames_count, -1)
    # 速度 [T, 225]
    velocity = np.diff(coord_feat, axis=0, prepend=coord_feat[0:1])
    # 相对距离 [T, 3] + [T, 3]
    lw_nose = lw - nose
    rw_nose = rw - nose
    
    # 拼接 [T, 531]
    mp_531 = np.concatenate([flat_feat, velocity, lw_nose, rw_nose], axis=-1).astype(np.float32)
    return torch.from_numpy(mp_531)

# --- 4. 单视频处理逻辑 ---
def process_single_video(video_path, holistic, cropper, cnn):
    cap = cv2.VideoCapture(video_path)
    raw_mp_data = []
    cnn_features = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # A. MediaPipe 原始数据收集 (含 0 值)
        def get_lms(res_lm, count):
            if res_lm: return [[lm.x, lm.y, lm.z, lm.visibility] for lm in res_lm.landmark]
            return [[0.0]*4] * count
        
        frame_mp = []
        frame_mp.extend(get_lms(results.pose_landmarks, 33))
        frame_mp.extend(get_lms(results.left_hand_landmarks, 21))
        frame_mp.extend(get_lms(results.right_hand_landmarks, 21))
        raw_mp_data.append(frame_mp)
        
        # B. CNN 特征提取 (平滑 ROI + ConvNeXt + LayerNorm)
        roi_img = cropper.process(frame, results)
        cnn_features.append(cnn.extract(roi_img))
        
    cap.release()
    if not raw_mp_data: return None

    # 合并
    mp_tensor = calc_mp_features(raw_mp_data) # [T, 531]
    cnn_tensor = torch.cat(cnn_features, dim=0) # [T, 1536]
    
    # 截断对齐
    min_len = min(len(mp_tensor), len(cnn_tensor))
    # 最终维度 [T, 1536 + 531] = [T, 2067]
    final_tensor = torch.cat((cnn_tensor[:min_len], mp_tensor[:min_len]), dim=1)
    return final_tensor

# --- 5. 量级分布报告生成器 ---
def generate_scale_report(data_list, split_name):
    print(f"\n📊 [{split_name}] 特征量级体检报告 (基于 LayerNorm 对齐后)")
    print("-" * 60)
    
    # 为了速度，随机采样最多 5000 帧进行统计，避免内存爆炸
    sample_frames = []
    total_samples_needed = 5000
    
    # 从列表中随机选一些视频采样
    random.shuffle(data_list)
    for tensor in data_list:
        if len(sample_frames) >= total_samples_needed: break
        # 每个视频取前 10 帧
        frames_to_take = min(10, tensor.shape[0])
        sample_frames.append(tensor[:frames_to_take])
        
    if not sample_frames:
        print("无数据，无法生成报告。")
        return

    # 拼接采样数据
    stacked_samples = torch.cat(sample_frames, dim=0) # [N, 2067]
    
    # 分割 CNN 和 MP
    cnn_part = stacked_samples[:, :1536]
    mp_part = stacked_samples[:, 1536:]
    
    # 计算统计量
    cnn_mean, cnn_std = cnn_part.mean().item(), cnn_part.std().item()
    mp_mean, mp_std = mp_part.mean().item(), mp_part.std().item()
    mp_abs_mean = mp_part.abs().mean().item() # 骨架特征均值接近0，看绝对值均值更有意义
    
    print(f"1. ConvNeXt (1536维) | 均值: {cnn_mean:6.3f} | 标准差: {cnn_std:6.3f}")
    print(f"2. MediaPipe (531维)  | 均值: {mp_mean:6.3f} | 标准差: {mp_std:6.3f} | 绝对值均值: {mp_abs_mean:6.3f}")
    
    print("-" * 60)
    if 0.5 < cnn_std < 1.5 and 0.1 < mp_std < 3.0:
        print("✅ 结论：量级对齐良好！两者处于同一数量级，利于模型训练。")
    else:
        print("⚠️ 结论：量级差异可能较大，请检查 LayerNorm 是否生效。")
    print("=" * 60 + "\n")

# --- 6. 主流程 ---
def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
        
    mp_holistic = mp.solutions.holistic
    # 初始化最强提取器
    cnn = ConvNeXtUltimateExtractor()
    
    for split in SPLITS:
        split_path = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_path): continue
            
        print(f"\n🚀 开始处理数据集: {split}")
        all_videos = glob.glob(os.path.join(split_path, "**", "*.mp4"), recursive=True)
        all_videos.sort()
        print(f"   发现视频数: {len(all_videos)}")
        
        temp_buffer = []
        batch_idx = 0
        
        with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, 
                                  min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
            
            for i, vid_path in enumerate(tqdm(all_videos, desc=f"Processing {split}")):
                cropper = SmoothedROICropper(max_memory=5)
                
                try:
                    tensor = process_single_video(vid_path, holistic, cropper, cnn)
                    if tensor is not None:
                        temp_buffer.append(tensor)
                except Exception as e:
                    print(f"Error processing {vid_path}: {e}")
                
                # 批次保存
                if len(temp_buffer) >= BATCH_SIZE:
                    torch.save(temp_buffer, os.path.join(OUTPUT_ROOT, f"temp_{split}_{batch_idx}.pt"))
                    temp_buffer = []
                    batch_idx += 1
            
            if temp_buffer:
                torch.save(temp_buffer, os.path.join(OUTPUT_ROOT, f"temp_{split}_{batch_idx}.pt"))
                
        # --- 合并与报告 ---
        print(f"📦 正在合并 {split} 数据...")
        full_data = []
        temp_files = glob.glob(os.path.join(OUTPUT_ROOT, f"temp_{split}_*.pt"))
        temp_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        for tf in temp_files:
            full_data.extend(torch.load(tf))
            
        final_path = os.path.join(OUTPUT_ROOT, f"{split}_features_ultimate.pt")
        # 保存为列表格式 [tensor1, tensor2, ...]
        torch.save(full_data, final_path)
        
        # *** 生成并打印体检报告 ***
        generate_scale_report(full_data, split)
        
        # 清理临时文件
        for tf in temp_files:
            os.remove(tf)
        print(f"✅ {split} 完成。")

if __name__ == "__main__":
    main()