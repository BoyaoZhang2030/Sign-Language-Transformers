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
import gc  # 导入垃圾回收机制

# ================= 配置区域 =================
DATA_ROOT = r"D:\Course Data\STL\CE-CSL\video"
OUTPUT_ROOT = r"D:\macang\py\sign_matrices_fusion\3.5"
SPLITS = ['dev', 'test', 'train']
BATCH_SIZE = 10 
EXPECTED_DIM = 1811  # 1280 (CNN) + 531 (MP)
# ===========================================

class MobileNetV3AlignedExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading MobileNetV3-Large on {self.device}...")
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        full_model = models.mobilenet_v3_large(weights=weights)
        self.features = full_model.features
        self.avgpool = full_model.avgpool
        self.fc = full_model.classifier[0]
        self.norm = nn.LayerNorm(1280)
        self.to(self.device)
        self.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, cv2_frame):
        img_rgb = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x = self.features(input_tensor)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            x = self.norm(x)
        return x.cpu()

class SmoothedROICropper:
    def __init__(self, target_size=224, max_memory=5):
        self.target_size = target_size
        self.last_bboxes = {'face': None, 'lh': None, 'rh': None}
        self.lost_count = {'face': 0, 'lh': 0, 'rh': 0}
        self.max_memory = max_memory

    def get_bbox(self, landmarks, w, h, key, padding=0.2):
        if landmarks:
            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]
            x1, x2 = min(xs)*w, max(xs)*w
            y1, y2 = min(ys)*h, max(ys)*h
            bw, bh = x2-x1, y2-y1
            x1, y1 = max(0, int(x1 - bw*padding)), max(0, int(y1 - bh*padding))
            x2, y2 = min(w, int(x2 + bw*padding)), min(h, int(y2 + bh*padding))
            bbox = (x1, y1, x2, y2)
            self.last_bboxes[key] = bbox
            self.lost_count[key] = 0
            return bbox
        self.lost_count[key] += 1
        if self.last_bboxes[key] and self.lost_count[key] <= self.max_memory:
            return self.last_bboxes[key]
        return None

    def crop(self, frame, bbox, out_w, out_h):
        if bbox is None: return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        try: return cv2.resize(frame[y1:y2, x1:x2], (out_w, out_h))
        except: return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    def process(self, frame, results):
        h, w, _ = frame.shape
        bb_f = self.get_bbox(results.face_landmarks, w, h, 'face', 0.1)
        bb_l = self.get_bbox(results.left_hand_landmarks, w, h, 'lh', 0.2)
        bb_r = self.get_bbox(results.right_hand_landmarks, w, h, 'rh', 0.2)
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        canvas[0:112, :] = self.crop(frame, bb_f, 224, 112)
        canvas[112:, 0:112] = self.crop(frame, bb_l, 112, 112)
        canvas[112:, 112:] = self.crop(frame, bb_r, 112, 112)
        return canvas

def calc_mp_features(raw_data):
    v_feat = np.array(raw_data)
    if v_feat.ndim < 3: return None
    frames_count = v_feat.shape[0]
    sc = (v_feat[:, 11, :3] + v_feat[:, 12, :3]) / 2
    v_feat[:, :, :3] -= sc[:, np.newaxis, :]
    dist = np.linalg.norm(v_feat[:, 11, :3] - v_feat[:, 12, :3], axis=1)
    scale = np.mean(dist) + 1e-6
    v_feat[:, :, :3] /= scale
    flat = v_feat.reshape(frames_count, -1)
    coord = v_feat[:, :, :3].reshape(frames_count, -1)
    vel = np.diff(coord, axis=0, prepend=coord[0:1])
    ln, rn = v_feat[:, 0, :3], v_feat[:, 15, :3]
    lw_n, rw_n = rn - ln, v_feat[:, 16, :3] - ln
    return torch.from_numpy(np.concatenate([flat, vel, lw_n, rw_n], axis=-1).astype(np.float32))

def main():
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
    mp_holistic = mp.solutions.holistic
    extractor = MobileNetV3AlignedExtractor()
    
    for split in SPLITS:
        split_path = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_path): continue
        
        # --- 解决重复文件隐患 ---
        raw_vids = glob.glob(os.path.join(split_path, "**", "*.mp4"), recursive=True)
        unique_vids = {}
        for v in raw_vids:
            # 仅保留标准的 train-XXXXX 核心名
            base_name = os.path.basename(v).split('.')[0].replace(" - 副本", "").strip()
            if base_name not in unique_vids:
                unique_vids[base_name] = v
        
        all_vids = [unique_vids[k] for k in sorted(unique_vids.keys())]
        print(f"\n🚀 去重后 Processing {split}: {len(all_vids)} videos")
        
        temp_buffer, batch_idx = [], 0
        for i, v_path in enumerate(tqdm(all_vids)):
            # 每个视频重置一次 Holistic，解决状态残留问题
            with mp_holistic.Holistic(model_complexity=1) as holistic:
                cropper = SmoothedROICropper()
                cap = cv2.VideoCapture(v_path)
                raw_mp, cnn_fs = [], []
                
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    def get_lms(l, c): return [[m.x,m.y,m.z,m.visibility] for m in l.landmark] if l else [[0.0]*4]*c
                    raw_mp.append(get_lms(res.pose_landmarks,33)+get_lms(res.left_hand_landmarks,21)+get_lms(res.right_hand_landmarks,21))
                    cnn_fs.append(extractor.extract(cropper.process(frame, res)))
                cap.release()

                # --- 解决索引断裂隐患 ---
                if raw_mp and len(cnn_fs) > 0:
                    mp_t = calc_mp_features(raw_mp)
                    cnn_t = torch.cat(cnn_fs, dim=0)
                    L = min(len(mp_t), len(cnn_t))
                    combined = torch.cat((cnn_t[:L], mp_t[:L]), dim=1)
                    temp_buffer.append(combined)
                else:
                    # 如果视频处理失败，插入全零占位符，保持索引一一对应
                    print(f"\n⚠️ Warning: {os.path.basename(v_path)} failed. Inserting placeholder.")
                    temp_buffer.append(torch.zeros((1, EXPECTED_DIM)))
            
            # 分批保存，释放内存
            if len(temp_buffer) >= BATCH_SIZE:
                torch.save(temp_buffer, os.path.join(OUTPUT_ROOT, f"temp_{split}_{batch_idx}.pt"))
                del temp_buffer
                gc.collect() # 强制清理显存和内存
                temp_buffer = []
                batch_idx += 1

        if temp_buffer:
            torch.save(temp_buffer, os.path.join(OUTPUT_ROOT, f"temp_{split}_{batch_idx}.pt"))

        # --- 解决内存堆积压力 ---
        print(f"正在逐个合并临时文件并释放内存...")
        full_data = []
        temps = sorted(glob.glob(os.path.join(OUTPUT_ROOT, f"temp_{split}_*.pt")), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        for t in temps:
            part_data = torch.load(t)
            full_data.extend(part_data)
            del part_data # 立即删除临时加载的小列表
            os.remove(t)
            gc.collect()
            
        final_path = os.path.join(OUTPUT_ROOT, f"{split}_features_mobilenet.pt")
        torch.save(full_data, final_path)
        print(f"✅ 处理完成！最终长度: {len(full_data)}, 保存路径: {final_path}")
        
        del full_data
        gc.collect()

if __name__ == "__main__":
    main()