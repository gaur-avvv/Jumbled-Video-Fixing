#!/usr/bin/env python3
#prepare model - extract frames and analyze with AI

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle
import argparse


def extract_frames(video_file, output_folder):
    #extract all frames from video
    print(f"\nextracting frames from {video_file}...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    video = cv2.VideoCapture(video_file)
    if not video.isOpened():
        print(f"error: couldn't open video file")
        return None
    
    #get video info
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"{width}x{height} at {fps}fps")
    
    #extract frames
    count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        cv2.imwrite(f"{output_folder}/frame{count:04d}.jpg", frame)
        count += 1
    
    video.release()
    print(f"extracted {count} frames")
    
    return {'fps': fps, 'width': width, 'height': height, 'total_frames': count}


def load_model():
    #load resnet50 model
    print("\nloading AI model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")
    
    resnet50 = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(resnet50.children())[:-1])
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print("model loaded")
    return model, device


def prepare_transform():
    #image preprocessing
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def analyze_frames(frames_folder, model, device):
    #analyze frames with AI
    print("\nanalyzing frames...")
    
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    transform = prepare_transform()
    
    all_features = []
    batch_size = 32
    
    for i in range(0, len(frame_files), batch_size):
        batch_files = frame_files[i:i+batch_size]
        
        #load images
        batch_images = []
        for filename in batch_files:
            img = Image.open(f"{frames_folder}/{filename}").convert('RGB')
            batch_images.append(transform(img))
        
        #get features
        batch = torch.stack(batch_images).to(device)
        with torch.no_grad():
            features = model(batch)
        
        features = features.squeeze().cpu().numpy()
        if len(batch_files) == 1:
            features = features.reshape(1, -1)
        all_features.append(features)
    
    print(f"analyzed {len(frame_files)} frames")
    return np.vstack(all_features)


def save_data(frames_folder, video_info, features):
    #save prepared data
    print("\nsaving data...")
    
    #save video info
    with open(f"{frames_folder}/video_info.pkl", 'wb') as f:
        pickle.dump(video_info, f)
    
    #save features
    np.save(f"{frames_folder}/features.npy", features)
    
    print("data saved")


def prepare_video(video_file, output_folder="./frames"):
    #prepare video for fixing
    print("\npreparing model data...")

    try:
        #extract frames
        video_info = extract_frames(video_file, output_folder)
        if not video_info:
            return False
        
        #load model
        model, device = load_model()
        
        #analyze frames
        features = analyze_frames(output_folder, model, device)
        
        #save data
        save_data(output_folder, video_info, features)
        
        print("\npreparation complete")
        print(f"frames folder: {output_folder}")
        print(f"total frames: {video_info['total_frames']}")
        print(f"next step: python fix_video.py {output_folder} output.mp4")
        
        return True
        
    except Exception as e:
        print(f"\nerror: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='prepare model for video fixing')
    parser.add_argument('input', help='input video file')
    parser.add_argument('--output', default='./frames', help='output folder (default: ./frames)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"error: file not found: {args.input}")
        return 1
    
    success = prepare_video(args.input, args.output)
    
    if success:
        print("\ncompleted successfully")
        return 0
    else:
        print("\nfailed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
