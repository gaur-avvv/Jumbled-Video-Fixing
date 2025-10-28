#!/usr/bin/env python3
#fix jumbled video - all-in-one workflow

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import shutil
import argparse


def extract_frames(video_file, output_folder):
    #extract all frames from the jumbled video
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


def load_ai_model():
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


def prepare_image_transform():
    #image preprocessing
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def analyze_frames_with_ai(frames_folder, model, device):
    #analyze frames with AI
    print("\nanalyzing frames...")
    
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    transform = prepare_image_transform()
    
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


def find_similar_frames(features):
    #compare all frames
    print("\ncomparing frames...")
    similarity_matrix = cosine_similarity(features)
    print("done")
    return similarity_matrix


def find_possible_start_frames(similarity_matrix):
    #find possible start frames
    print("\nfinding start frames...")
    
    neighbor_counts = []
    for i in range(len(similarity_matrix)):
        similar_count = np.sum(similarity_matrix[i] > 0.95) - 1
        neighbor_counts.append(similar_count)
    
    possible_starts = np.argsort(neighbor_counts)[:10]
    print(f"found {len(possible_starts)} candidates")
    return possible_starts


def build_frame_sequence(similarity_matrix, start_frame):
    #build sequence from start frame
    sequence = [start_frame]
    used_frames = {start_frame}
    current = start_frame
    
    for _ in range(len(similarity_matrix) - 1):
        similarities = similarity_matrix[current].copy()
        for used in used_frames:
            similarities[used] = -1
        
        next_frame = np.argmax(similarities)
        sequence.append(next_frame)
        used_frames.add(next_frame)
        current = next_frame
    
    return sequence


def calculate_sequence_quality(similarity_matrix, sequence):
    #calculate quality score
    scores = [similarity_matrix[sequence[i], sequence[i+1]] for i in range(len(sequence) - 1)]
    return np.mean(scores)


def find_correct_sequence(similarity_matrix, possible_starts):
    #find best sequence
    print("\nfinding correct sequence...")
    
    best_sequence = None
    best_quality = 0
    
    for start_frame in possible_starts:
        sequence = build_frame_sequence(similarity_matrix, start_frame)
        quality = calculate_sequence_quality(similarity_matrix, sequence)
        
        if quality > best_quality:
            best_quality = quality
            best_sequence = sequence
    
    print(f"quality: {best_quality*100:.1f}%")
    return best_sequence, best_quality


def check_if_backward(similarity_matrix, sequence):
    #check if video is backward
    first_frame = sequence[0]
    similarities = [similarity_matrix[first_frame, sequence[i]] for i in range(1, min(30, len(sequence)))]
    trend = np.polyfit(range(len(similarities)), similarities, 1)[0]
    
    if trend > 0:
        print("reversing sequence...")
        sequence = sequence[::-1]
    
    quality = calculate_sequence_quality(similarity_matrix, sequence)
    return sequence, quality


def create_fixed_video(frames_folder, sequence, video_info, output_file):
    #create fixed video
    print(f"\ncreating video...")
    
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, video_info['fps'], 
                                   (video_info['width'], video_info['height']))
    
    for frame_idx in sequence:
        frame = cv2.imread(f"{frames_folder}/{frame_files[frame_idx]}")
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"video created: {file_size:.1f} MB")
    return file_size


def get_quality_stats(similarity_matrix, sequence):
    #get quality stats
    scores = [similarity_matrix[sequence[i], sequence[i+1]] for i in range(len(sequence) - 1)]
    return {
        'mean': np.mean(scores),
        'min': np.min(scores),
        'max': np.max(scores)
    }


def print_summary(video_info, quality_stats, total_time, output_file):
    #print summary
    print(f"\noutput: {output_file}")
    print(f"quality: {quality_stats['mean']*100:.1f}%")
    print(f"time: {total_time:.1f} seconds")


def fix_jumbled_video(input_video, output_video, keep_temp_files=False):
    #fix jumbled video
    print(f"\nprocessing: {input_video}")
    
    start_time = time.time()
    temp_folder = "./temp_frames"
    
    try:
        #extract frames
        video_info = extract_frames(input_video, temp_folder)
        if not video_info:
            return False
        
        #load model
        model, device = load_ai_model()
        
        #analyze frames
        features = analyze_frames_with_ai(temp_folder, model, device)
        
        #find similarities
        similarity_matrix = find_similar_frames(features)
        
        #reconstruct sequence
        possible_starts = find_possible_start_frames(similarity_matrix)
        sequence, quality = find_correct_sequence(similarity_matrix, possible_starts)
        sequence, quality = check_if_backward(similarity_matrix, sequence)
        
        #create video
        create_fixed_video(temp_folder, sequence, video_info, output_video)
        
        #show results
        quality_stats = get_quality_stats(similarity_matrix, sequence)
        total_time = time.time() - start_time
        print_summary(video_info, quality_stats, total_time, output_video)
        
        #cleanup
        if not keep_temp_files and os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        
        return True
        
    except Exception as e:
        print(f"\nerror: {e}")
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        return False


def main():
    parser = argparse.ArgumentParser(description='fix jumbled videos')
    parser.add_argument('input', help='input video file')
    parser.add_argument('output', help='output video file')
    parser.add_argument('--keep-temp-files', action='store_true', help='keep temporary files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"error: file not found: {args.input}")
        return 1
    
    if os.path.exists(args.output):
        response = input(f"{args.output} exists. overwrite? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    success = fix_jumbled_video(args.input, args.output, args.keep_temp_files)
    
    if success:
        print("\ndone")
        return 0
    else:
        print("\nfailed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
