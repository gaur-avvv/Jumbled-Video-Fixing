#!/usr/bin/env python3
#fix jumbled video using prepared data

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import shutil
import argparse
import time


def load_prepared_data(frames_folder):
    #load prepared data
    print("\nloading prepared data...")
    
    with open(f"{frames_folder}/video_info.pkl", 'rb') as f:
        video_info = pickle.load(f)
    
    features = np.load(f"{frames_folder}/features.npy")
    
    print(f"loaded {len(features)} frame features")
    return video_info, features


def find_similarities(features):
    #compare all frames
    print("\ncomparing frames...")
    similarity_matrix = cosine_similarity(features)
    print("done")
    return similarity_matrix


def find_start_candidates(similarity_matrix):
    #find possible start frames
    print("\nfinding start frames...")
    
    neighbor_counts = []
    for i in range(len(similarity_matrix)):
        similar_count = np.sum(similarity_matrix[i] > 0.95) - 1
        neighbor_counts.append(similar_count)
    
    candidates = np.argsort(neighbor_counts)[:10]
    print(f"found {len(candidates)} candidates")
    return candidates


def build_sequence(similarity_matrix, start_frame):
    #build sequence from start frame
    sequence = [start_frame]
    used = {start_frame}
    current = start_frame
    
    for _ in range(len(similarity_matrix) - 1):
        similarities = similarity_matrix[current].copy()
        for u in used:
            similarities[u] = -1
        
        next_frame = np.argmax(similarities)
        sequence.append(next_frame)
        used.add(next_frame)
        current = next_frame
    
    return sequence


def get_quality(similarity_matrix, sequence):
    #calculate quality score
    scores = [similarity_matrix[sequence[i], sequence[i+1]] for i in range(len(sequence) - 1)]
    return np.mean(scores)


def find_best_sequence(similarity_matrix, candidates):
    #find best sequence
    print("\nfinding correct sequence...")
    
    best_sequence = None
    best_quality = 0
    
    for start in candidates:
        sequence = build_sequence(similarity_matrix, start)
        quality = get_quality(similarity_matrix, sequence)
        
        if quality > best_quality:
            best_quality = quality
            best_sequence = sequence
    
    print(f"quality: {best_quality*100:.1f}%")
    return best_sequence, best_quality


def check_direction(similarity_matrix, sequence):
    #check if video is backward
    first = sequence[0]
    similarities = [similarity_matrix[first, sequence[i]] for i in range(1, min(30, len(sequence)))]
    trend = np.polyfit(range(len(similarities)), similarities, 1)[0]
    
    if trend > 0:
        print("reversing sequence...")
        sequence = sequence[::-1]
    
    quality = get_quality(similarity_matrix, sequence)
    return sequence, quality


def create_video(frames_folder, sequence, video_info, output_file):
    #create fixed video
    print(f"\ncreating video...")
    
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, video_info['fps'], 
                                   (video_info['width'], video_info['height']))
    
    for idx in sequence:
        frame = cv2.imread(f"{frames_folder}/{frame_files[idx]}")
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"video created: {file_size:.1f} MB")


def show_results(quality, total_time, output_file):
    #show results
    print("\nvideo fixed")
    print(f"output file: {output_file}")
    print(f"quality: {quality*100:.1f}%")
    print(f"processing time: {total_time:.1f} seconds")


def fix_video(frames_folder, output_file, cleanup=False):
    #fix jumbled video
    print("\nstarting video fix...")
    
    start_time = time.time()
    
    try:
        video_info, features = load_prepared_data(frames_folder)
        similarity_matrix = find_similarities(features)
        candidates = find_start_candidates(similarity_matrix)
        sequence, quality = find_best_sequence(similarity_matrix, candidates)
        sequence, quality = check_direction(similarity_matrix, sequence)
        create_video(frames_folder, sequence, video_info, output_file)
        
        total_time = time.time() - start_time
        show_results(quality, total_time, output_file)
        
        if cleanup:
            print(f"\nremoving {frames_folder}")
            shutil.rmtree(frames_folder)
        
        return True
        
    except Exception as e:
        print(f"\nerror: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='fix jumbled video')
    parser.add_argument('frames', help='frames folder with prepared data')
    parser.add_argument('output', help='output video file')
    parser.add_argument('--cleanup', action='store_true', help='delete frames after fixing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.frames):
        print(f"error: frames folder not found: {args.frames}")
        print("run prepare_model.py first!")
        return 1
    
    if not os.path.exists(f"{args.frames}/video_info.pkl"):
        print(f"error: prepared data not found in {args.frames}")
        print("run prepare_model.py first!")
        return 1
    
    success = fix_video(args.frames, args.output, args.cleanup)
    
    if success:
        print("\ncompleted successfully")
        return 0
    else:
        print("\nfailed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
