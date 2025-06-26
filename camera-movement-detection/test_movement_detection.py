#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import List, Dict, Tuple
import movement_detector

def create_synthetic_frames(width: int = 640, height: int = 480, num_frames: int = 20) -> List[np.ndarray]:
    """
    Test için sentetik kamera hareketi simülasyonu oluştur
    """
    frames = []
    
    # Temel sahne oluştur (dama tahtası deseni)
    base_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Dama tahtası deseni
    square_size = 40
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                base_frame[i:i+square_size, j:j+square_size] = [255, 255, 255]
            else:
                base_frame[i:i+square_size, j:j+square_size] = [100, 100, 100]
    
    # Bazı özellikler ekle (daireler)
    cv2.circle(base_frame, (150, 150), 30, (0, 255, 0), -1)
    cv2.circle(base_frame, (400, 300), 25, (255, 0, 0), -1)
    cv2.circle(base_frame, (500, 150), 20, (0, 0, 255), -1)
    
    # Kareler oluştur
    for i in range(num_frames):
        frame = base_frame.copy()
        
        # Farklı hareket türleri simüle et
        if 5 <= i <= 7:  # Translation (öteleme)
            dx, dy = i * 10, i * 5
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (width, height))
            
        elif 10 <= i <= 12:  # Rotation (döndürme)
            angle = (i - 10) * 15
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            frame = cv2.warpAffine(frame, M, (width, height))
            
        elif 15 <= i <= 17:  # Scaling (ölçekleme)
            scale = 1.0 + (i - 15) * 0.1
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, 0, scale)
            frame = cv2.warpAffine(frame, M, (width, height))
        
        frames.append(frame)
    
    return frames

def test_detection_accuracy(frames: List[np.ndarray], 
                          expected_movements: List[int]) -> Dict[str, float]:
    """
    Algılama doğruluğunu test et
    """
    detector = movement_detector.MovementDetector()
    result = detector.detect_significant_movement(frames)
    detected = set(result["movement_indices"])
    expected = set(expected_movements)
    
    # Metrics hesapla
    true_positives = len(detected & expected)
    false_positives = len(detected - expected)
    false_negatives = len(expected - detected)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "detected": list(detected),
        "expected": list(expected),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def benchmark_performance(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    Performans benchmark'ı yap
    """
    detector = movement_detector.MovementDetector()
    
    # Zaman ölçümü
    start_time = time.time()
    result = detector.detect_significant_movement(frames)
    end_time = time.time()
    
    processing_time = end_time - start_time
    fps = len(frames) / processing_time
    
    return {
        "total_time": processing_time,
        "fps": fps,
        "frames_processed": len(frames),
        "avg_time_per_frame": processing_time / len(frames)
    }

def test_parameter_sensitivity():
    """
    Parametre hassasiyetini test et
    """
    frames = create_synthetic_frames(num_frames=15)
    expected_movements = [5, 6, 7, 10, 11, 12, 15, 16, 17]  # Bilinen hareket kareleri
    
    # Farklı threshold değerleri test et
    thresholds = [10, 20, 30, 40, 50]
    results = []
    
    for threshold in thresholds:
        detector = movement_detector.MovementDetector(diff_threshold=threshold)
        result = detector.detect_significant_movement(frames)
        accuracy = test_detection_accuracy(frames, expected_movements)
        results.append({
            "threshold": threshold,
            "f1_score": accuracy["f1_score"],
            "precision": accuracy["precision"],
            "recall": accuracy["recall"],
            "detected_count": len(result["movement_indices"])
        })
    
    return results

def visualize_detection_results(frames: List[np.ndarray], 
                              result: Dict, 
                              save_path: str = "detection_results.png"):
    """
    Algılama sonuçlarını görselleştir
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Camera Movement Detection Results', fontsize=16)
    
    # 1. Kombine skorlar
    scores = result["scores"]
    movement_indices = result["movement_indices"]
    
    axes[0, 0].plot(range(1, len(scores) + 1), scores, 'b-', linewidth=2, label='Combined Score')
    if movement_indices:
        movement_scores = [scores[i-1] for i in movement_indices if i-1 < len(scores)]
        axes[0, 0].scatter(movement_indices, movement_scores, color='red', s=100, 
                          label='Movement Detected', zorder=5)
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('Movement Score')
    axes[0, 0].set_title('Combined Movement Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Yöntem karşılaştırması
    method_results = result["method_results"]
    frame_numbers = range(1, len(scores) + 1)
    
    if "frame_diff" in method_results:
        axes[0, 1].plot(frame_numbers, method_results["frame_diff"], 
                       label='Frame Diff', linewidth=2)
    if "feature_matching" in method_results:
        axes[0, 1].plot(frame_numbers, method_results["feature_matching"], 
                       label='Feature Match', linewidth=2)
    if "optical_flow" in method_results:
        axes[0, 1].plot(frame_numbers, method_results["optical_flow"], 
                       label='Optical Flow', linewidth=2)
    
    axes[0, 1].set_xlabel('Frame Number')
    axes[0, 1].set_ylabel('Method Score')
    axes[0, 1].set_title('Method Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. İlk ve son kare karşılaştırması
    if len(frames) >= 2:
        axes[1, 0].imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('First Frame')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Last Frame')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def run_comprehensive_tests():
    """
    Kapsamlı test süitini çalıştır
    """
    print("🎥 Camera Movement Detection - Comprehensive Test Suite")
    print("=" * 60)
    
    # 1. Sentetik veri testi
    print("\n1️⃣ Synthetic Data Test")
    print("-" * 30)
    
    frames = create_synthetic_frames(num_frames=20)
    expected_movements = [6, 7, 8, 11, 12, 13, 16, 17, 18]  # Hareket beklenen kareler
    
    accuracy = test_detection_accuracy(frames, expected_movements)
    print(f"Precision: {accuracy['precision']:.3f}")
    print(f"Recall: {accuracy['recall']:.3f}")
    print(f"F1-Score: {accuracy['f1_score']:.3f}")
    print(f"Detected: {accuracy['detected']}")
    print(f"Expected: {accuracy['expected']}")
    
    # 2. Performans testi
    print("\n2️⃣ Performance Benchmark")
    print("-" * 30)
    
    performance = benchmark_performance(frames)
    print(f"Total Processing Time: {performance['total_time']:.3f} seconds")
    print(f"Processing Speed: {performance['fps']:.2f} FPS")
    print(f"Average Time per Frame: {performance['avg_time_per_frame']*1000:.2f} ms")
    
    # 3. Parametre hassasiyeti testi
    print("\n3️⃣ Parameter Sensitivity Test")
    print("-" * 30)
    
    sensitivity_results = test_parameter_sensitivity()
    print("Threshold | F1-Score | Precision | Recall | Detected")
    print("-" * 50)
    for result in sensitivity_results:
        print(f"{result['threshold']:8d} | {result['f1_score']:8.3f} | "
              f"{result['precision']:9.3f} | {result['recall']:6.3f} | "
              f"{result['detected_count']:8d}")
    
    # 4. Görselleştirme
    print("\n4️⃣ Visualization")
    print("-" * 30)
    
    detector = movement_detector.MovementDetector()
    result = detector.detect_significant_movement(frames)
    visualize_detection_results(frames, result)
    print("Results visualization saved as 'detection_results.png'")
    
    # 5. Özet
    print("\n📊 Test Summary")
    print("-" * 30)
    best_threshold = max(sensitivity_results, key=lambda x: x['f1_score'])
    print(f"Best Threshold: {best_threshold['threshold']}")
    print(f"Best F1-Score: {best_threshold['f1_score']:.3f}")
    print(f"System Performance: {performance['fps']:.1f} FPS")
    
    return {
        "accuracy": accuracy,
        "performance": performance,
        "sensitivity": sensitivity_results,
        "best_config": best_threshold
    }

def test_real_video(video_path: str):
    """
    Gerçek video dosyası ile test
    """
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None
    
    print(f"\n🎬 Testing with real video: {video_path}")
    print("-" * 50)
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Video karelerini oku
    frame_count = 0
    max_frames = 50  # İlk 50 kareyi test et
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    if len(frames) < 2:
        print("❌ Insufficient frames in video")
        return None
    
    # Hareket algılama
    detector = movement_detector.MovementDetector()
    result = detector.detect_significant_movement(frames)
    
    print(f"✅ Processed {len(frames)} frames")
    print(f"🎯 Movement detected in {len(result['movement_indices'])} frames")
    print(f"📊 Movement indices: {result['movement_indices']}")
    
    # Performans
    performance = benchmark_performance(frames)
    print(f"⚡ Processing speed: {performance['fps']:.2f} FPS")
    
    return result

if __name__ == "__main__":
    # Ana test süiti
    test_results = run_comprehensive_tests()
    
    print("\n✅ All tests completed successfully!")
    print("📄 Check 'detection_results.png' for visualization results.")