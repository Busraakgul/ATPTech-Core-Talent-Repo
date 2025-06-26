import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovementDetector:
    """
    Gelişmiş kamera hareketi algılama sınıfı.
    Birden fazla tekniği birleştirerek daha doğru sonuçlar elde eder.
    """
    
    def __init__(self, 
                 diff_threshold: float = 30.0,
                 feature_threshold: float = 0.7,
                 homography_threshold: float = 5.0,
                 min_match_count: int = 10):
        self.diff_threshold = diff_threshold
        self.feature_threshold = feature_threshold
        self.homography_threshold = homography_threshold
        self.min_match_count = min_match_count
        
        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # FLANN matcher for feature matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    def detect_significant_movement(self, frames: List[np.ndarray]) -> Dict:
        """
        Ana fonksiyon: Kamera hareketini algılar
        """
        if len(frames) < 2:
            return {"movement_indices": [], "scores": [], "method_results": {}}
            
        movement_indices = []
        movement_scores = []
        method_results = {
            "frame_diff": [],
            "feature_matching": [],
            "optical_flow": []
        }
        
        prev_gray = None
        prev_keypoints = None
        prev_descriptors = None
        
        for idx in range(1, len(frames)):
            current_frame = frames[idx]
            prev_frame = frames[idx-1]
            
            # Gri tonlamaya çevir
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Frame Differencing Yöntemi
            diff_score = self._frame_difference_method(prev_gray, current_gray)
            method_results["frame_diff"].append(diff_score)
            
            # 2. Feature Matching Yöntemi
            feature_score = self._feature_matching_method(
                prev_gray, current_gray, prev_keypoints, prev_descriptors
            )
            method_results["feature_matching"].append(feature_score)
            
            # 3. Optical Flow Yöntemi
            flow_score = self._optical_flow_method(prev_gray, current_gray)
            method_results["optical_flow"].append(flow_score)
            
            # Skorları birleştir (ağırlıklı ortalama)
            combined_score = (
                0.4 * diff_score + 
                0.4 * feature_score + 
                0.2 * flow_score
            )
            
            movement_scores.append(combined_score)
            
            # Hareket algılandı mı?
            if self._is_significant_movement(diff_score, feature_score, flow_score):
                movement_indices.append(idx)
                
            # Bir sonraki iterasyon için güncelle
            prev_gray = current_gray.copy()
            
        return {
            "movement_indices": movement_indices,
            "scores": movement_scores,
            "method_results": method_results,
            "frame_count": len(frames)
        }
    
    def _frame_difference_method(self, prev_gray: np.ndarray, current_gray: np.ndarray) -> float:
        """Frame differencing ile hareket algılama"""
        try:
            diff = cv2.absdiff(prev_gray, current_gray)
            score = np.mean(diff)
            return min(score / self.diff_threshold * 100, 100)
        except Exception as e:
            logger.warning(f"Frame difference hesaplamasında hata: {e}")
            return 0.0
    
    def _feature_matching_method(self, prev_gray: np.ndarray, current_gray: np.ndarray,
                               prev_keypoints: Optional[List], prev_descriptors: Optional[np.ndarray]) -> float:
        """Feature matching ile hareket algılama"""
        try:
            # Keypoint ve descriptor'ları bul
            kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
            kp2, des2 = self.orb.detectAndCompute(current_gray, None)
            
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return 0.0
            
            # Feature matching
            matches = self.flann.knnMatch(des1, des2, k=2)
            
            # Good matches'i filtrele (Lowe's ratio test)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.feature_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < self.min_match_count:
                return 100.0  # Çok az eşleşme = büyük hareket
            
            # Homografi hesapla
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                return 100.0
            
            # Transformation matrix'den hareket miktarını hesapla
            translation = np.sqrt(M[0, 2]**2 + M[1, 2]**2)
            rotation = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
            scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)
            
            # Normalize edilmiş hareket skoru
            movement_score = min((translation + abs(rotation) + abs(scale_x - 1) * 10 + abs(scale_y - 1) * 10) * 2, 100)
            
            return movement_score
            
        except Exception as e:
            logger.warning(f"Feature matching hesaplamasında hata: {e}")
            return 0.0
    
    def _optical_flow_method(self, prev_gray: np.ndarray, current_gray: np.ndarray) -> float:
        """Optical flow ile hareket algılama"""
        try:
            # Lucas-Kanade optical flow
            # İyi köşeleri bul
            corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            if corners is None or len(corners) < 10:
                return 0.0
            
            # Optical flow hesapla
            flow, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, corners, None)
            
            # İyi flow'ları seç
            good_flow = flow[status == 1]
            good_corners = corners[status == 1]
            
            if len(good_flow) < 5:
                return 100.0
            
            # Flow vektörlerinin büyüklüğünü hesapla
            flow_magnitude = np.sqrt((good_flow[:, 0] - good_corners[:, 0])**2 + 
                                   (good_flow[:, 1] - good_corners[:, 1])**2)
            
            avg_flow = np.mean(flow_magnitude)
            return min(avg_flow * 2, 100)
            
        except Exception as e:
            logger.warning(f"Optical flow hesaplamasında hata: {e}")
            return 0.0
    
    def _is_significant_movement(self, diff_score: float, feature_score: float, flow_score: float) -> bool:
        """Hareketin önemli olup olmadığını belirle"""
        # Adaptif threshold - en az iki yöntemden yüksek skor gelmeli
        high_threshold = 60
        medium_threshold = 40
        
        high_scores = sum([diff_score > high_threshold, 
                          feature_score > high_threshold, 
                          flow_score > high_threshold])
        
        medium_scores = sum([diff_score > medium_threshold, 
                           feature_score > medium_threshold, 
                           flow_score > medium_threshold])
        
        return high_scores >= 1 or medium_scores >= 2

def detect_significant_movement(frames: List[np.ndarray], threshold: float = 50.0) -> List[int]:
    """
    Backward compatibility için basit interface
    """
    detector = MovementDetector(diff_threshold=threshold)
    result = detector.detect_significant_movement(frames)
    return result["movement_indices"]

def get_detailed_analysis(frames: List[np.ndarray]) -> Dict:
    """
    Detaylı analiz için gelişmiş interface
    """
    detector = MovementDetector()
    return detector.detect_significant_movement(frames)