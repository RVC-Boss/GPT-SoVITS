"""
detectors.py - Face Detection Algorithms and Validators
Contains different face detection methods and quality validation
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class EnhancedFaceDetector:
    """Enhanced face detector using multiple detection methods"""
    
    def __init__(self):
        self.detectors = {}
        self.load_detectors()
    
    def load_detectors(self):
        """Load multiple face detection algorithms"""
        # 1. MediaPipe Face Detection (primary)
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1, 
                min_detection_confidence=0.5  # Relaxed threshold
            )
            self.detectors['mediapipe'] = True
            logger.info("✅ MediaPipe Face Detector loaded")
        except ImportError:
            logger.warning("⚠️ MediaPipe not available - install with: pip install mediapipe")
            self.detectors['mediapipe'] = False
        
        # 2. Haar Cascade (fallback)
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.detectors['haar'] = True
            logger.info("✅ Haar cascade loaded as fallback")
        except Exception as e:
            logger.error(f"❌ Haar cascade loading error: {e}")
            self.detectors['haar'] = False
    
    def detect_faces_mediapipe(self, image):
        """Detect faces using MediaPipe (most accurate)"""
        if not self.detectors.get('mediapipe', False):
            return []
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detector.process(rgb_image)
            
            faces = []
            if results.detections:
                h, w = image.shape[:2]
                
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    confidence = detection.score[0]
                    
                    # Convert relative coordinates to absolute
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure face is within image bounds
                    x = min(x, w - 1)
                    y = min(y, h - 1)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width > 30 and height > 30:  # Minimum size
                        faces.append((x, y, width, height, confidence))
            
            return faces
            
        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")
            return []
    
    def detect_faces_haar(self, image):
        """Detect faces using Haar cascades (fallback)"""
        if not self.detectors.get('haar', False):
            return []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3,      # Relaxed from 5
                minSize=(30, 30),    # Relaxed from (80, 80)
                maxSize=(500, 500)
            )
            
            # Add estimated confidence
            faces_with_conf = []
            for (x, y, w, h) in faces:
                # Estimate confidence based on face size
                confidence = min(0.9, max(0.5, (w * h) / (100 * 100)))
                faces_with_conf.append((x, y, w, h, confidence))
            
            return faces_with_conf
            
        except Exception as e:
            logger.error(f"Haar detection error: {e}")
            return []
    
    def detect_faces_dnn(self, image):
        """Detect faces using OpenCV DNN (optional third method)"""
        try:
            # This requires pre-trained DNN model files
            # For now, we'll skip this implementation
            # You can add DNN detection here if you have the model files
            return []
        except Exception as e:
            logger.error(f"DNN detection error: {e}")
            return []
    
    def detect_faces(self, image):
        """Main face detection using best available method"""
        all_faces = []
        
        # Try MediaPipe first (most accurate)
        faces = self.detect_faces_mediapipe(image)
        if faces:
            all_faces.extend([(x, y, w, h, conf, 'mediapipe') for x, y, w, h, conf in faces])
        
        # If no MediaPipe faces or low confidence, try Haar
        if not all_faces or max([f[4] for f in all_faces]) < 0.7:
            faces = self.detect_faces_haar(image)
            all_faces.extend([(x, y, w, h, conf, 'haar') for x, y, w, h, conf in faces])
        
        # Filter and remove duplicates
        filtered_faces = self.filter_and_deduplicate_faces(all_faces)
        
        return filtered_faces
    
    def filter_and_deduplicate_faces(self, faces):
        """Remove duplicate faces and apply quality filters"""
        if not faces:
            return []
        
        # Sort by confidence (highest first)
        faces.sort(key=lambda x: x[4], reverse=True)
        
        filtered = []
        for face in faces:
            x, y, w, h, conf, method = face
            
            # Apply quality checks
            if not self.is_valid_face_detection(x, y, w, h, conf):
                continue
            
            # Check for overlap with existing faces
            is_duplicate = False
            for existing in filtered:
                if self.faces_overlap(face, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(face)
        
        return filtered
    
    def is_valid_face_detection(self, x, y, w, h, confidence):
        """Validate face detection quality"""
        # Minimum confidence threshold (relaxed)
        if confidence < 0.3:
            return False
        
        # Minimum size check (relaxed)
        if w < 30 or h < 30:
            return False
        
        # Aspect ratio check (more permissive)
        aspect_ratio = w / h
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False
        
        # Coordinates should be positive
        if x < 0 or y < 0:
            return False
        
        return True
    
    def faces_overlap(self, face1, face2, threshold=0.5):
        """Check if two face detections overlap significantly"""
        x1, y1, w1, h1 = face1[:4]
        x2, y2, w2, h2 = face2[:4]
        
        # Calculate intersection area
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return False
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union area
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        # Calculate IoU (Intersection over Union)
        iou = intersection / union if union > 0 else 0
        
        return iou > threshold


class FaceQualityValidator:
    """Validate face quality before analysis"""
    
    @staticmethod
    def is_face_clear(face_img, blur_threshold=30):
        """Check if face is clear enough (not too blurry)"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var > blur_threshold
        except:
            return True  # If check fails, assume it's OK
    
    @staticmethod
    def has_sufficient_size(face_img, min_size=30):
        """Check if face has sufficient resolution"""
        h, w = face_img.shape[:2]
        return min(h, w) >= min_size
    
    @staticmethod
    def is_properly_aligned(face_img):
        """Basic check for face alignment (relaxed)"""
        try:
            h, w = face_img.shape[:2]
            # More permissive aspect ratio
            aspect_ratio = w / h
            return 0.3 <= aspect_ratio <= 3.0
        except:
            return True  # If check fails, assume it's OK
    
    @staticmethod
    def has_good_contrast(face_img, min_std=20):
        """Check if face has sufficient contrast"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            std_dev = np.std(gray)
            return std_dev > min_std
        except:
            return True  # If check fails, assume it's OK
    
    @staticmethod
    def is_well_lit(face_img, min_brightness=30, max_brightness=220):
        """Check if face is well lit (not too dark or overexposed)"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            return min_brightness < mean_brightness < max_brightness
        except:
            return True  # If check fails, assume it's OK
    
    @staticmethod
    def validate_face(face_img):
        """Complete face validation with relaxed criteria"""
        if face_img is None or face_img.size == 0:
            return False, "Empty face image"
        
        if not FaceQualityValidator.has_sufficient_size(face_img):
            return False, "Face too small"
        
        # Skip strict checks for now - they were too restrictive
        # if not FaceQualityValidator.is_face_clear(face_img):
        #     return False, "Face too blurry"
        
        if not FaceQualityValidator.is_properly_aligned(face_img):
            return False, "Face poorly aligned"
        
        # Optional additional checks (commented out for relaxed validation)
        # if not FaceQualityValidator.has_good_contrast(face_img):
        #     return False, "Face has poor contrast"
        
        # if not FaceQualityValidator.is_well_lit(face_img):
        #     return False, "Face is poorly lit"
        
        return True, "Face valid"
    
    @staticmethod
    def get_face_quality_score(face_img):
        """Get overall quality score for face (0-100)"""
        if face_img is None or face_img.size == 0:
            return 0
        
        score = 0
        
        # Size score (0-25 points)
        h, w = face_img.shape[:2]
        min_dim = min(h, w)
        if min_dim >= 100:
            score += 25
        elif min_dim >= 60:
            score += 20
        elif min_dim >= 30:
            score += 15
        else:
            score += 5
        
        # Clarity score (0-25 points)
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var > 100:
                score += 25
            elif laplacian_var > 50:
                score += 20
            elif laplacian_var > 20:
                score += 15
            else:
                score += 10
        except:
            score += 15  # Default moderate score
        
        # Alignment score (0-25 points)
        try:
            aspect_ratio = w / h
            if 0.8 <= aspect_ratio <= 1.2:
                score += 25  # Perfect square-ish
            elif 0.6 <= aspect_ratio <= 1.6:
                score += 20  # Good
            elif 0.4 <= aspect_ratio <= 2.0:
                score += 15  # Acceptable
            else:
                score += 5   # Poor
        except:
            score += 15  # Default moderate score
        
        # Contrast score (0-25 points)
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            std_dev = np.std(gray)
            if std_dev > 60:
                score += 25
            elif std_dev > 40:
                score += 20
            elif std_dev > 20:
                score += 15
            else:
                score += 5
        except:
            score += 15  # Default moderate score
        
        return min(100, score)
    
    @staticmethod
    def enhance_face_for_analysis(face_img):
        """Apply basic enhancement to improve face for analysis"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel (brightness)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            l = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply slight Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Face enhancement error: {e}")
            return face_img


class FaceTracker:
    """Track faces across frames for better stability"""
    
    def __init__(self, max_distance=50, max_age=30):
        self.tracks = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_age = max_age
    
    def update(self, detections):
        """Update tracks with new detections"""
        # Calculate distances between existing tracks and new detections
        matched_tracks = {}
        unmatched_detections = list(detections)
        
        for track_id, track in self.tracks.items():
            if track['age'] > self.max_age:
                continue  # Skip old tracks
            
            best_match = None
            best_distance = float('inf')
            
            for i, detection in enumerate(unmatched_detections):
                x, y, w, h = detection[:4]
                center_x, center_y = x + w//2, y + h//2
                
                track_x, track_y = track['center']
                distance = np.sqrt((center_x - track_x)**2 + (center_y - track_y)**2)
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                detection = unmatched_detections[best_match]
                x, y, w, h = detection[:4]
                
                # Update track
                self.tracks[track_id].update({
                    'center': (x + w//2, y + h//2),
                    'bbox': (x, y, w, h),
                    'age': 0,
                    'confidence': detection[4] if len(detection) > 4 else 0.5
                })
                
                matched_tracks[track_id] = detection
                unmatched_detections.pop(best_match)
        
        # Age existing tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            x, y, w, h = detection[:4]
            self.tracks[self.next_id] = {
                'center': (x + w//2, y + h//2),
                'bbox': (x, y, w, h),
                'age': 0,
                'confidence': detection[4] if len(detection) > 4 else 0.5,
                'created_frame': self.next_id
            }
            matched_tracks[self.next_id] = detection
            self.next_id += 1
        
        return matched_tracks
    
    def get_stable_faces(self, min_track_length=3):
        """Get faces that have been tracked for a minimum number of frames"""
        stable_tracks = {}
        for track_id, track in self.tracks.items():
            if track['age'] < min_track_length:
                stable_tracks[track_id] = track
        return stable_tracks


class MultiScaleDetector:
    """Detect faces at multiple scales for better accuracy"""
    
    def __init__(self, base_detector):
        self.base_detector = base_detector
        self.scales = [1.0, 0.8, 1.2]  # Different scales to try
    
    def detect_faces(self, image):
        """Detect faces at multiple scales"""
        all_detections = []
        h, w = image.shape[:2]
        
        for scale in self.scales:
            if scale != 1.0:
                # Resize image
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(image, (new_w, new_h))
            else:
                resized = image
            
            # Detect faces
            faces = self.base_detector.detect_faces(resized)
            
            # Scale coordinates back to original size
            for face in faces:
                x, y, w_f, h_f, conf, method = face
                if scale != 1.0:
                    x = int(x / scale)
                    y = int(y / scale)
                    w_f = int(w_f / scale)
                    h_f = int(h_f / scale)
                
                all_detections.append((x, y, w_f, h_f, conf, f"{method}_scale_{scale}"))
        
        # Remove duplicates and return best detections
        return self.base_detector.filter_and_deduplicate_faces(all_detections)


# Factory function to create detector with best available methods
def create_face_detector(use_tracking=False, use_multiscale=False):
    """
    Factory function to create the best available face detector
    
    Args:
        use_tracking: Enable face tracking across frames
        use_multiscale: Enable multi-scale detection
    
    Returns:
        Configured face detector
    """
    base_detector = EnhancedFaceDetector()
    
    if use_multiscale:
        detector = MultiScaleDetector(base_detector)
    else:
        detector = base_detector
    
    if use_tracking:
        # Note: Tracking would need to be integrated into the main detection loop
        logger.info("Face tracking enabled")
    
    return detector