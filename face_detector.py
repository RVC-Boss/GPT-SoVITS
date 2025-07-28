"""
face_detector.py - Core Face Detection and Analysis
Exact same logic as your working code, just modularized
"""

import cv2
import numpy as np
from PIL import Image
import torch
import time
import threading
from queue import Queue
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class AgeGenderDetector:
    """Enhanced Age & Gender Detection System - EXACT SAME LOGIC AS YOUR WORKING CODE"""
    
    def __init__(self):
        self.face_results = {}
        self.face_encodings = {}
        self.person_counter = 0
        self.analysis_queue = Queue()
        self.running = True
        
        # Load models
        self.load_models()
        
        # Start analysis worker
        self.analysis_thread = threading.Thread(target=self.analysis_worker, daemon=True)
        self.analysis_thread.start()
        
        logger.info("✅ AgeGenderDetector initialized")
    
    def load_models(self):
        """Load AI models - EXACT SAME AS YOUR WORKING CODE"""
        try:
            # Load DeepFace
            from deepface import DeepFace
            self.deepface = DeepFace
            logger.info("✅ DeepFace loaded")
        except ImportError:
            logger.error("❌ DeepFace not available")
            self.deepface = None
        
        try:
            # Load HuggingFace age model
            from transformers import AutoImageProcessor, SiglipForImageClassification
            model_name = "prithivMLmods/facial-age-detection"
            self.age_model = SiglipForImageClassification.from_pretrained(model_name)
            self.age_processor = AutoImageProcessor.from_pretrained(model_name)
            logger.info("✅ HuggingFace age model loaded")
        except Exception as e:
            logger.error(f"❌ HuggingFace model error: {e}")
            self.age_model = None
            self.age_processor = None
        
        # Age labels
        self.id2label = {
            "0": "01-10", "1": "11-20", "2": "21-30", "3": "31-40",
            "4": "41-55", "5": "56-65", "6": "66-80", "7": "80+"
        }
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def analysis_worker(self):
        """Background analysis worker - EXACT SAME AS YOUR WORKING CODE"""
        while self.running:
            try:
                if not self.analysis_queue.empty():
                    task = self.analysis_queue.get(timeout=0.1)
                    if task is None:
                        break
                    
                    person_id = task['id']
                    face_img = task['image']
                    callback = task.get('callback')
                    
                    # Analyze
                    age, age_conf = self.analyze_age(face_img)
                    gender, gender_conf = self.analyze_gender(face_img)
                    
                    # Store results
                    current_time = time.time()
                    if person_id in self.face_results:
                        first_seen = self.face_results[person_id].get('first_seen', current_time)
                    else:
                        first_seen = current_time
                    
                    result = {
                        'age': age,
                        'age_conf': age_conf,
                        'gender': gender,
                        'gender_conf': gender_conf,
                        'timestamp': current_time,
                        'first_seen': first_seen
                    }
                    
                    self.face_results[person_id] = result
                    
                    # Call callback if provided
                    if callback:
                        callback(person_id, result)
                
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Analysis worker error: {e}")
                time.sleep(0.1)
    
    def analyze_age(self, face_img):
        """Analyze age using HuggingFace - EXACT SAME AS YOUR WORKING CODE"""
        if self.age_model is None or face_img.size == 0:
            return "Unknown", 0.0
        
        try:
            # Convert to PIL
            if len(face_img.shape) == 3:
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            else:
                face_pil = Image.fromarray(face_img).convert("RGB")
            
            # Process
            inputs = self.age_processor(images=face_pil, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.age_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
            
            # Get prediction
            max_idx = probs.index(max(probs))
            age_range = self.id2label[str(max_idx)]
            confidence = probs[max_idx] * 100
            
            return age_range, confidence
        except Exception as e:
            logger.error(f"Age analysis error: {e}")
            return "Unknown", 0.0
    
    def analyze_gender(self, face_img):
        """Analyze gender using DeepFace - EXACT SAME AS YOUR WORKING CODE"""
        if self.deepface is None or face_img.size == 0:
            return "Unknown", 0.0
        
        try:
            result = self.deepface.analyze(
                face_img, 
                actions=['gender'], 
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                analysis = result[0]
            else:
                analysis = result
            
            gender = analysis.get('dominant_gender', 'Unknown')
            gender_probs = analysis.get('gender', {})
            confidence = max(gender_probs.values()) if gender_probs else 0.0
            
            # Simplify gender
            if gender in ['Man', 'Male']:
                gender = 'Male'
            elif gender in ['Woman', 'Female']:
                gender = 'Female'
            
            return gender, confidence
        except Exception as e:
            logger.error(f"Gender analysis error: {e}")
            return "Unknown", 0.0
    
    def get_face_encoding(self, face_img):
        """Get face encoding for recognition - EXACT SAME AS YOUR WORKING CODE"""
        if self.deepface is None or face_img.size == 0:
            return None
        
        try:
            # Preprocess
            face_resized = cv2.resize(face_img, (160, 160))
            
            # Get embedding
            embedding = self.deepface.represent(
                face_resized, 
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(embedding, list) and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            elif isinstance(embedding, dict):
                return np.array(embedding['embedding'])
            return None
        except Exception as e:
            # Fallback encoding
            try:
                face_resized = cv2.resize(face_img, (64, 64))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256])
                return hist.flatten()
            except:
                return None
    
    def find_matching_person(self, face_img, threshold=0.4):
        """Find matching person - EXACT SAME AS YOUR WORKING CODE"""
        current_encoding = self.get_face_encoding(face_img)
        if current_encoding is None:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        for person_id, stored_encoding in self.face_encodings.items():
            try:
                # Cosine similarity
                similarity = np.dot(current_encoding, stored_encoding) / (
                    np.linalg.norm(current_encoding) * np.linalg.norm(stored_encoding)
                )
                
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_id
            except:
                continue
        
        return best_match, best_similarity if best_match else (None, 0)
    
    def register_new_person(self, face_img):
        """Register new person - EXACT SAME AS YOUR WORKING CODE"""
        encoding = self.get_face_encoding(face_img)
        if encoding is None:
            return None
        
        self.person_counter += 1
        person_id = f"person_{self.person_counter}"
        self.face_encodings[person_id] = encoding
        
        logger.info(f"👤 NEW PERSON: {person_id}")
        return person_id
    
    def identify_person(self, face_img):
        """Identify person (new or existing) - EXACT SAME AS YOUR WORKING CODE"""
        match_result = self.find_matching_person(face_img)
        
        if match_result[0]:
            person_id, similarity = match_result
            logger.info(f"👤 RECOGNIZED: {person_id} ({similarity:.3f})")
            return person_id, False
        else:
            person_id = self.register_new_person(face_img)
            return person_id, True
    
    def detect_faces(self, image):
        """Detect faces in image - EXACT SAME AS YOUR WORKING CODE"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        return faces
    
    def process_image(self, image, callback=None):
        """Process image and return results - EXACT SAME AS YOUR WORKING CODE"""
        faces = self.detect_faces(image)
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            face_img = image[y:y+h, x:x+w]
            person_id, is_new = self.identify_person(face_img)
            
            if person_id:
                # Get existing result or create placeholder
                result = self.face_results.get(person_id, {
                    'age': 'Analyzing...',
                    'age_conf': 0,
                    'gender': 'Analyzing...',
                    'gender_conf': 0,
                    'timestamp': time.time(),
                    'first_seen': time.time()
                })
                
                # Add to analysis queue
                task = {
                    'id': person_id,
                    'image': face_img,
                    'callback': callback
                }
                self.analysis_queue.put(task)
                
                # Determine status
                current_time = time.time()
                first_seen = result.get('first_seen', current_time)
                time_known = current_time - first_seen
                
                if time_known < 3:
                    status = "NEW"
                elif time_known < 60:
                    status = "CURRENT"
                else:
                    status = "RETURNING"
                
                # Convert age to approximate number
                age_display = result['age']
                if result['age'] in self.id2label.values():
                    age_map = {
                        "01-10": "~6 years", "11-20": "~16 years", "21-30": "~25 years",
                        "31-40": "~35 years", "41-55": "~48 years", "56-65": "~60 years",
                        "66-80": "~73 years", "80+": "~85 years"
                    }
                    age_display = age_map.get(result['age'], result['age'])
                
                results.append({
                    'person_id': person_id,
                    'status': status,
                    'age': age_display,
                    'age_confidence': result['age_conf'],
                    'gender': result['gender'],
                    'gender_confidence': result['gender_conf'],
                    'face_coordinates': [int(x), int(y), int(w), int(h)],
                    'is_new': is_new
                })
        
        return results
    
    def cleanup_old_results(self):
        """Cleanup old results - EXACT SAME AS YOUR WORKING CODE"""
        current_time = time.time()
        old_persons = [
            pid for pid, result in self.face_results.items()
            if current_time - result.get('timestamp', 0) > 300  # 5 minutes
        ]
        
        for person_id in old_persons:
            self.face_results.pop(person_id, None)
            self.face_encodings.pop(person_id, None)
            logger.info(f"🗑️ REMOVED: {person_id}")
    
    def __del__(self):
        """Cleanup when detector is destroyed"""
        self.running = False
        if hasattr(self, 'analysis_thread'):
            self.analysis_thread.join(timeout=1.0)