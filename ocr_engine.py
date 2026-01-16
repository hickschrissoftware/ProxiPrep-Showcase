"""
OCR engine module for ProxiPrep.

This module wraps Google Cloud Vision API logic for extracting text
from inventory and prep documents, with image preprocessing for
improved handwriting recognition.
"""

import io
import os
from typing import Optional

from google.cloud import vision
from google.cloud.vision_v1 import types

# Optional: PIL for image preprocessing
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not installed. Image preprocessing disabled.")

# Optional: OpenCV for advanced image preprocessing (better for handwriting)
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) not installed. Advanced preprocessing disabled.")


class OCREngine:
    """
    Wrapper around the Google Cloud Vision API client.

    Provides high-level methods for initializing credentials and extracting
    text from image inputs using document text detection, with optional
    image preprocessing to improve OCR accuracy on handwritten text.
    """

    def __init__(self, credentials_path: str = 'credentials.json', preprocess: bool = True) -> None:
        """
        Initialize the OCR engine and load credentials.

        Args:
            credentials_path: Path to the Google Cloud service account JSON file.
                            Defaults to 'credentials.json' in the project root.
            preprocess: Whether to apply image preprocessing for better OCR results.
                       Defaults to True.

        Raises:
            FileNotFoundError: If the credentials file does not exist.
            RuntimeError: If the Vision API client cannot be initialized.
        """
        self.credentials_path = credentials_path
        self.preprocess = preprocess and PIL_AVAILABLE
        self.client: Optional[vision.ImageAnnotatorClient] = None
        self._load_credentials()
        self._initialize_client()

    def _load_credentials(self) -> None:
        """
        Load Google Cloud credentials from the JSON file.

        Sets the GOOGLE_APPLICATION_CREDENTIALS environment variable.

        Raises:
            FileNotFoundError: If the credentials file does not exist.
        """
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found: {self.credentials_path}. "
                "Please ensure the Google Cloud service account JSON file exists."
            )

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
        print(f"Loaded credentials from: {self.credentials_path}")

    def _initialize_client(self) -> None:
        """
        Initialize the Google Cloud Vision API client.

        Raises:
            RuntimeError: If the client cannot be initialized.
        """
        try:
            self.client = vision.ImageAnnotatorClient()
            print("Google Cloud Vision API client initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Vision API client: {e}")

    def enhance_image(self, image_path: str, mode: str = 'light') -> bytes:
        """
        Enhance an image for optimal OCR accuracy on handwritten text.

        IMPORTANT: Google Vision's AI is trained on REAL photos, not binary images.
        Heavy thresholding can HURT accuracy by destroying information.

        Modes (ordered from least to most processing):
        - 'none': Send original image (Google's AI is very good!)
        - 'light': Gentle contrast boost + denoise (RECOMMENDED - preserves detail)
        - 'medium': Stronger contrast + sharpening (for faded text)
        - 'heavy': Full thresholding pipeline (only for extremely poor images)
        - 'whiteboard': Optimized for colored markers on white backgrounds
                        Uses color channel extraction (saturation + inverted blue)
                        to isolate ink before processing. Best for blue/red markers.
        - 'auto': Tries 'light' first; if image is very dark/faded, uses 'medium'

        Also includes:
        - Auto-upscaling small images (OCR needs ~300 DPI)
        - Deskewing (straightens rotated text)

        Args:
            image_path: Path to the image file to enhance.
            mode: Enhancement mode - 'none', 'light', 'medium', 'heavy', 'whiteboard', or 'auto'.

        Returns:
            Enhanced image as bytes in PNG format.
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) required. Install: pip install opencv-python")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # ===== MODE: RAW - Absolute zero processing =====
            # Send the original file bytes directly to Google Vision
            # This bypasses ALL OpenCV processing which might corrupt the image
            if mode == 'raw':
                with open(image_path, 'rb') as f:
                    raw_bytes = f.read()
                print("[Enhance] Mode 'raw': Sending original file directly (zero processing)")
                return raw_bytes

            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not decode image: {image_path}")

            h, w = img.shape[:2]
            print(f"[Enhance] Loaded: {image_path} ({w}x{h} pixels)")
            print(f"[Enhance] Mode: {mode}")

            # ===== MODE: NONE - Minimal processing (just re-encode) =====
            # Skip upscaling and deskewing, just re-encode as PNG
            if mode == 'none':
                success, encoded = cv2.imencode('.png', img)
                if success:
                    print("[Enhance] Mode 'none': Returning re-encoded image (no enhancements)")
                    return encoded.tobytes()

            # ===== STEP 0: UPSCALE SMALL IMAGES =====
            # WHY: OCR works best at ~300 DPI. Small phone photos often have
            # text that's too small for the AI to read clearly.
            # Rule: If width < 1500px, double the size.
            min_width = 1500
            if w < min_width:
                scale = min_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                # INTER_CUBIC gives smoother upscaling than default
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                print(f"[Enhance] Upscaled to {new_w}x{new_h} (small images hurt OCR)")

            # ===== STEP 1: DESKEW (STRAIGHTEN ROTATED TEXT) =====
            # WHY: Even a 5-degree rotation hurts OCR accuracy significantly.
            # This detects the angle and straightens the image.
            # SKIP for 'light' and 'whiteboard' modes to avoid potential corruption
            if mode not in ('light', 'whiteboard'):
                img = self._deskew(img)

            # For whiteboard mode, skip grayscale conversion (we need color channels)
            if mode == 'whiteboard':
                # Whiteboard processing happens in the mode section below
                # using the color image directly
                gray = None
            else:
                # Convert to grayscale for processing
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ===== MODE: AUTO - Analyze image and pick best mode =====
            if mode == 'auto':
                # Check image characteristics to pick the best mode
                mean_brightness = np.mean(gray)
                std_brightness = np.std(gray)
                print(f"[Enhance] Image analysis: brightness={mean_brightness:.1f}, contrast={std_brightness:.1f}")

                if std_brightness < 30:
                    # Very low contrast - needs stronger enhancement
                    mode = 'medium'
                    print("[Enhance] Auto-selected 'medium' (low contrast detected)")
                else:
                    mode = 'light'
                    print("[Enhance] Auto-selected 'light' (image looks reasonable)")

            # ===== MODE: LIGHT (Recommended!) =====
            # Gentle enhancement that PRESERVES the grayscale information
            # Google's AI can use this info to distinguish similar letters
            if mode == 'light':
                # Gentle CLAHE - boosts contrast without destroying detail
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # Light denoising - removes grain but keeps text sharp
                denoised = cv2.fastNlMeansDenoising(enhanced, h=5)

                # Slight sharpening using unsharp mask
                blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
                sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

                final = sharpened
                print("[Enhance] Applied light enhancement (contrast + denoise + sharpen)")

            # ===== MODE: MEDIUM =====
            # Stronger enhancement for faded/light text
            elif mode == 'medium':
                # Stronger CLAHE
                clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # Denoise
                denoised = cv2.fastNlMeansDenoising(enhanced, h=8)

                # Gamma correction - brightens dark areas, darkens light areas
                # This helps faint ink stand out more
                gamma = 0.7  # < 1 brightens midtones
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255
                                  for i in np.arange(0, 256)]).astype("uint8")
                gamma_corrected = cv2.LUT(denoised, table)

                # Strong sharpening
                blurred = cv2.GaussianBlur(gamma_corrected, (0, 0), 3)
                sharpened = cv2.addWeighted(gamma_corrected, 2.0, blurred, -1.0, 0)

                final = sharpened
                print("[Enhance] Applied medium enhancement (strong contrast + gamma + sharpen)")

            # ===== MODE: HEAVY =====
            # Full thresholding - only for extremely poor quality images
            elif mode == 'heavy':
                # Strong CLAHE
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # Bilateral filter to preserve edges
                denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

                # Adaptive thresholding
                thresh = cv2.adaptiveThreshold(
                    denoised, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    15, 4
                )

                # Light cleanup
                kernel = np.ones((2, 2), np.uint8)
                final = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                print("[Enhance] Applied heavy enhancement (thresholding)")

            # ===== MODE: WHITEBOARD =====
            # Optimized for colored markers on white background
            # Exploits color information (saturation, blue channel) before grayscale
            elif mode == 'whiteboard':
                # Step 1: Light denoise preserving edges
                denoised_color = cv2.bilateralFilter(img, 9, 75, 75)
                
                # Step 2: Remove glare (specular highlights)
                hsv = cv2.cvtColor(denoised_color, cv2.COLOR_BGR2HSV)
                h_ch, s_ch, v_ch = cv2.split(hsv)
                
                # Detect glare: high brightness + low saturation
                glare_mask = ((v_ch > 240) & (s_ch < 30)).astype(np.uint8) * 255
                if np.sum(glare_mask) > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
                    denoised_color = cv2.inpaint(denoised_color, glare_mask, 5, cv2.INPAINT_TELEA)
                    hsv = cv2.cvtColor(denoised_color, cv2.COLOR_BGR2HSV)
                    h_ch, s_ch, v_ch = cv2.split(hsv)
                    print(f"[Enhance] Removed glare from {np.sum(glare_mask > 0)} pixels")
                
                # Step 3: Extract ink using saturation (colored ink = high saturation)
                sat_boosted = cv2.convertScaleAbs(s_ch, alpha=2.0, beta=0)
                
                # Step 4: Extract inverted blue channel (blue markers appear dark in blue)
                b_ch, g_ch, r_ch = cv2.split(denoised_color)
                blue_inverted = cv2.subtract(np.uint8(255), b_ch)
                
                # Step 5: Combine channels (max of saturation and inverted blue)
                combined = cv2.max(sat_boosted, blue_inverted)
                
                # Step 6: CLAHE enhancement
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(combined)
                
                # Step 7: Adaptive thresholding
                final = cv2.adaptiveThreshold(
                    enhanced, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    25, 5
                )
                
                # Step 8: Morphological cleanup
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)
                final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
                
                print("[Enhance] Applied whiteboard enhancement (color channel extraction)")

            else:
                raise ValueError(f"Unknown mode: {mode}")

            success, encoded = cv2.imencode('.png', final)
            if not success:
                raise RuntimeError("Failed to encode image")

            print("[Enhance] Enhancement complete!")
            return encoded.tobytes()

        except Exception as e:
            raise RuntimeError(f"Image enhancement failed: {e}")

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Automatically detect and correct image rotation (skew).

        WHY: Even small rotations (5-10 degrees) significantly hurt OCR.
        This finds lines of text and calculates the rotation angle,
        then straightens the image.

        Args:
            img: Input image (BGR color).

        Returns:
            Deskewed image.
        """
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=100,
                minLineLength=100,
                maxLineGap=10
            )

            if lines is None or len(lines) == 0:
                print("[Deskew] No lines detected, skipping")
                return img

            # Calculate angles of all detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines (text lines)
                if -45 < angle < 45:
                    angles.append(angle)

            if not angles:
                print("[Deskew] No horizontal lines found, skipping")
                return img

            # Use median angle to avoid outliers
            median_angle = np.median(angles)

            # Only correct if angle is significant but not too extreme
            if abs(median_angle) < 0.5:
                print(f"[Deskew] Angle {median_angle:.2f}° is negligible, skipping")
                return img
            if abs(median_angle) > 15:
                print(f"[Deskew] Angle {median_angle:.2f}° is too extreme, skipping")
                return img

            # Rotate to correct the skew
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                img, rotation_matrix, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            print(f"[Deskew] Corrected rotation: {median_angle:.2f}°")
            return rotated

        except Exception as e:
            print(f"[Deskew] Failed ({e}), using original")
            return img

    def scan_image(self, image_path: str, enhanced_content: bytes = None) -> str:
        """
        Send image to Google Cloud Vision for document text detection.

        Uses the document_text_detection method which is specifically
        optimized for dense text documents AND handwriting recognition.

        Args:
            image_path: Path to the original image file (used for logging).
            enhanced_content: Optional pre-enhanced image bytes from enhance_image().
                            If provided, uses this instead of reading from disk.
                            If None, reads the original image from image_path.

        Returns:
            The full_text_annotation.text string from the Vision API response.

        Raises:
            FileNotFoundError: If image_path doesn't exist and no enhanced_content provided.
            RuntimeError: If the API call fails.
        """
        try:
            # Use enhanced content if provided, otherwise read from file
            if enhanced_content is not None:
                content = enhanced_content
                print(f"[OCR] Using pre-enhanced image content")
            else:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                with open(image_path, 'rb') as image_file:
                    content = image_file.read()
                print(f"[OCR] Loaded original image from: {image_path}")

            # Create Vision API image object
            image = types.Image(content=content)

            # ===== IMAGE CONTEXT FOR HANDWRITING =====
            # WHY: This tells Google Vision to use its handwriting-specific AI model.
            # The handwriting model is trained on:
            #   - Cursive and connected letters
            #   - Inconsistent letter sizes and spacing
            #   - Tilted or slanted writing
            #   - Personal handwriting "quirks"
            #
            # language_hints=['en'] tells the AI to expect English characters,
            # which helps it distinguish similar-looking letters (like 'l' vs '1').
            image_context = types.ImageContext(
                text_detection_params=types.TextDetectionParams(
                    enable_text_detection_confidence_score=True
                ),
                language_hints=['en']
            )

            # ===== DOCUMENT_TEXT_DETECTION ENDPOINT =====
            # WHY: Google has two OCR endpoints:
            #   1. text_detection - for signs, labels, short text
            #   2. document_text_detection - for pages of text, forms, HANDWRITING
            #
            # document_text_detection understands document STRUCTURE:
            #   - Pages, blocks, paragraphs, words, symbols
            #   - Reading order (top-to-bottom, left-to-right)
            #   - Line breaks and spacing
            #
            # This is critical for your inventory lists where order matters!
            print("[OCR] Sending to Google Cloud Vision (document_text_detection)...")
            print(f"[OCR] Image size: {len(content)} bytes")
            
            response = self.client.document_text_detection(
                image=image,
                image_context=image_context
            )

            # Check for API errors
            if response.error.message:
                raise RuntimeError(f"Vision API error: {response.error.message}")

            # Extract full text from the response
            full_text = ""
            if response.full_text_annotation:
                full_text = response.full_text_annotation.text
            
            # DEBUG: Show ALL detected text blocks
            print(f"[OCR] === DEBUG: All detected text ===")
            if response.text_annotations:
                for i, annotation in enumerate(response.text_annotations[:10]):  # First 10
                    desc = annotation.description.replace('\n', '\\n')
                    print(f"[OCR]   [{i}]: {repr(desc[:80])}")
            else:
                print("[OCR]   No text_annotations found!")
            
            # Also show full_text_annotation blocks
            if response.full_text_annotation:
                print(f"[OCR] === Full text annotation ===")
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        block_text = ""
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                word_text = "".join([symbol.text for symbol in word.symbols])
                                block_text += word_text + " "
                        print(f"[OCR]   Block: {repr(block_text.strip()[:80])}")
            print(f"[OCR] ================================")

            print(f"[OCR] Successfully extracted {len(full_text)} characters")
            return full_text

        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to scan image '{image_path}': {e}")

    def smart_direction_scan(self, image_path: str) -> str:
        """
        Intelligently scan image, adding rotations only if needed.
        
        Strategy:
        1. Scan at 0° first (fastest path for normal documents)
        2. Analyze quality: word count, valid word ratio, confidence
        3. If quality is poor, try additional rotations and combine
        
        This balances speed/cost with accuracy for rotated text.
        
        Args:
            image_path: Path to the image file to scan.
            
        Returns:
            Best combined OCR text.
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for smart direction scan")
            
        print(f"[SmartDir] Starting smart directional scan: {image_path}")
        
        # Load the original image
        with open(image_path, 'rb') as f:
            original_bytes = f.read()
        
        # Decode image with OpenCV
        nparr = np.frombuffer(original_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"Could not decode image: {image_path}")
        
        # ===== PHASE 1: Scan at 0° =====
        print(f"\n[SmartDir] Phase 1: Scanning at 0°...")
        result_0 = self._scan_with_confidence(original_bytes, 0)
        
        # Analyze quality of 0° results
        words_0 = result_0.get('words', [])
        valid_words_0 = [w for w in words_0 if self._is_likely_word(w['text'])]
        
        quality_score = self._calculate_quality_score(words_0, valid_words_0)
        
        print(f"[SmartDir] 0° results: {len(words_0)} words, {len(valid_words_0)} valid")
        print(f"[SmartDir] Quality score: {quality_score:.2f}/100")
        
        # ===== DECISION: Do we need more rotations? =====
        # More aggressive thresholds to catch rotated text
        MIN_VALID_WORDS = 2
        MIN_QUALITY_SCORE = 50  # Raised from 40
        MIN_VALID_RATIO = 0.6   # At least 60% of words must be valid
        
        # Calculate valid word ratio
        total_words = len(words_0)
        valid_ratio = len(valid_words_0) / total_words if total_words > 0 else 0
        
        # Check if we should stop at 0° or continue
        quality_ok = quality_score >= MIN_QUALITY_SCORE
        count_ok = len(valid_words_0) >= MIN_VALID_WORDS
        ratio_ok = valid_ratio >= MIN_VALID_RATIO
        
        print(f"[SmartDir] Valid ratio: {valid_ratio:.1%} (threshold: {MIN_VALID_RATIO:.0%})")
        
        if quality_ok and count_ok and ratio_ok:
            print(f"[SmartDir] ✓ Quality acceptable, using 0° results only (1 API call)")
            return result_0.get('full_text', '')
        
        # ===== PHASE 2: Need additional rotations =====
        print(f"\n[SmartDir] ✗ Quality below threshold, scanning additional rotations...")
        
        all_results = {0: result_0}
        all_words = {}
        
        # Collect words from 0° first
        for w in valid_words_0:
            word_lower = w['text'].lower().strip()
            all_words[word_lower] = (w.get('confidence', 0.5), 0, w['text'])
        
        # Scan remaining rotations
        for angle in [90, 180, 270]:
            print(f"[SmartDir] Scanning at {angle}°...")
            
            # Rotate image
            if angle == 90:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_img = cv2.rotate(img, cv2.ROTATE_180)
            else:  # 270
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Encode and scan
            success, encoded = cv2.imencode('.png', rotated_img)
            if not success:
                continue
            
            try:
                result = self._scan_with_confidence(encoded.tobytes(), angle)
                all_results[angle] = result
                
                # Collect valid words
                for w in result.get('words', []):
                    if self._is_likely_word(w['text']):
                        word_lower = w['text'].lower().strip()
                        conf = w.get('confidence', 0.5)
                        if word_lower not in all_words or conf > all_words[word_lower][0]:
                            all_words[word_lower] = (conf, angle, w['text'])
                            
            except Exception as e:
                print(f"[SmartDir] Error at {angle}°: {e}")
        
        # ===== PHASE 3: Combine results =====
        print(f"\n[SmartDir] === Results Summary ===")
        for angle, data in all_results.items():
            word_count = len(data.get('words', []))
            valid_count = len([w for w in data.get('words', []) if self._is_likely_word(w['text'])])
            print(f"  {angle:3}°: {word_count} total, {valid_count} valid")
        
        # Find the rotation with best quality
        best_angle = 0
        best_score = 0
        for angle, data in all_results.items():
            words = data.get('words', [])
            valid = [w for w in words if self._is_likely_word(w['text'])]
            score = self._calculate_quality_score(words, valid)
            if score > best_score:
                best_score = score
                best_angle = angle
        
        print(f"[SmartDir] Best rotation: {best_angle}° (score: {best_score:.2f})")
        
        # Use best rotation's text as base
        base_text = all_results[best_angle].get('full_text', '').strip()
        base_words = set()
        for line in base_text.split('\n'):
            for w in line.split():
                base_words.add(w.lower().strip('.,!?'))
        
        # Add unique words from other rotations
        additional = []
        for word_lower, (conf, angle, original) in all_words.items():
            if word_lower not in base_words and angle != best_angle:
                additional.append(original)
                print(f"  + Adding '{original}' from {angle}°")
        
        # Combine
        if additional:
            final_text = base_text + '\n' + '\n'.join(additional)
        else:
            final_text = base_text
        
        total_calls = len(all_results)
        print(f"\n[SmartDir] Complete: {total_calls} API calls used")
        
        return final_text
    
    def _calculate_quality_score(self, all_words: list, valid_words: list) -> float:
        """
        Calculate a quality score for OCR results.
        
        Score is based on:
        - Number of valid words (0-50 points)
        - Ratio of valid to total words (0-30 points)  
        - Average confidence (0-20 points)
        
        Args:
            all_words: List of all word dicts from OCR.
            valid_words: List of words that passed validation.
            
        Returns:
            Score from 0-100.
        """
        if not all_words:
            return 0
        
        # Points for word count (max 50 for 5+ valid words)
        word_points = min(len(valid_words) * 10, 50)
        
        # Points for valid ratio (max 30)
        valid_ratio = len(valid_words) / len(all_words) if all_words else 0
        ratio_points = valid_ratio * 30
        
        # Points for confidence (max 20)
        if valid_words:
            avg_conf = sum(w.get('confidence', 0.5) for w in valid_words) / len(valid_words)
            conf_points = avg_conf * 20
        else:
            conf_points = 0
        
        return word_points + ratio_points + conf_points

    def multi_direction_scan(self, image_path: str) -> str:
        """
        Scan image at multiple rotations to capture text in any direction.
        
        Rotates the image 0°, 90°, 180°, 270° and runs OCR on each.
        Uses a dictionary-based word validation to filter out garbage
        and deduplicate results.
        
        Args:
            image_path: Path to the image file to scan.
            
        Returns:
            Combined OCR text with duplicates removed.
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for multi-direction scan")
            
        print(f"[MultiDir] Starting multi-directional scan: {image_path}")
        
        # Load the original image
        with open(image_path, 'rb') as f:
            original_bytes = f.read()
        
        # Decode image with OpenCV
        nparr = np.frombuffer(original_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"Could not decode image: {image_path}")
        
        rotations = [0, 90, 180, 270]
        all_words = {}  # word -> (confidence, rotation)
        all_text_by_rotation = {}
        
        for angle in rotations:
            print(f"\n[MultiDir] Scanning at {angle}°...")
            
            # Rotate image
            if angle == 0:
                rotated_img = img
            elif angle == 90:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_img = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Encode rotated image
            success, encoded = cv2.imencode('.png', rotated_img)
            if not success:
                print(f"[MultiDir] Failed to encode {angle}° rotation")
                continue
            
            rotated_bytes = encoded.tobytes()
            
            # Run OCR
            try:
                text = self._scan_with_confidence(rotated_bytes, angle)
                all_text_by_rotation[angle] = text
                
                # Extract individual words with their info
                words = text.get('words', [])
                for word_info in words:
                    word = word_info['text'].lower().strip()
                    conf = word_info.get('confidence', 0.5)
                    
                    # Only keep words that look like real words (mostly letters)
                    if len(word) >= 2 and sum(c.isalpha() for c in word) >= len(word) * 0.6:
                        if word not in all_words or conf > all_words[word][0]:
                            all_words[word] = (conf, angle, word_info['text'])
                            
            except Exception as e:
                print(f"[MultiDir] Error at {angle}°: {e}")
        
        # Build the final result by combining best words
        print(f"\n[MultiDir] === Results Summary ===")
        for angle, data in all_text_by_rotation.items():
            word_count = len(data.get('words', []))
            print(f"  {angle:3}°: {word_count} words detected")
        
        # Filter words using quality heuristics
        validated_words = []
        for word_lower, (conf, angle, original_text) in all_words.items():
            # Basic quality filter
            is_valid = self._is_likely_word(original_text)
            if is_valid:
                validated_words.append({
                    'text': original_text,
                    'confidence': conf,
                    'angle': angle
                })
                print(f"  ✓ '{original_text}' (conf={conf:.2f}, {angle}°)")
            else:
                print(f"  ✗ '{original_text}' (rejected)")
        
        # Sort by confidence and build output
        validated_words.sort(key=lambda x: -x['confidence'])
        
        # Group by likely structure (sentences vs items)
        result_lines = []
        
        # First, add full text from 0° rotation as base (usually best for sentences)
        if 0 in all_text_by_rotation and all_text_by_rotation[0].get('full_text'):
            base_text = all_text_by_rotation[0]['full_text'].strip()
            if base_text:
                result_lines.append(base_text)
        
        # Then add unique words found in other rotations
        base_words = set()
        if result_lines:
            for line in result_lines:
                for w in line.split():
                    base_words.add(w.lower().strip('.,!?'))
        
        additional_words = []
        for w in validated_words:
            w_clean = w['text'].lower().strip('.,!?')
            if w_clean not in base_words and w['angle'] != 0:
                additional_words.append(w['text'])
                base_words.add(w_clean)
        
        if additional_words:
            result_lines.extend(additional_words)
        
        final_text = '\n'.join(result_lines)
        
        print(f"\n[MultiDir] Final combined text:")
        print(f"  {repr(final_text[:200])}")
        
        return final_text
    
    def _scan_with_confidence(self, image_bytes: bytes, angle: int) -> dict:
        """
        Scan image bytes and return words with confidence scores.
        
        Args:
            image_bytes: Image data as bytes.
            angle: Rotation angle (for logging).
            
        Returns:
            Dict with 'full_text' and 'words' (list of word dicts).
        """
        image = types.Image(content=image_bytes)
        
        image_context = types.ImageContext(
            text_detection_params=types.TextDetectionParams(
                enable_text_detection_confidence_score=True
            ),
            language_hints=['en']
        )
        
        response = self.client.document_text_detection(
            image=image,
            image_context=image_context
        )
        
        if response.error.message:
            raise RuntimeError(f"Vision API error: {response.error.message}")
        
        full_text = ""
        words = []
        
        if response.full_text_annotation:
            full_text = response.full_text_annotation.text
            
            # Extract words with confidence
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join([s.text for s in word.symbols])
                            # Get average symbol confidence
                            confidences = [s.confidence for s in word.symbols if hasattr(s, 'confidence')]
                            avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
                            
                            words.append({
                                'text': word_text,
                                'confidence': avg_conf
                            })
        
        print(f"[MultiDir] {angle}°: Found {len(words)} words")
        
        return {
            'full_text': full_text,
            'words': words
        }
    
    def _is_likely_word(self, text: str) -> bool:
        """
        Check if text is likely a real English word vs OCR garbage.
        
        Uses strict heuristics to filter out common OCR errors:
        - Minimum length requirements
        - Must be mostly letters
        - Must have vowels (for words 3+ chars)
        - No garbage characters
        - Checks against common English words for short words
        
        Args:
            text: The word to validate.
            
        Returns:
            True if likely a real word.
        """
        if not text:
            return False
        
        # Strip punctuation for analysis
        clean = text.strip('.,!?;:\'"()-')
        
        if len(clean) < 2:
            # Only accept single-char words that are valid English
            return clean.lower() in ('i', 'a')
        
        # Reject if contains garbage characters
        garbage_chars = set('$+=#@*&%^~`|\\/<>{}[]0123456789')
        if any(c in garbage_chars for c in clean):
            return False
        
        # Count character types
        letters = sum(c.isalpha() for c in clean)
        
        # Must be ALL letters for short words
        if len(clean) <= 3 and letters != len(clean):
            return False
        
        # Must be mostly letters for longer words
        if letters < len(clean) * 0.8:
            return False
        
        # Check for repeated characters (OCR artifact)
        for i in range(len(clean) - 2):
            if clean[i] == clean[i+1] == clean[i+2]:
                return False
        
        # Words 3+ chars must have vowels
        if len(clean) >= 3:
            vowels = sum(c.lower() in 'aeiou' for c in clean)
            if vowels == 0:
                return False
        
        # For short words (2-3 chars), verify against common words
        if len(clean) <= 3:
            common_short_words = {
                'a', 'i', 'an', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 
                'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to',
                'up', 'us', 'we', 'hi', 'ok', 'am', 'as', 'at', 'ax',
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                'his', 'how', 'its', 'let', 'may', 'new', 'now', 'old',
                'see', 'two', 'way', 'who', 'boy', 'did', 'get', 'put',
                'say', 'she', 'too', 'use', 'egg', 'add', 'buy', 'eat',
                'hot', 'mix', 'oil', 'pan', 'raw', 'soy', 'tea', 'ham'
            }
            if clean.lower() not in common_short_words:
                return False
        
        return True

    def smart_scan(self, image_path: str) -> str:
        """
        Intelligently scan an image using multiple enhancement profiles.

        This method eliminates manual tuning by:
        1. Trying multiple enhancement modes (auto, adaptive, aggressive)
        2. Running OCR on each enhanced version
        3. Returning the result with the MOST text extracted

        WHY THIS WORKS: Different images need different preprocessing.
        A photo taken in good lighting needs different settings than
        a photo taken in shadow. Instead of guessing, we try all
        approaches and let the results speak for themselves.

        Cost consideration: This uses 3x the API calls, but ensures
        maximum accuracy. For production, you might want to use
        just 'auto' mode for most images.

        Args:
            image_path: Path to the image file to scan.

        Returns:
            The best OCR result (highest confidence/most readable text).
        """
        print(f"[SmartScan] Starting intelligent multi-mode scan: {image_path}")

        # Try modes from least processing to most
        # Often, less processing = better results with Google's AI
        modes = ['none', 'light', 'medium', 'heavy']
        results = []

        for mode in modes:
            try:
                print(f"\n{'='*50}")
                print(f"[SmartScan] Testing mode: '{mode}'")
                print('='*50)

                enhanced = self.enhance_image(image_path, mode=mode)
                text = self.scan_image(image_path, enhanced_content=enhanced)

                # Calculate quality metrics
                text_clean = text.strip()
                char_count = len(text_clean)
                word_count = len(text_clean.split())

                # Count "readable" words (3+ chars, mostly letters)
                words = text_clean.split()
                readable_words = sum(1 for w in words
                                    if len(w) >= 2 and sum(c.isalpha() for c in w) >= len(w) * 0.5)

                # Count lines (useful for list detection)
                line_count = len([l for l in text_clean.split('\n') if l.strip()])

                results.append({
                    'mode': mode,
                    'text': text,
                    'chars': char_count,
                    'words': word_count,
                    'readable_words': readable_words,
                    'lines': line_count
                })

                print(f"[SmartScan] Results: {word_count} words, {readable_words} readable, {line_count} lines")

            except Exception as e:
                print(f"[SmartScan] Mode '{mode}' failed: {e}")

        if not results:
            raise RuntimeError("All enhancement modes failed")

        # Score each result - prioritize readable words and line structure
        # This helps distinguish real text from noise
        for r in results:
            # Score = readable words + bonus for having multiple lines
            r['score'] = r['readable_words'] + (r['lines'] * 0.5)

        # Pick best result
        best = max(results, key=lambda x: x['score'])

        print(f"\n{'='*50}")
        print(f"[SmartScan] WINNER: mode='{best['mode']}'")
        print(f"[SmartScan] {best['words']} words, {best['readable_words']} readable, {best['lines']} lines")
        print('='*50)

        # Show comparison
        print("\n[SmartScan] All results comparison:")
        for r in results:
            marker = "  <-- BEST" if r['mode'] == best['mode'] else ""
            print(f"  {r['mode']:8} | {r['readable_words']:3} readable words | {r['lines']:2} lines{marker}")

        return best['text']
