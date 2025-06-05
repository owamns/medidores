import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from typing import Union, Dict, List, Optional, Tuple

class MeterProcessor:
    def __init__(
        self,
        crop_model_path: str,
        digital_model_path: str,
        electronic_model_path: str,
        scale_factor: int = 4,
        conf_threshold: float = 0.0,
        device: str = 'auto'
    ):
        self.device = self._setup_device(device)
        self.is_gpu = self.device.startswith('cuda')
        torch.backends.cudnn.enabled = self.is_gpu
        torch.backends.cudnn.benchmark = self.is_gpu  # Optimize for GPU
        self.crop_model = YOLO(crop_model_path).to(self.device)
        self.digital_model = YOLO(digital_model_path).to(self.device)
        self.electronic_model = YOLO(electronic_model_path).to(self.device)
        self.class_names = {0: 'd', 1: 'e'}
        self.scale_factor = scale_factor
        self.conf_threshold = conf_threshold
        self.digit_priority = {d: i for i, d in enumerate(range(10))}
        print(f"MeterProcessor initialized with {'GPU' if self.is_gpu else 'CPU'} device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if self.is_gpu:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    def _setup_device(self, device_config: str) -> str:
        device_config = device_config.lower()
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print("Auto-selected device: GPU (CUDA)")
            else:
                device = 'cpu'
                print("Auto-selected device: CPU (CUDA not available)")
        elif device_config in ('gpu', 'cuda'):
            if torch.cuda.is_available():
                device = 'cuda'
                print("Forced device: GPU (CUDA)")
            else:
                raise RuntimeError("GPU requested but CUDA is not available")
        elif device_config == 'cpu':
            device = 'cpu'
            print("Forced device: CPU")
        elif device_config.isdigit():
            gpu_id = int(device_config)
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                device = f'cuda:{gpu_id}'
                print(f"Selected specific GPU device: {device}")
            else:
                raise RuntimeError(f"GPU device {gpu_id} is not available")
        else:
            raise ValueError(f"Invalid device configuration: {device_config}")
        return device

    def get_device_info(self) -> Dict[str, Union[str, bool, int]]:
        info = {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
        }
        if self.is_gpu:
            info.update({
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_current_device': torch.cuda.current_device(),
                'cuda_device_name': torch.cuda.get_device_name(),
                'cuda_memory_allocated': torch.cuda.memory_allocated(),
                'cuda_memory_reserved': torch.cuda.memory_reserved(),
            })
        return info

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, M, (new_w, new_h))

    def calculate_rotation_angle(self, box_points: np.ndarray) -> float:
        points = box_points.copy()
        points[:, 0] = -points[:, 0]
        first_dist = np.linalg.norm(points[0] - points[1])
        second_dist = np.linalg.norm(points[1] - points[2])
        if first_dist < second_dist:
            angle = 180 - np.degrees(np.arctan2(points[1][1] - points[2][1], points[1][0] - points[2][0]))
        else:
            angle = -np.degrees(np.arctan2(points[0][1] - points[1][1], points[0][0] - points[1][0]))
        return angle

    def crop_rotated_roi(self, image: np.ndarray, box_points: np.ndarray) -> np.ndarray:
        rect_order = np.zeros((4, 2), dtype=np.float32)
        s = box_points.sum(axis=1)
        rect_order[0] = box_points[np.argmin(s)]
        rect_order[2] = box_points[np.argmax(s)]
        diff = np.diff(box_points, axis=1)
        rect_order[1] = box_points[np.argmin(diff)]
        rect_order[3] = box_points[np.argmax(diff)]
        width = int(np.linalg.norm(rect_order[0] - rect_order[1]))
        height = int(np.linalg.norm(rect_order[1] - rect_order[2]))
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect_order, dst)
        warped = cv2.warpPerspective(image, M, (width, height))
        angle = self.calculate_rotation_angle(box_points)
        if abs(angle) > 45:
            warped = self.rotate_image(warped, angle)
        return warped

    def select_largest_side(self, boxes: np.ndarray) -> Optional[int]:
        best_length = 0
        best_idx = None
        for idx, box in enumerate(boxes):
            side1 = np.linalg.norm(box[1] - box[0])
            side2 = np.linalg.norm(box[2] - box[1])
            side3 = np.linalg.norm(box[3] - box[2])
            side4 = np.linalg.norm(box[0] - box[3])
            length = (side1 + side3) / 2
            width = (side2 + side4) / 2
            current_max = max(length, width)
            if current_max > best_length:
                best_length = current_max
                best_idx = idx
        return best_idx

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if self.scale_factor > 1:
            return cv2.resize(image, None, fx=self.scale_factor, fy=self.scale_factor,
                             interpolation=cv2.INTER_CUBIC)
        return image

    def calculate_overlap(self, box1: np.ndarray, box2: np.ndarray) -> float:
        if hasattr(box1, 'cpu'):
            box1 = box1.cpu().numpy()
        if hasattr(box2, 'cpu'):
            box2 = box2.cpu().numpy()
        if box1.shape[0] != 4 or box2.shape[0] != 4:
            return 0.0
        try:
            poly1 = box1.astype(np.float32)
            poly2 = box2.astype(np.float32)
            intersection_result, intersection_poly = cv2.intersectConvexConvex(poly1, poly2)
            if intersection_result <= 0 or intersection_poly is None:
                return 0.0
            if len(intersection_poly) < 3:
                return 0.0
            intersection_area = cv2.contourArea(intersection_poly)
            area1 = cv2.contourArea(poly1)
            area2 = cv2.contourArea(poly2)
            if area1 <= 0 or area2 <= 0:
                return 0.0
            return intersection_area / min(area1, area2)
        except Exception:
            return self.calculate_overlap_fallback(box1, box2)

    def calculate_overlap_fallback(self, box1: np.ndarray, box2: np.ndarray) -> float:
        center1 = np.mean(box1, axis=0)
        center2 = np.mean(box2, axis=0)
        distance = np.linalg.norm(center1 - center2)
        def get_avg_size(box: np.ndarray) -> float:
            side1 = np.linalg.norm(box[1] - box[0])
            side2 = np.linalg.norm(box[2] - box[1])
            side3 = np.linalg.norm(box[3] - box[2])
            side4 = np.linalg.norm(box[0] - box[3])
            width = (side1 + side3) / 2
            height = (side2 + side4) / 2
            return (width + height) / 2
        avg_size1 = get_avg_size(box1)
        avg_size2 = get_avg_size(box2)
        min_size = min(avg_size1, avg_size2)
        if distance < min_size * 0.5:
            return max(0, 1.0 - (distance / (min_size * 0.5)))
        return 0.0

    def filter_overlapping_digits(
        self, digits: List[Tuple[float, int, np.ndarray, float]], overlap_threshold: float = 0.3
    ) -> List[Tuple[float, int, np.ndarray, float]]:
        if len(digits) <= 1:
            return digits
        filtered_digits = []
        used_indices = set()
        for i, (x1_i, digit_i, box_i, conf_i) in enumerate(digits):
            if i in used_indices:
                continue
            overlapping_group = [(x1_i, digit_i, box_i, conf_i, i)]
            for j, (x1_j, digit_j, box_j, conf_j) in enumerate(digits):
                if j != i and j not in used_indices:
                    overlap = self.calculate_overlap(box_i, box_j)
                    if overlap > overlap_threshold:
                        overlapping_group.append((x1_j, digit_j, box_j, conf_j, j))
            if len(overlapping_group) > 1:
                best_digit = max(overlapping_group, key=lambda x: (x[3], -self.digit_priority[x[1]]))
                filtered_digits.append((best_digit[0], best_digit[1], best_digit[2], best_digit[3]))
                used_indices.update(item[4] for item in overlapping_group)
            else:
                filtered_digits.append((x1_i, digit_i, box_i, conf_i))
                used_indices.add(i)
        return filtered_digits

    def has_valid_digits(self, model: YOLO, image: np.ndarray) -> bool:
        try:
            results = model.predict(image, device=self.device)
            if not hasattr(results[0], 'obb') or results[0].obb is None:
                return False
            return any(conf.item() >= self.conf_threshold for conf in results[0].obb.conf)
        except Exception:
            return False

    def predict_digits(self, model: YOLO, image: np.ndarray, meter_type: str) -> Optional[Union[float, str]]:
        results = model.predict(image, device=self.device)
        digits = []
        if not hasattr(results[0], 'obb') or results[0].obb is None:
            return None
        for box, cls, conf in zip(results[0].obb.xyxyxyxy, results[0].obb.cls, results[0].obb.conf):
            if conf.item() >= self.conf_threshold:
                x1 = box[0][0].item()
                digit = int(cls.item())
                digits.append((x1, digit, box, conf.item()))
        if not digits:
            return None
        digits.sort(key=lambda x: x[0])
        filtered_digits = self.filter_overlapping_digits(digits, overlap_threshold=0.5)
        final_digits = []
        i = 0
        while i < len(filtered_digits):
            current_x1, current_digit, current_box, current_conf = filtered_digits[i]
            similar_digits = [(current_x1, current_digit, current_box, current_conf)]
            j = i + 1
            while j < len(filtered_digits) and abs(filtered_digits[j][0] - current_x1) < 5:
                similar_digits.append(filtered_digits[j])
                j += 1
            best_digit = max(similar_digits, key=lambda x: (x[3], -self.digit_priority[x[1]]))
            final_digits.append(best_digit)
            i = j
        final_digits.sort(key=lambda x: x[0])
        if not final_digits:
            return None
        number = ''.join(str(d[1]) for d in final_digits)
        if meter_type == 'e' and len(final_digits) >= 6:
            number = number[:-1] + '.' + number[-1]
        elif meter_type == 'd' and len(final_digits) > 1:
            areas = [cv2.contourArea(d[2].cpu().numpy().astype(np.float32)) for d in final_digits]
            if len(areas) > 1:
                avg_area = np.mean(areas[:-1])
                last_area = areas[-1]
                if last_area < 0.9 * avg_area:
                    number = number[:-1] + '.' + number[-1]
                elif len(final_digits) == 8:
                    number = number[:-2] + '.' + number[-2:]
                elif len(final_digits) == 7:
                    number = number[:-1] + '.' + number[-1]
        try:
            return float(number.lstrip('0') or '0')
        except ValueError:
            return number

    def try_both_models(self, warped: np.ndarray, initial_meter_type: str) -> Tuple[Union[float, str], str]:
        primary_model, secondary_model, secondary_type = (
            (self.digital_model, self.electronic_model, 'e')
            if initial_meter_type == 'd'
            else (self.electronic_model, self.digital_model, 'd')
        )
        result = self.predict_digits(primary_model, warped, initial_meter_type)
        if result is not None and result != 0:
            return result, initial_meter_type
        result = self.predict_digits(secondary_model, warped, secondary_type)
        if result is not None and result != 0:
            return result, secondary_type
        return 0, initial_meter_type

    def process_image(self, image: np.ndarray) -> Dict[str, Union[str, float, np.ndarray, None]]:
        det = self.crop_model.predict(image, device=self.device)[0]
        if not hasattr(det, 'obb'):
            return {'detected_number': 0, 'meter_type': None, 'error': f'No objects detected in image'}
        all_boxes = det.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        all_cls = det.obb.cls.cpu().numpy()
        best_box_idx = self.select_largest_side(all_boxes)
        if best_box_idx is not None:
            cls = int(all_cls[best_box_idx])
            initial_meter_type = self.class_names[cls]
            box_points = all_boxes[best_box_idx].astype(np.float32)
            warped = self.crop_rotated_roi(image, box_points)
            warped = self.preprocess_image(warped)
            number, final_meter_type = self.try_both_models(warped, initial_meter_type)
            return {
                'image': image,
                'meter_type': final_meter_type,
                'detected_number': number
            }
        return {'detected_number': 0, 'meter_type': None, 'error': 'No valid box selected'}

    def process_images_from_folder(self, input_folder: str) -> List[Dict[str, Union[str, float, None]]]:
        results = []
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            result = self.process_image(image)
            result['image'] = image_file
            results.append(result)
        return results

if __name__ == "__main__":
    crop_model_path = 'models/best-recorte.pt'
    digital_model_path = 'models/best-digital.pt'
    electronic_model_path = 'models/best-electronico.pt'
    processor = MeterProcessor(crop_model_path, digital_model_path, electronic_model_path, scale_factor=4)
    print(f"Processing images from folder './test' using {'GPU' if processor.is_gpu else 'CPU'}")
    print(processor.process_images_from_folder('./test'))