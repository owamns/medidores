import cv2
import numpy as np
import os
import uuid
from ultralytics import YOLO

class MeterCropper:
    def __init__(self, model_path, input_folder):
        self.detection_model = YOLO(model_path)
        self.class_names = {0: 'analogico', 1: 'digital'}
        self.input_folder = input_folder
        os.makedirs('cropped/analogico', exist_ok=True)
        os.makedirs('cropped/digital', exist_ok=True)

    def rotate_image(self, image, angle):
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

    def calculate_rotation_angle(self, box_points):
        points = box_points.copy()
        points[:, 0] = -points[:, 0]

        first_dist = np.linalg.norm(points[0] - points[1])
        second_dist = np.linalg.norm(points[1] - points[2])

        if first_dist < second_dist:
            angle = 180 - np.degrees(
                np.arctan2(points[1][1] - points[2][1], points[1][0] - points[2][0]))
        else:
            angle = -np.degrees(
                np.arctan2(points[0][1] - points[1][1], points[0][0] - points[1][0]))

        return angle

    def crop_rotated_roi(self, image, box_points):
        rect_order = np.zeros((4, 2), dtype=np.float32)
        s = box_points.sum(axis=1)
        rect_order[0] = box_points[np.argmin(s)]
        rect_order[2] = box_points[np.argmax(s)]
        diff = np.diff(box_points, axis=1)
        rect_order[1] = box_points[np.argmin(diff)]
        rect_order[3] = box_points[np.argmax(diff)]

        width = int(np.linalg.norm(rect_order[0] - rect_order[1]))
        height = int(np.linalg.norm(rect_order[1] - rect_order[2]))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect_order, dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        angle = self.calculate_rotation_angle(box_points)
        if abs(angle) > 45:
            warped = self.rotate_image(warped, angle)
        return warped

    def select_largest_side(self, boxes):
        """Selecciona la caja con el lado más largo (ideal para medidores rectangulares grandes)."""
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

    def process_images(self):
        results = []
        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"No se pudo leer la imagen: {image_path}")
                continue

            det = self.detection_model.predict(image)[0]

            if not hasattr(det, 'obb'):
                print(f"No se detectaron objetos en: {image_path}")
                continue

            all_boxes = det.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)  # Formato (N,4,2)
            all_cls = det.obb.cls.cpu().numpy()

            best_box_idx = self.select_largest_side(all_boxes)

            if best_box_idx is not None:
                cls = int(all_cls[best_box_idx])
                meter_type = self.class_names[cls]
                box_points = all_boxes[best_box_idx].astype(np.float32)

                # Obtener la imagen recortada
                warped = self.crop_rotated_roi(image, box_points)

                # Guardar la imagen recortada en la carpeta correspondiente
                cropped_filename = f"cropped/{meter_type}/{meter_type}_{str(uuid.uuid4())[:8]}.png"
                cv2.imwrite(cropped_filename, warped)

                results.append({
                    'image': image_file,
                    'type': meter_type,
                    'cropped_image': cropped_filename
                })
            else:
                print(f"No se seleccionó ninguna caja en: {image_path}")

        return results

if __name__ == "__main__":
    model_path = './model_recorte.pt'
    input_folder = './images'
    cropper = MeterCropper(model_path, input_folder)
    results = cropper.process_images()
    for res in results:
        print(f"Imagen: {res['image']} - Tipo: {res['type']} - Guardada en: {res['cropped_image']}")