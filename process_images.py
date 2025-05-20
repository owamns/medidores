import cv2
import numpy as np
import os
from ultralytics import YOLO


class MeterProcessor:
    def __init__(self, crop_model_path, digital_model_path, analog_model_path, input_folder, scale_factor=4,
                 conf_threshold=0):
        self.crop_model = YOLO(crop_model_path)
        self.digital_model = YOLO(digital_model_path)
        self.analog_model = YOLO(analog_model_path)
        self.class_names = {0: 'analogico', 1: 'digital'}
        self.input_folder = input_folder
        self.scale_factor = scale_factor
        self.conf_threshold = conf_threshold
        self.digit_priority = {d: i for i, d in enumerate([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])}
        os.makedirs('predict/analogico/obb', exist_ok=True)
        os.makedirs('predict/analogico/pain', exist_ok=True)
        os.makedirs('predict/digital/obb', exist_ok=True)
        os.makedirs('predict/digital/pain', exist_ok=True)

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
            angle = 180 - np.degrees(np.arctan2(points[1][1] - points[2][1], points[1][0] - points[2][0]))
        else:
            angle = -np.degrees(np.arctan2(points[0][1] - points[1][1], points[0][0] - points[1][0]))
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
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect_order, dst)
        warped = cv2.warpPerspective(image, M, (width, height))
        angle = self.calculate_rotation_angle(box_points)
        if abs(angle) > 45:
            warped = self.rotate_image(warped, angle)
        return warped

    def select_largest_side(self, boxes):
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

    def preprocess_image(self, image):
        global resize_image
        if self.scale_factor > 1:
            resize_image = cv2.resize(image, None, fx=self.scale_factor, fy=self.scale_factor,
                                        interpolation=cv2.INTER_CUBIC)

        return resize_image

    def predict_and_plot(self, model, image, image_file, meter_type, show=False):
        results = model.predict(image)
        digits = []
        for box, cls, conf in zip(results[0].obb.xyxyxyxy, results[0].obb.cls, results[0].obb.conf):
            if conf.item() >= self.conf_threshold:
                x1 = box[0][0].item()
                digit = int(cls.item())
                digits.append((x1, digit, box, conf.item()))

        filtered_digits = []
        digits.sort(key=lambda x: x[0])
        i = 0
        while i < len(digits):
            current_x1, current_digit, current_box, current_conf = digits[i]
            similar_digits = [(current_x1, current_digit, current_box, current_conf)]
            j = i + 1
            while j < len(digits) and abs(digits[j][0] - current_x1) < 10:
                similar_digits.append(digits[j])
                j += 1
            if len(similar_digits) > 1:
                best_digit = max(similar_digits, key=lambda x: (x[3], -self.digit_priority[x[1]]))
                filtered_digits.append(best_digit)
            else:
                filtered_digits.append(similar_digits[0])
            i = j

        filtered_digits.sort(key=lambda x: x[0])
        number = ''.join(str(d[1]) for d in filtered_digits)

        if meter_type == 'digital' and len(filtered_digits) > 1:
            areas = [cv2.contourArea(np.array(d[2].cpu(), dtype=np.float32)) for d in filtered_digits]
            if len(areas) > 1:
                avg_area = np.mean(areas[:-1])
                last_area = areas[-1]
                if last_area < 0.9 * avg_area:
                    number = number[:-1] + '.' + number[-1]
                elif len(filtered_digits) >= 7:
                    number = number[:-1] + '.' + number[-1]

        plotted_image = results[0].plot()
        height, width, _ = plotted_image.shape
        extension_width = 200
        extended_image_obb = np.ones((height, width + extension_width, 3), dtype=np.uint8) * 255
        extended_image_obb[:, :width, :] = plotted_image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 0)
        thickness = 2
        text_position = (width + 10, height // 2)
        cv2.putText(extended_image_obb, number, text_position, font, font_scale, font_color, thickness)

        output_obb_path = os.path.join(f'predict/{meter_type}/obb', f"pred_{image_file}")
        cv2.imwrite(output_obb_path, extended_image_obb)

        extended_image_pain = np.ones((height, width + extension_width, 3), dtype=np.uint8) * 255
        extended_image_pain[:, :width, :] = image
        cv2.putText(extended_image_pain, number, text_position, font, font_scale, font_color, thickness)
        output_pain_path = os.path.join(f'predict/{meter_type}/pain', f"pred_{image_file}")
        cv2.imwrite(output_pain_path, extended_image_pain)

        if show:
            cv2.imshow(f'Dígitos detectados en {image_file}', extended_image_obb)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return number

    def process_images(self, show=False):
        results = []
        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"No se pudo leer la imagen: {image_path}")
                continue

            det = self.crop_model.predict(image)[0]
            if not hasattr(det, 'obb'):
                print(f"No se detectaron objetos en: {image_path}")
                continue

            all_boxes = det.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
            all_cls = det.obb.cls.cpu().numpy()
            best_box_idx = self.select_largest_side(all_boxes)

            if best_box_idx is not None:
                cls = int(all_cls[best_box_idx])
                meter_type = self.class_names[cls]
                box_points = all_boxes[best_box_idx].astype(np.float32)
                warped = self.crop_rotated_roi(image, box_points)
                warped = self.preprocess_image(warped)
                model = self.digital_model if meter_type == 'digital' else self.analog_model
                number = self.predict_and_plot(model, warped, image_file, meter_type, show)

                results.append({
                    'image': image_file,
                    'type': meter_type,
                    'detected_number': number
                })
            else:
                print(f"No se seleccionó ninguna caja en: {image_path}")

        return results


if __name__ == "__main__":
    crop_model_path = './model_recorte.pt'
    digital_model_path = './models/best01.pt'
    analog_model_path = './models/best.pt'
    input_folder = './images'
    processor = MeterProcessor(crop_model_path, digital_model_path, analog_model_path, input_folder, scale_factor=4)
    results = processor.process_images(show=False)