import os
import cv2


def normalize_labels(images_dir, labels_dir):
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            image_file = label_file.replace('.txt', '.png')  # Cambia '.png' si tus imágenes tienen otra extensión
            image_path = os.path.join(images_dir, image_file)
            if not os.path.exists(image_path):
                print(f"Imagen no encontrada para {label_file}")
                continue

            # Leer tamaño de la imagen
            img = cv2.imread(image_path)
            if img is None:
                print(f"No se pudo leer la imagen {image_path}")
                continue
            height, width, _ = img.shape

            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            normalized_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 9:  # 1 class_id + 8 coordenadas (x1,y1,x2,y2,x3,y3,x4,y4)
                    print(f"Formato incorrecto en {label_file}: {line}")
                    continue
                class_id = parts[0]
                coords = list(map(float, parts[1:]))
                # Normalizar coordenadas
                normalized_coords = []
                for i in range(0, 8, 2):
                    x = coords[i] / width
                    y = coords[i + 1] / height
                    normalized_coords.extend([x, y])
                normalized_lines.append(f"{class_id} {' '.join(map(str, normalized_coords))}\n")

            # Sobrescribir el archivo con coordenadas normalizadas
            with open(label_path, 'w') as f:
                f.writelines(normalized_lines)
            print(f"Normalizado {label_file}")


if __name__ == "__main__":
    images_dir = '/home/owams/Desktop/lds/projects/yolo-detection/digital_digit_dataset/images'
    labels_dir = '/home/owams/Desktop/lds/projects/yolo-detection/digital_digit_dataset/labels'
    normalize_labels(images_dir, labels_dir)