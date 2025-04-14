import cv2
import numpy as np
import os
from glob import glob

# Настройки
input_folder = '1'
output_folder = '2'
black_threshold = 30  # Порог для чёрного цвета (0-30)
min_line_width = 1    # Минимальная ширина линии
max_line_width = 10   # Максимальная ширина линии
min_line_height = 0.35  # Минимальная высота линии (относительно высоты изображения)
threshold = 127       # Порог для бинаризации (белый фон)

# Создаем папку для результатов
os.makedirs(output_folder, exist_ok=True)

def count_line_intersections(line_segment, filtered_lines):
    """Считает сколько линий пересекает отрезок"""
    x1, y1, x2, y2 = line_segment
    intersections = 0
    
    for (lx, _, lw, _) in filtered_lines:
        line_x = lx + lw/2  # Считаем центр линии
        # Проверяем пересечение отрезка с вертикальной линией
        if min(x1, x2) <= line_x <= max(x1, x2):
            intersections += 1
            
    return intersections

def find_yellow_red_regions(img):
    """Поиск областей от жёлтого до красного"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Объединённая маска для жёлтого, оранжевого и красного
    yellow_orange_mask = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
    red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask |= cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    combined_mask = cv2.bitwise_or(yellow_orange_mask, red_mask)
    
    # Маскируем чёрные области
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    combined_mask &= black_mask
    
    # Улучшаем маску
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Контуры
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours if contours else None

# Получаем список изображений
image_paths = sorted(glob(os.path.join(input_folder, '*')))

for i, image_path in enumerate(image_paths, start=1):
    try:
        print(f"\nОбработка изображения {i}: {image_path}")
        
        # 1. Загрузка изображения
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        # 2. Обрезка по контурам цветных областей
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        color_mask = cv2.bitwise_not(white_mask)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        combined_mask = cv2.bitwise_and(color_mask, thresh)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Контуры не найдены, используется исходное изображение")
            cropped = img
        else:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            x = min(x, img.shape[1] - 1)
            y = min(y, img.shape[0] - 1)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            cropped = img[y:y + h, x:x + w]

        # 3. Поиск чёрных линий
        hsv_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        black_mask = cv2.inRange(hsv_cropped, np.array([0, 0, 0]), np.array([180, 255, black_threshold]))
        clean_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8), iterations=1)
        enhanced_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8), iterations=1)
        contours, _ = cv2.findContours(enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_lines = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            line_height_ratio = h / cropped.shape[0]
            if (min_line_width <= w <= max_line_width) and (line_height_ratio >= min_line_height):
                filtered_lines.append((x, y, w, h))

        # 4. Проверка наличия вертикальных линий на краях
        start_x = 0
        end_x = cropped.shape[1] - 1
        has_start_line = any(x == start_x for x, _, _, _ in filtered_lines)
        has_end_line = any(x == end_x for x, _, _, _ in filtered_lines)

        if not has_start_line:
            print(f"Добавлена линия на позиции x={start_x}")
            filtered_lines.append((start_x, 0, 1, cropped.shape[0]))
        if not has_end_line:
            print(f"Добавлена линия на позиции x={end_x}")
            filtered_lines.append((end_x, 0, 1, cropped.shape[0]))

        # Сортируем линии по координате x
        filtered_lines.sort(key=lambda line: line[0])

        # 5. Найти цветные области
        color_contours = find_yellow_red_regions(cropped)

        # 6. Пересечения линий с цветной областью
        if color_contours is None:
            print("Цветные области не найдены")
            result_code = 0
            largest_start_area = None
            largest_end_area = None
            main_contour = None
        else:
            # Выбираем основную область (самую большую)
            main_contour = max(color_contours, key=cv2.contourArea)
            color_x, color_y, color_w, color_h = cv2.boundingRect(main_contour)
            first_line_x = filtered_lines[0][0]
            last_line_x = filtered_lines[-1][0]

            # Все цветные области
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            yellow_orange_mask = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
            red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            red_mask |= cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
            combined_mask = cv2.bitwise_or(yellow_orange_mask, red_mask)
            black_mask = cv2.threshold(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)[1]
            combined_mask &= black_mask
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            color_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Поиск областей, пересекающих обе линии
            double_intersect_areas = []
            largest_start_area = None
            max_start_area = 0
            largest_end_area = None
            max_end_area = 0

            for contour in color_contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                intersect_first = (first_line_x >= x and first_line_x <= x + w)
                intersect_last = (last_line_x >= x and last_line_x <= x + w)

                if intersect_first and intersect_last:
                    double_intersect_areas.append((area, contour))

                if intersect_first and area > max_start_area:
                    max_start_area = area
                    largest_start_area = contour

                if intersect_last and area > max_end_area:
                    max_end_area = area
                    largest_end_area = contour

            # Если есть области, пересекающие обе линии - берём наибольшую
            if double_intersect_areas:
                max_area_contour = max(double_intersect_areas, key=lambda x: x[0])[1]
                main_contour = max_area_contour
                color_x, color_y, color_w, color_h = cv2.boundingRect(main_contour)
                print("Найдена область, пересекающая обе линии")

            # Проверка пересечения линий с основной цветной областью
            intersecting_lines = 0
            first_line_intersects = False
            last_line_intersects = False

            for (x, y, w, h) in filtered_lines:
                if (x >= color_x and x <= color_x + color_w) or \
                   (color_x >= x and color_x <= x + w):
                    intersecting_lines += 1

                    if x == first_line_x:
                        first_line_intersects = True
                    if x == last_line_x:
                        last_line_intersects = True

            total_lines = len(filtered_lines)
            print(f"Всего линий: {total_lines}, пересекающих область: {intersecting_lines}")

            # Определение результата с учетом минимального расстояния
            if intersecting_lines < total_lines:
                if not first_line_intersects and not last_line_intersects:
                    result_code = 3
                elif not first_line_intersects:
                    result_code = 1
                    if not double_intersect_areas and largest_start_area is not None:
                        # Ищем ближайшие точки с условием пересечения ≤1 линии
                        min_dist = float('inf')
                        best_pt1 = None
                        best_pt2 = None
                        
                        for pt1 in main_contour[:, 0]:
                            for pt2 in largest_start_area[:, 0]:
                                dist = np.linalg.norm(pt1 - pt2)
                                if dist < min_dist:
                                    line_segment = (pt1[0], pt1[1], pt2[0], pt2[1])
                                    if count_line_intersections(line_segment, filtered_lines) <= 1:
                                        min_dist = dist
                                        best_pt1 = pt1
                                        best_pt2 = pt2
                        
                        if best_pt1 is not None:
                            main_contour = np.concatenate((main_contour, largest_start_area))
                            color_x, color_y, color_w, color_h = cv2.boundingRect(main_contour)
                            print(f"Объединение с областью у первой линии, расстояние: {min_dist:.2f}")
                            
                elif not last_line_intersects:
                    result_code = 2
                    if not double_intersect_areas and largest_end_area is not None:
                        # Аналогично для последней линии
                        min_dist = float('inf')
                        best_pt1 = None
                        best_pt2 = None
                        
                        for pt1 in main_contour[:, 0]:
                            for pt2 in largest_end_area[:, 0]:
                                dist = np.linalg.norm(pt1 - pt2)
                                if dist < min_dist:
                                    line_segment = (pt1[0], pt1[1], pt2[0], pt2[1])
                                    if count_line_intersections(line_segment, filtered_lines) <= 1:
                                        min_dist = dist
                                        best_pt1 = pt1
                                        best_pt2 = pt2
                        
                        if best_pt1 is not None:
                            main_contour = np.concatenate((main_contour, largest_end_area))
                            color_x, color_y, color_w, color_h = cv2.boundingRect(main_contour)
                            print(f"Объединение с областью у последней линии, расстояние: {min_dist:.2f}")
                            
                else:
                    result_code = 4
            else:
                result_code = 0

        # Визуализируем результат
        result = cropped.copy()

        # ... (предыдущий код визуализации main_contour)

        if main_contour is not None:
            # 1. Получаем координаты вертикальных линий
            line_x_coords = [line[0] + line[2]//2 for line in filtered_lines]
            line_x_coords.sort()
            
            # 2. Рассчитываем середины между линиями
            mid_points = []
            for i in range(len(line_x_coords)-1):
                mid_x = (line_x_coords[i] + line_x_coords[i+1]) // 2
                mid_points.append(mid_x)
            
            # 3. Находим Y-координаты внутри main_contour (исключая зелёные зоны)
            trajectory_points = []
            for x in mid_points:
                for y in range(0, cropped.shape[0]):
                    if cv2.pointPolygonTest(main_contour, (x, y), False) >= 0:
                        pixel_color = cropped[y, x]
                        if not np.array_equal(pixel_color, [0, 255, 0]):  # Исключаем зелёный
                            trajectory_points.append((x, y))
                            break
                        
            # 4. Рисуем ТОНКУЮ ЗЕЛЁНУЮ линию траектории (толщина 1px)
            if len(trajectory_points) > 1:
                for i in range(len(trajectory_points)-1):
                    pt1 = trajectory_points[i]
                    pt2 = trajectory_points[i+1]
                    cv2.line(result, pt1, pt2, (0, 255, 0), 1)  # Зелёный цвет, толщина 1
            
            # 5. Рисуем красные точки (как в исходном коде)
            for (x, y) in trajectory_points:
                cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
        
        # ... (остальной код сохранения)

        # Рисуем вертикальные линии
        for (x, y, w, h) in filtered_lines:
            line_color = (0, 255, 0)
            if x == first_line_x:
                line_color = (0, 255, 255)
            elif x == last_line_x:
                line_color = (255, 0, 255)
            cv2.rectangle(result, (x, 0), (x + w, cropped.shape[0]), line_color, 2)
            if x == first_line_x:
                cv2.putText(result, "First", (x + 5, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)
            elif x == last_line_x:
                cv2.putText(result, "Last", (x - 30, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)

        # Сохраняем результат
        cv2.imwrite(os.path.join(output_folder, f"{i}_result.png"), result)
        print(f"Найдено вертикальных линий: {len(filtered_lines)}")

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {str(e)}")

print("\nОбработка завершена!")
