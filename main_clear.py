# TODO vylepšit thresholdování - očividně tam chybí občas důležité kořínky
from tqdm.contrib.concurrent import process_map
from multiprocessing import freeze_support
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans
from dataclasses import dataclass
from library import read_barcodes
from datetime import datetime
from glob import glob
import pandas as pd
import numpy as np
import itertools
import parse
import yaml
import math
import cv2
import os

np.seterr(divide='ignore') 


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

COMBINATIONS = list(itertools.product([-1, 0, 1], repeat=2))
COMBINATIONS.remove((0, 0))

KERNEL_CENTER = 10
KERNEL_COUNT = np.array((
        [1, 1,             1],
        [1, KERNEL_CENTER, 1],
        [1, 1,             1])
)

ROTATION = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE
}

MARGIN_CONV = 4
ROOT_THRESHOLD_LOW = 20
ROOT_THRESHOLD_HIGH = 30


with open("configs/clear.yaml") as f:
    config = yaml.safe_load(f)

@dataclass
class Image:
    date: datetime
    path: str
    barcode: str

    image = None
    process = None
    canny = None
    diff = None
    total_length = None
    total_area = None

    green_areas = []
    positions_x = []
    positions_y = []
    plant_length = []
    longest = []


dictionary: dict[str, list[Image]] = {}

# Create data structure
for i in glob(os.path.join(config["data"]["input"], "*")):
    s = parse.parse(config["data"]["filename_template"], os.path.splitext(i)[0]).named

    date = datetime.strptime(s["date"].replace("_1", ""), config["data"]["date_format"])

    dictionary.setdefault(s["group"], []).append(Image(date, i, s["group"].split("/")[-1]))

# Sort by date
for k in dictionary.keys():
    dictionary[k] = sorted(dictionary[k], key=lambda z: z.date)

os.makedirs(config["data"]["output"], exist_ok=True)

def test(values):
    df: list[dict] = []
    
    try:
        for value in values:
            # Load image
            original = cv2.imread(value.path)

            # Rotate the image if needed
            if config["rotation"]:
                original = cv2.rotate(original, ROTATION[config["rotation"]])

            # Use just red channel - there is no blue background in it
            # _, _, r = cv2.split(original)
            # _, thresh = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            thresh = cv2.inRange(cv2.cvtColor(original, cv2.COLOR_BGR2HSV), (70, 0, 0), (140, 255, 255))  # TODO const

            # Speed optimization
            thresh = cv2.erode(thresh, None, iterations=3)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Automatic cropping
            min_x, min_y, max_x, max_y = original.shape[1], original.shape[0], 0, 0
            for cnt in contours:
                if cv2.contourArea(cnt) > 100:
                    x, y, w, h = cv2.boundingRect(cnt)

                    min_x = min_x if x > min_x else x
                    min_y = min_y if y > min_y else y
                    max_x = max_x if x + w < max_x else x + w
                    max_y = max_y if y + h < max_y else y + h

            cropped = original[min_y:max_y, min_x:max_x]
            cropped = cropped[round(cropped.shape[0] * 0.1):round(cropped.shape[0] * 0.85)]  # TODO const

            # Find green parts
            green_threshold = cv2.inRange(cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV), (28, 166, 50), (42, 255, 200))  # TODO const
            green_threshold = cv2.erode(cv2.dilate(green_threshold, None, iterations=3), None, iterations=1)

            # Crop to green parts
            contours, _ = cv2.findContours(green_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            min_y = cropped.shape[0]

            positions_x = []
            positions_y = []
            areas = []

            origo = cropped.copy()

            for cnt in contours:
                if len(cnt) > 5 and cv2.contourArea(cnt) > 50:  # TODO const
                    x, y, w, h = cv2.boundingRect(cnt)

                    if y < min_y:
                        min_y = y

                    cv2.drawContours(cropped, [cnt], -1, (0, 0, 0), cv2.FILLED)

                    # Count green area
                    cnt[:, 0, 0] -= x
                    cnt[:, 0, 1] -= y

                    mask_cnt = np.zeros((h, w))
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, cv2.FILLED)

                    for n in range(1):
                        positions_x.append(x + w / 2)
                        positions_y.append(y + h / 2)
                        areas.append(cv2.countNonZero(mask_cnt) if n == 0 else 0)

            try:
                value.barcode_read = read_barcodes(cropped[:min_y])[0]
            except Exception as e:
                value.barcode_read = ""  # todo
                print("barcode reading error", e)

            cropped = cropped[min_y:]
            value.image = origo[min_y:].copy()

            y_median = np.median(positions_y)

            # Add prior centroids (in case there is no visible green part)
            #for i in np.linspace(0, cropped.shape[1], 8)[1:-1]:
            #    positions_x.append(i)
            #    positions_y.append(y_median - min_y)
            #    areas.append(0)  # It is just point, so area is 0
            #    cv2.circle(value.image, (round(i), round(y_median - min_y)), 15, (255, 255, 255), -1)

            # Find centroids of green parts using KMeans
            positions_x = np.array(positions_x).reshape(-1, 1)
            positions_y = np.array(positions_y).reshape(-1, 1)
            areas = np.array(areas).reshape(-1, 1)

            try:
                kmeans = KMeans(n_clusters=config["n_clusters"]).fit_predict(positions_x)
                positions_x_median = [int(np.around(np.median(positions_x[kmeans == i]))) for i in range(config["n_clusters"])]
                positions_y_median = [int(np.around(np.median(positions_y[kmeans == i]))) for i in range(config["n_clusters"])]
                areas_sum = [np.sum(areas[kmeans == i]) for i in range(config["n_clusters"])]
            except Exception as e:
                print("error - někde chybí 6. rostlina", e, values)
                return df

            # Sort centroids by x (left to right)
            positions_x_median_sorted = []
            positions_y_median_sorted = []
            areas_sum_sorted = []
            for x, y, area in sorted(zip(positions_x_median, positions_y_median, areas), key=lambda z: z[0]):
                positions_x_median_sorted.append(x)
                positions_y_median_sorted.append(y - min_y)
                areas_sum_sorted.append(area)

            value.positions_x = positions_x_median_sorted
            value.positions_y = positions_y_median_sorted
            value.green_areas = areas_sum

            # Remove background gradient
            img = cropped.astype(int) - cv2.medianBlur(cropped, 101).astype(int)
            img[img < 0] = 0
            img = cv2.medianBlur(img.astype(np.uint8), 5)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            value.process = img
            value.canny = cv2.Canny(img, 100, 200)

        # Register images
        for n in range(len(values) - 1):
            template = values[n].canny
            h, w = template.shape[:2]

            img_canny = values[n + 1].canny
            img_debug = values[n + 1].image
            img_process = values[n + 1].process

            mx = img_canny.shape[1] // MARGIN_CONV
            my = img_canny.shape[0] // MARGIN_CONV

            img_canny = cv2.copyMakeBorder(img_canny, my, my, mx, mx, cv2.BORDER_CONSTANT, value=0)
            img_debug = cv2.copyMakeBorder(img_debug, my, my, mx, mx, cv2.BORDER_CONSTANT, value=0)
            img_process = cv2.copyMakeBorder(img_process, my, my, mx, mx, cv2.BORDER_CONSTANT, value=0)

            # Apply template Matching
            left, top = cv2.minMaxLoc(cv2.matchTemplate(img_canny, template, cv2.TM_CCOEFF))[3]

            values[n + 1].image = img_debug[top:top + h, left:left + w]
            values[n + 1].process = img_process[top:top + h, left:left + w]
            values[n + 1].canny = img_canny[top:top + h, left:left + w]
            values[n + 1].positions_x = [x + mx - left for x in values[n + 1].positions_x]
            values[n + 1].positions_y = [y + my - top for y in values[n + 1].positions_y]

            values[n + 1].diff = values[n + 1].process.astype(int) - values[n].process.astype(int)
            values[n + 1].diff[values[n + 1].diff < 0] = 0
            values[n + 1].diff = values[n + 1].diff.astype(np.uint8)

        pos_x_median = []
        pos_y_median = []
        for i in range(config["n_clusters"]):
            pos_x_median.append([])
            pos_y_median.append([])
            for value_n, value in enumerate(values):
                pos_x_median[-1].append(value.positions_x[i])
                pos_y_median[-1].append(value.positions_y[i])

            pos_x_median[-1] = round(np.median(pos_x_median[-1]))
            pos_y_median[-1] = round(np.median(pos_y_median[-1]))

        # Process images
        for value_n, value in enumerate(values):
            for i in range(len(pos_x_median)):
                cv2.circle(value.image, (pos_x_median[i], pos_y_median[i]), 10, COLORS[i], 4)
                cv2.putText(value.image, str(i + 1), (pos_x_median[i] + 10, pos_y_median[i] - 10), cv2.FONT_HERSHEY_PLAIN, 2, COLORS[i], 2)

            # Threshold image
            thresh_low = cv2.threshold(value.process, ROOT_THRESHOLD_LOW, 255, cv2.THRESH_BINARY)[1]
            thresh_high = cv2.threshold(value.process, ROOT_THRESHOLD_HIGH, 255, cv2.THRESH_BINARY)[1]
            thresh = np.zeros_like(thresh_low)

            contours, hierarchy = cv2.findContours(thresh_low, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            to_draw = []
            ignore = []
            for cnt_n, (cnt, h) in enumerate(zip(contours, hierarchy[0])):
                temp_mask = np.zeros_like(thresh_low)
                cv2.drawContours(temp_mask, [cnt], 0, 255, cv2.FILLED)

                if h[3] == -1:
                    if len(cnt) > 15 and cv2.countNonZero(cv2.bitwise_and(thresh_high, temp_mask)) / cv2.countNonZero(temp_mask) > 0.2:
                        # Fit ellipse
                        # (x, y), (minor, major), angle = cv2.fitEllipse(cnt)
                        # if minor and major/minor >= 1.3:
                        to_draw.append(cnt)
                        # else:
                        #     ignore.append(cnt_n)
                    else:
                        ignore.append(cnt_n)

            for cnt_n, (cnt, h) in enumerate(zip(contours, hierarchy[0])):
                if h[3] != -1 and h[3] not in ignore:
                    to_draw.append(cnt)

            cv2.drawContours(thresh, to_draw, -1, 255, cv2.FILLED)

            # cv2.imshow("ori", value.image)
            # cv2.imshow("trh", thresh)
            # cv2.waitKey(0)

            # Crop borders caused by box
            margin_sides = round(config["margin"] * thresh.shape[1])
            thresh[:, 0:margin_sides] = 0
            thresh[:, -margin_sides:] = 0

            if value.diff is not None:
                thresh_new = cv2.threshold(value.diff, ROOT_THRESHOLD_LOW, 255, cv2.THRESH_BINARY)[1]
                thresh_new[:, 0:margin_sides] = 0
                thresh_new[:, -margin_sides:] = 0
                values[value_n].new_area = cv2.countNonZero(thresh_new)
                values[value_n].new_parts = len([cnt for cnt in cv2.findContours(thresh_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0] if cv2.contourArea(cnt) > 100 and len(cnt) > 15])
            else:
                values[value_n].new_area = None
                values[value_n].new_parts = None

            # Remove non roots
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]  # TODO CONST

            thresh_big = np.zeros_like(thresh)
            cv2.drawContours(thresh_big, contours, -1, 1, cv2.FILLED)

            values[value_n].total_area = cv2.countNonZero(thresh_big)

            # Skeletonize roots
            skeleton = skeletonize(thresh_big).astype(np.float32)

            values[value_n].total_length = cv2.countNonZero(skeleton)

            for x, y in zip(pos_x_median, pos_y_median):
                skeleton[
                    :y,
                    max(x - skeleton.shape[1] // 12, 0):min(x + skeleton.shape[1] // 12, skeleton.shape[1] - 1)
                ] = 0
            conv = cv2.filter2D(src=skeleton, ddepth=-1, kernel=KERNEL_COUNT)

            # Remove pixels, where center pixel in kernel was not 1
            conv = conv - KERNEL_CENTER

            # Leave in the image just intersections (1 = end of root, 2 = part of root, 3+ = intersection)
            ends = [(x, y) for y, x in zip(*np.where(conv == 1))]
            conv[conv <= 2] = 0
            conv[conv >= 3] = 255

            # Enlarge brush for intersection pixels to delete
            conv = cv2.dilate(conv, None, iterations=1)

            skeleton = skeleton.astype(int) - conv.astype(int)
            skeleton[skeleton < 0] = 0
            skeleton = skeleton.astype(np.uint8) * 255

            contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            skeleton = np.zeros_like(skeleton)
            cv2.drawContours(skeleton, [cnt for cnt in contours if len(cnt) > 15], -1, 255, 1)  # TODO const

            skeleton2 = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

            for x, y in zip(pos_x_median, pos_y_median):
                cv2.circle(skeleton2, (x, y), 17, (0, 0, 255), cv2.FILLED)

            contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            corners_upper = []
            corners_lower = [(x, y, 90) for x, y in zip(pos_x_median, pos_y_median)]  # Starting points
            colored = {(x, y): n for n, (x, y) in enumerate(zip(pos_x_median, pos_y_median))}

            pos_x_last = []
            pos_y_last = []
            for i in range(config["n_clusters"]):
                pos_x_last.append([])
                pos_y_last.append([])
                for vn, vv in enumerate(values[max(value_n - 2, 0):value_n+2]):
                    pos_x_last[-1].append(vv.positions_x[i])
                    pos_y_last[-1].append(vv.positions_y[i])

                pos_x_last[-1] = round(np.median(pos_x_last[-1]))
                pos_y_last[-1] = round(np.median(pos_y_last[-1]))
            for n, (x, y) in enumerate(zip(pos_x_last, pos_y_last)):
                y = min(pos_y_median) if y <= min(pos_y_median) else y
                corners_lower.append((x, y, 90))
                colored[(x, y)] = n
                cv2.circle(value.image, (x, y), 10, COLORS[n], cv2.FILLED)

            used = corners_lower.copy()
            # TODO use information from conv, 1 = corner
            for cnt_n, cnt in enumerate(contours):
                corners = []

                for pix in range(len(cnt)):
                    pixel = cnt[pix][0]
                    pixel_x, pixel_y = pixel
                    total = 0

                    for plus_x, plus_y in COMBINATIONS:
                        current_x = pixel_x + plus_x
                        current_y = pixel_y + plus_y

                        if current_y < 0 or current_y >= skeleton.shape[0]:
                            continue

                        if current_x < 0 or current_x >= skeleton.shape[1]:
                            continue

                        if skeleton[current_y][current_x]:
                            total += 1

                    if total == 1:
                        corners.append(tuple(pixel))

                if len(corners):
                    assert len(corners) == 2, f"Corner count is not 2! It is {len(corners)}"
                    upper_corner = [corners[0] if corners[0][1] < corners[1][1] else corners[1]]
                    lower_corner = [corners[0] if corners[0][1] > corners[1][1] else corners[1]]

                    while len(upper_corner) < 10:  # TODO const
                        for pix in range(len(cnt)):
                            pixel = tuple(cnt[pix][0])
                            pixel_x, pixel_y = pixel

                            if pixel not in upper_corner and abs(pixel_x - upper_corner[-1][0]) <= 1 and abs(
                                    pixel_y - upper_corner[-1][1]) <= 1:
                                upper_corner.append(pixel)
                                break
                        else:
                            break

                    while len(lower_corner) < 10:
                        for pix in range(len(cnt)):
                            pixel = tuple(cnt[pix][0])
                            pixel_x, pixel_y = pixel

                            if pixel not in lower_corner and abs(pixel_x - lower_corner[-1][0]) <= 1 and abs(
                                    pixel_y - lower_corner[-1][1]) <= 1:
                                lower_corner.append(pixel)
                                break
                        else:
                            break

                    diff_upper = np.mean(np.diff(upper_corner[::-1], axis=0), axis=0)
                    diff_lower = np.mean(np.diff(lower_corner[::-1], axis=0), axis=0)

                    angle_upper = (math.degrees(math.atan2(diff_upper[1], diff_upper[0])) + 180) % 360
                    angle_lower = (math.degrees(math.atan2(diff_lower[1], diff_lower[0]))) % 360

                    # cv2.putText(skeleton2, f"{angle_upper:.0f}", upper_corner[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    # cv2.putText(skeleton2, f"{angle_lower:.0f}", lower_corner[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                    if not lower_corner[0] in ends or np.median(pos_y_median) + skeleton.shape[0] // 10 > lower_corner[0][1]:
                        cv2.circle(skeleton2, lower_corner[0], 3, (255, 0, 255), 1)
                        corners_lower.append((lower_corner[0][0], lower_corner[0][1], angle_lower, cnt_n))
                        # cv2.putText(skeleton2, f"{angle_lower:.0f}", (lower_corner[0][0], lower_corner[0][1] - random.randint(0, 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    else:
                        cv2.circle(skeleton2, lower_corner[0], 3, (0, 0, 255), 1)

                    corners_upper.append((upper_corner[0][0], upper_corner[0][1], angle_upper, cnt_n, lower_corner[0]))
                    cv2.circle(skeleton2, upper_corner[0], 3, (0, 255, 255), 1)
                    # cv2.putText(skeleton2, f"{angle_upper:.0f}", (upper_corner[0][0], upper_corner[0][1] + random.randint(0, 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            corners_upper = sorted(corners_upper, key=lambda p: p[1])
            corners_lower = sorted(corners_lower, key=lambda p: p[1])

            min_diff_x = min(np.diff(pos_x_median))
            pairs = []
            colored_samples = {n: set() for n in range(config["n_clusters"])}
            for cu in corners_upper:
                distances = []
                angles = []
                for cl in corners_lower:
                    if cl[1] > cu[1]:
                        break

                    # Distance between points
                    distance = np.linalg.norm(np.array(cu[:2]) - np.array(cl[:2]))
                    if (cl[0] in positions_x_median or cl[0] in pos_x_last) and (cl[1] in positions_y_median or cl[1] in pos_y_last) and (abs(cu[0] - cl[0]) > min_diff_x // 2 or abs(cu[1] - cl[1]) > min_diff_x * 2 // 3):  #distance >= min_diff_x * 3 // 2:
                        distance = math.inf
                    elif distance > min_diff_x // 2:
                        distance = math.inf

                    # How different are the angles of the lines
                    distance += abs(cu[2] - cl[2]) * min_diff_x / 180 / 4

                    # How exactly first derivative prediction mismatches
                    angle = math.degrees(np.arctan2(cl[1] - cu[1], cl[0] - cu[0])) + 180

                    distance += abs(angle - cu[2]) * min_diff_x / 180 / 2
                    distance += abs(angle - cl[2]) * min_diff_x / 180 / 2

                    distances.append(distance)  # if distance < 2* min_diff_x else math.inf
                    angles.append([angle, cu[2], cl[2]])

                if len(distances) and min(distances) != math.inf:
                    new_pos = corners_lower[np.argmin(distances)]
                    used.append(new_pos[:3])
                    new_pos = new_pos[:2]

                    if new_pos in colored:
                        if value_n == 0:
                            new_color = colored[new_pos]
                        else:
                            # counts = []
                            for co, se in values[value_n - 1].colored_samples.items():
                                a = set(map(tuple, contours[cu[3]][:, 0]))
                                inte = a.intersection(se)
                                count = len(inte)
                                # inter = round(len(a.intersection(se)) / len(se) * 100, 1)
                                # if inter > 0:
                                if count >= 1:
                                    # print("Aplikuji pravidlo, co bylo drive, je ted")
                                    new_color = co

                                    # Při změně barvy je nutné i přidat novou pozici pro pozdější detekci hl. kořene
                                    colored_k = [ck for ck, cv in colored.items() if cv == new_color]
                                    colored_distances = [abs(xx - cu[0]) + abs(yy - cu[1]) for xx, yy in colored_k]
                                    new_pos = colored_k[np.argmin(colored_distances)]

                                    # ti = value.image.copy()
                                    # cv2.drawContours(ti, [contours[cu[3]]], 0, (0, 0, 255), 3)
                                    # for intere in inte:
                                    #     cv2.circle(ti, intere, 1, (0, 255, 255), 3)
                                    # cv2.imshow("inter", ti)
                                    # cv2.waitKey(0)
                                    break
                            else:
                                new_color = colored[new_pos]
                                # counts.append(count)
                            # print(counts)

                        colored[cu[4]] = new_color

                        colored_samples[new_color] = colored_samples[new_color].union(set(map(tuple, contours[cu[3]][:, 0])))
                        cv2.drawContours(skeleton2, [contours[cu[3]]], 0, COLORS[colored[new_pos]], 1)

                        # if cu[4]:
                        #     cv2.putText(skeleton2, str(cu[3]), ((cu[0] + cu[4][0]) // 2, (cu[1] + cu[4][1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[colored[new_pos]], 1)
                        cv2.drawContours(value.image, [contours[cu[3]]], 0, COLORS[colored[new_pos]], 3)

                    else:
                        # If the root would be colored as 'blank'
                        colored_k = list(colored.keys())
                        colored_v = list(colored.values())
                        colored_distances = [abs(xx - cu[0]) + abs(yy - cu[1]) for xx, yy in colored_k]
                        color_index = np.argmin(colored_distances)
                        new_pos = colored_k[color_index]
                        colored[cu[4]] = colored_v[color_index]

                    # else:
                    #     print("ERR")
                    cv2.line(skeleton2, cu[:2], new_pos, (255, 255, 255), 1)
                    cv2.line(value.image, cu[:2], new_pos, (255, 255, 255), 1)

                    pairs.append((cu[:2], new_pos))
                    # if angles[np.argmin(distances)][-1] > 50:
                    # s = "; ".join([str(round(i)) for i in angles[np.argmin(distances)]])
                    # cv2.putText(skeleton2, s, ((cu[0] + new_pos[0]) // 2, (cu[1] + new_pos[1]) // 2 + random.randint(0, 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            values[value_n].plant_length = []
            values[value_n].longest = []
            values[value_n].colored_samples = colored_samples
            for k in range(config["n_clusters"]):
                colored_k = {ck: cv for ck, cv in colored.items() if cv == k}
                plant_length = [contours[cu[3]] for cu in corners_upper if cu[4] in colored_k]
                mask_to_count = np.zeros_like(skeleton)
                cv2.drawContours(mask_to_count, plant_length, -1, 255, cv2.FILLED)
                values[value_n].plant_length.append(cv2.countNonZero(mask_to_count))

                used = set([u[:2] for u in used])

                end_roots = len([cu for cu in corners_upper if cu[4] in colored and colored[cu[4]] == k and cu[4] not in used])

                top = min(colored_k.keys(), key=lambda z: z[1] if z is not None else math.inf)
                bottom = max(colored_k.keys(), key=lambda z: z[1] if z is not None else -math.inf)

                plant_main_root_depth = bottom[1] - top[1]
                cv2.line(value.image, top, (top[0] + 100, top[1]), (255, 255, 255), 1)
                cv2.line(value.image, (top[0] + 100, top[1]), (top[0] + 100, bottom[1]), (255, 255, 255), 3)
                cv2.line(value.image, (top[0] + 100, bottom[1]), bottom, (255, 255, 255), 1)

                mask_longest = np.zeros_like(skeleton)
                conts = []
                last_len = None
                while last_len != len(conts):
                    last_len = len(conts)
                    for j in corners_upper:
                        if j[4] == bottom:
                            for p1, p2 in pairs:
                                if p1 == j[:2]:
                                    conts.append(contours[j[3]])
                                    bottom = p2
                                    break
                            break

                cv2.drawContours(mask_longest, conts, -1, 255, cv2.FILLED)
                cv2.drawContours(value.image, conts, -1, tuple([min(c + 170, 255) for c in COLORS[k]]), 12)

                # cv2.imshow("long", mask_longest)
                # cv2.imshow("count", mask_to_count)
                # cv2.waitKey(0)

                time_delta = (value.date - values[value_n - 1].date).days
                values[value_n].longest.append(cv2.countNonZero(mask_longest))

                df.append({
                    # Image statistics and information
                    "image_date": value.date,
                    "image_barcode": value.barcode,
                    "image_barcode_read": value.barcode_read,
                    "image_path": value.path,
                    "image_total_area": value.total_area,
                    "image_total_length": value.total_length,
                    "image_new_area": value.new_area,
                    "image_new_parts": value.new_parts,
                    "image_area_change": (abs((value.total_area - value.new_area) - values[value_n - 1].total_area) / values[value_n - 1].total_area * 100) if value_n != 0 else None,

                    # Plant statistics and information
                    "plant_id": k + 1,
                    "plant_center_x": pos_x_median[k],
                    "plant_center_y": pos_y_median[k],
                    "plant_green_area": value.green_areas[k],
                    "plant_root_count": end_roots,
                    "plant_total_length": value.plant_length[k],
                    "plant_total_length_RGR": (np.log(values[value_n].plant_length[k]) - np.log(values[value_n - 1].plant_length[k])) / time_delta if value_n != 0 else None,
                    "plant_main_root_depth": plant_main_root_depth,
                    "plant_main_root_length": values[value_n].longest[k],
                    "plant_main_root_length_RGR": (np.log(values[value_n].longest[k]) - np.log(values[value_n - 1].longest[k])) / time_delta if value_n != 0 else None
                })
            cv2.imwrite(os.path.join(config["data"]["output"], os.path.split(value.path)[-1]), value.image)

    except Exception as e:
        print("error", e, values)

    for key in ["image_total_area", "image_total_length", "plant_total_length", "plant_main_root_depth", "plant_main_root_length"]:
        for plant_id in range(1, config["n_clusters"] + 1):
            temp_df = {}
            for df_i in df:
                if df_i["plant_id"] != plant_id:
                    continue
                if key in temp_df:
                    if df_i[key] < temp_df[key]:
                        print(f"For the image '{df_i['image_path']}' older value of '{key}' is greater than the newer one ({df_i[key]}-{temp_df[key]} = {df_i[key] - temp_df[key]})")
                temp_df[key] = df_i[key]

    return df

if __name__ == "__main__":
    # for i in dictionary.values():
    #    if "23_02-1" in i[0].path:
    #        data = [test(i)]
    freeze_support()
    data = process_map(test, list(dictionary.values()), max_workers=os.cpu_count())

    # Create and save dataframe
    df = pd.DataFrame.from_records([item for sublist in data for item in sublist])
    df.to_csv(config["data"]["statistics"], index=False)
