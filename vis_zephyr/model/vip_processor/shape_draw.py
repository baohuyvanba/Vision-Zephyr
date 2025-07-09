# =================================================================================================
# File: vis_zephyr/model/vip_processorprocessor/shape_draw.py
# Draw: Arrow, Rectangle, Eclipse, Point (x), Scribble, Mask Contour, Mask, Triangle
# Description: Contains functions to draw various shapes on images for visual prompts.
# =================================================================================================
import math
import random
import numpy as np
import random
from shapely.geometry import Point, Polygon
from scipy.stats import multivariate_normal

# ARROW ==============================================================================================================================
def draw_arrow(to_draw, bbox_coor, outline_color, line_width, max_arrow_length=100, max_image_size=336, image_size_anchor = 336):
    left, top, right, bottom = bbox_coor
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2

    #Arrow length ~ BBox size
    bbox_side_length = min(right - left, bottom - top)
    arrow_length     = random.uniform(0.8*bbox_side_length, max_arrow_length)

    #Arrow angle ~ randomize
    arrow_angle = random.uniform(0, 2 * math.pi)
    center_x   += random.uniform(-0.25, 0.25) * (right - left)
    center_y   += random.uniform(-0.25, 0.25) * (bottom - top)

    #Arrow's head size ~ arrow length
    arrow_head_size = max(random.uniform(0.2, 0.5) * arrow_length, int(6*max_image_size/image_size_anchor))

    #Arrow end -> connect with arrow head
    arrow_end_x = center_x + (arrow_length - arrow_head_size)*math.cos(arrow_angle)
    arrow_end_y = center_y + (arrow_length - arrow_head_size)*math.sin(arrow_angle)

    if random.random() < 0.5:
        #Draw with a "wobble": in a human-way
        mid_x = (center_x + arrow_end_x) / 2 + random.uniform(-5, 5)*int(max_image_size/image_size_anchor)
        mid_y = (center_y + arrow_end_y) / 2 + random.uniform(-5, 5)*int(max_image_size/image_size_anchor)
        #Draw the line
        to_draw.line([(center_x, center_y), (mid_x, mid_y), (arrow_end_x, arrow_end_y)],
                     fill  = outline_color,
                     width = line_width
                    )
    else:
        #Draw the line normally
        to_draw.line([(center_x, center_y), (arrow_end_x, arrow_end_y)],
                     fill  = outline_color,
                     width = line_width
                    )
    
    #Draw the arrow head
    arrow_end_x = center_x
    arrow_end_y = center_y
    if random.random() < 0.5:
        to_draw.polygon([
            (arrow_end_x + arrow_head_size * math.cos(arrow_angle + math.pi / 3), arrow_end_y + arrow_head_size * math.sin(arrow_angle + math.pi / 3)),
            (arrow_end_x, arrow_end_y),
            (arrow_end_x + arrow_head_size * math.cos(arrow_angle - math.pi / 3), arrow_end_y + arrow_head_size * math.sin(arrow_angle - math.pi / 3))
        ], fill = outline_color)
    else:
        to_draw.line([
            (arrow_end_x + arrow_head_size * math.cos(arrow_angle + math.pi / 3), arrow_end_y + arrow_head_size * math.sin(arrow_angle + math.pi / 3)),
            (arrow_end_x, arrow_end_y),
            (arrow_end_x + arrow_head_size * math.cos(arrow_angle - math.pi / 3), arrow_end_y + arrow_head_size * math.sin(arrow_angle - math.pi / 3))
        ], fill = outline_color, width = line_width)

# RECTANGLE ==========================================================================================================================
def draw_rectangle(to_draw, bbox_coor, outline_color, line_width):
    left, top, right, bottom = bbox_coor
    #Draw the rectangle
    to_draw.rectangle([(left, top), (right, bottom)], outline = outline_color, width = line_width)

# ELLIPSE ============================================================================================================================
def draw_ellipse(to_draw, bbox_coor, mask_polygon, outline_color, line_width, size_ratio = 1, aspect_ratio = 1):
    if mask_polygon is not None:
        min_x, min_y, max_x, max_y = mask_polygon.bounds
    else:
        min_x, min_y, max_x, max_y = bbox_coor
    
    #BBox Center
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    #New BBox size
    ellipse_width  = (max_x - min_x) * size_ratio * aspect_ratio
    ellipse_height = (max_y - min_y) * size_ratio / aspect_ratio
    min_x = center_x - ellipse_width  / 2
    min_y = center_y - ellipse_height / 2
    max_x = center_x + ellipse_width  / 2
    max_y = center_y + ellipse_height / 2

    #Draw the ellipse
    bbox = [min_x, min_y, max_x, max_y]
    to_draw.ellipse(bbox,
                    outline = outline_color,
                    width   = line_width
    )

# POINT ==============================================================================================================================
def draw_point(to_draw, bbox_coor, mask_polygon, outline_color, radius = 3, aspect_ratio = 1.0):
    if mask_polygon is not None:
        min_x, min_y, max_x, max_y = mask_polygon.bounds
    else:
        min_x, min_y, max_x, max_y = bbox_coor

    #Mean ~ Center of BBox
    mean = [(max_x + min_x) / 2, (max_y + min_y) / 2]
    #Covariance: area to draw the point
    cov  = [[(max_x - min_x) / 8, 0], [0, (max_y - min_y) / 8]]

    #Counter of points had been drawn to track
    counter   = 0
    max_tries = 10

    while True:
        point_x, point_y = multivariate_normal.rvs(mean = mean, cov = cov)
        point = Point(point_x, point_y)

        if mask_polygon.contains(point):
            break
            #Found a point within the mask_polygon

        counter += 1
        if counter >= max_tries:
            #If reach max tries, just draw 10th point
            point_x, point_y = multivariate_normal.rvs(mean = mean, cov = cov)
            center_point = Point(point_x, point_y)
            break
    
    x_radius = radius * aspect_ratio
    y_radius = radius / aspect_ratio
    
    #Draw the point
    bbox = [point_x - x_radius, point_y - y_radius, point_x + x_radius, point_y + y_radius]
    to_draw.ellipse(bbox,
                    fill    = outline_color,
                    outline = outline_color
    )

# SCRIBBLE ===========================================================================================================================
def draw_scribble(to_draw, bbox_coor, mask_polygon, outline_color = (255, 0, 0), line_width = 3, max_image_size = 336, image_size_anchor = 336):
    #Save previous point to connect with next point (current point)
    previous_point = None

    if mask_polygon is not None:
        point_0 = get_random_point_within_polygon(mask_polygon)
        point_1 = get_random_point_within_polygon(mask_polygon)
        point_2 = get_random_point_within_polygon(mask_polygon)
        point_3 = get_random_point_within_polygon(mask_polygon)
    else:
        point_0 = get_random_point_within_bbox(bbox_coor)
        point_1 = get_random_point_within_bbox(bbox_coor)
        point_2 = get_random_point_within_bbox(bbox_coor)
        point_3 = get_random_point_within_bbox(bbox_coor)
    
    for t in np.linspace(0, 1, int(1000 * max_image_size / image_size_anchor)):
        #Calculate the current point using cubic Bezier curve
        x = (1 - t)**3 * point_0[0] + 3 * (1 - t)**2 * t * point_1[0] + 3 * (1 - t) * t**2 * point_2[0] + t**3 * point_3[0]
        y = (1 - t)**3 * point_0[1] + 3 * (1 - t)**2 * t * point_1[1] + 3 * (1 - t) * t**2 * point_2[1] + t**3 * point_3[1]

        current_point = (x, y)
        if previous_point:
            #Draw the line between previous and current points
            to_draw.line([previous_point, current_point],
                         fill  = outline_color,
                         width = line_width
            )
        
        previous_point = current_point

# MASK CONTOUR =======================================================================================================================
def draw_mask_contour(to_draw, bbox_coor, segmentation_coor, color = "red", width = 1):
    #bbox: left, top, right, bottom
    if segmentation_coor is None:
        #Rectangular contour: (min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)
        segmentation_coor = [[bbox_coor[0], bbox_coor[1], bbox_coor[0], bbox_coor[3],
                              bbox_coor[2], bbox_coor[3], bbox_coor[2], bbox_coor[1]]]
    
    for segmentation in segmentation_coor:
        coords = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                #Draw the contour with a thickness
                shifted_coords = [(x + dx, y + dy) for x, y in coords]
                to_draw.polygon(shifted_coords, outline = color)
        
# MASK ===============================================================================================================================
def draw_mask(to_draw, bbox_coor, segmentation_coor, color = "red", width = 1):
    #bbox: left, top, right, bottom
    if segmentation_coor is None:
        #Rectangular contour: (min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)
        segmentation_coor = [[bbox_coor[0], bbox_coor[1], bbox_coor[0], bbox_coor[3],
                              bbox_coor[2], bbox_coor[3], bbox_coor[2], bbox_coor[1]]]
    
    for segmentation in segmentation_coor:
        coords = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
        #Draw the mask
        to_draw.polygon(coords, outline = None, fill = color, width = width)

# TRIANGLE ===========================================================================================================================
def draw_triangle(to_draw, bbox_coor, mask_polygon, outline_color, line_width):
    while True:
        points = []
        for _ in range(3):
            if mask_polygon is not None:
                point = get_random_point_within_polygon(mask_polygon)
            else:
                point = get_random_point_within_bbox(bbox_coor)
            points.append(point)
        
        if is_max_triangle_angle_less_than_150(points):
            break
    
    #Draw the triangle
    to_draw.line([points[0], points[1], points[2], points[0]], fill = outline_color, width = line_width, joint = "curve")



# UTILITIES ==========================================================================================================================
def get_random_point_within_bbox(bbox_coor):
    """Generate a random point within the given bounding box coordinates."""
    left, top, right, bottom = bbox_coor
    x = np.random.uniform(left, right)
    y = np.random.uniform(top, bottom)
    return x, y

def get_random_point_within_polygon(mask_polygon):
    """Generate a random point within the given polygon mask."""
    min_x, min_y, max_x, max_y = mask_polygon.bounds
    trial_num = 0
    while True:
        if trial_num < 50:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = Point(x, y)
            if mask_polygon.contains(point):
                return x, y
            trial_num += 1
        else:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            return x, y

def is_max_triangle_angle_less_than_150(points):
    """Filter out triangles with angles greater than 150 degrees."""
    for i in range(3):
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % 3])
        p3 = np.array(points[(i + 2) % 3])
        
        a  = np.linalg.norm(p3 - p2)
        b  = np.linalg.norm(p1 - p3)
        c  = np.linalg.norm(p1 - p2)
        
        #Calculate angle at p2 using cosine rule
        angle_at_p2 = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
        
        if angle_at_p2 > 150:
            return False
    return True


    



