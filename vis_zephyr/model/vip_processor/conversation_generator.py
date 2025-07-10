# =================================================================================================
# File: vis_zephyr/model/vip_processorprocessor/conservation_organizer.py
#
# =================================================================================================
import random
from PIL import Image, ImageDraw
from networkx import draw
from shapely.ops import unary_union
from shapely.geometry import Polygon

from .configuration import visual_prompt_config, visual_prompt_config_test, color_pool, words_shape
from .shape_draw import draw_rectangle, draw_ellipse, draw_triangle, draw_point, draw_scribble, draw_mask_contour, draw_mask, draw_arrow

def image_blending(
        image, shape = "rectangle",
        bbox_coor = None, segmentation = None,
        image_size_anchor = 336, rgb_color = None,
        vip_style = None, alpha = None, width = None
    ):
    """
    Blends the visual prompt shape onto the image.
    """
    #Image -> RGB
    image = image.convert("RGB")
    img_w, img_h   = image.size
    max_image_size = max(img_w, img_h)

    #Blank RGBA Image
    vip_img    = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    vip_canvas = ImageDraw.Draw(vip_img)

    #Transparency: alpha blending value
    if alpha is None:
        alpha = random.randint(96, 255) if shape != "mask" else random.randint(48, 128)
    color_alpha = rgb_color + (alpha,)

    if segmentation is not None:
        try:
            polygons = []
            for segmentation_coord in segmentation:
                mask_polygon = Polygon([(segmentation_coord[i], segmentation_coord[i+1]) for i in range(0, len(segmentation_coord), 2)])
                polygons.append(mask_polygon)
            mask_polygon = random.choice(polygons)
            try: 
                all_polygons_union = unary_union(polygons)
            except:
                all_polygons_union = None
                # print('Error in all_polygons_union')
        except:
            mask_polygon = None
            # print('Error in Polygon Generation')
    else:
        all_polygons_union = mask_polygon = None

    #DRAW shape on canvas
    if shape == "rectangle":
        line_width = max(int(3 * max_image_size / image_size_anchor), 1) if vip_style == "constant" else max(random.randint( int(2 *max_image_size/image_size_anchor), int(8 * max_image_size/image_size_anchor)), 1)
        line_width = max(int(width * max_image_size / image_size_anchor), 1) if width != None else line_width
        draw_rectangle(
            to_draw       = vip_canvas,
            bbox_coor     = bbox_coor,
            outline_color = color_alpha,
            line_width    = line_width,
        )
    elif shape == "ellipse":
        line_width =  max(random.randint( int(2 *max_image_size/image_size_anchor), int(8 * max_image_size/image_size_anchor)), 1)
        line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        size_ratio = random.uniform(1, 1.5)
        draw_ellipse(
            to_draw       = vip_canvas,
            bbox_coor     = bbox_coor,
            mask_polygon  = all_polygons_union,
            outline_color = color_alpha,
            line_width    = line_width,
            size_ratio    = size_ratio
        )
    elif shape == "arrow":
        line_width = max(random.randint(int(1 * max_image_size / image_size_anchor), int(6 * max_image_size / image_size_anchor)), 1)
        line_width = max(int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        max_arrow_length = max( int(50 * max_image_size/image_size_anchor), 1)
        draw_arrow(
            to_draw = vip_canvas,
            bbox_coor = bbox_coor,
            outline_color = color_alpha,
            line_width = line_width,
            max_arrow_length = max_arrow_length,
            max_image_size = max_image_size,
            image_size_anchor = image_size_anchor
        )
    elif shape == "triangle":
        line_width =  max(random.randint(int(2 *  max_image_size/image_size_anchor), int(8 * max_image_size/image_size_anchor)), 1)
        line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        draw_triangle(
            to_draw       = vip_canvas,
            bbox_coor     = bbox_coor,
            mask_polygon  = all_polygons_union,
            outline_color = color_alpha,
            line_width    = line_width
        )
    elif shape == "point":
        radius = max( int(8 * max_image_size/image_size_anchor), 1) if vip_style == 'constant' else  max(random.randint(int(5 * max_image_size/image_size_anchor),  int(20 *max_image_size/image_size_anchor)), 1)
        aspect_ratio = 1 if random.random() < 0.5 or  vip_style == 'constant' else random.uniform(0.5, 2.0)
        draw_point(
            to_draw       = vip_canvas,
            bbox_coor     = bbox_coor,
            mask_polygon  = all_polygons_union,
            outline_color = color_alpha,
            radius        = radius,
            aspect_ratio  = aspect_ratio
        )
    elif shape == "scribble":
        line_width = max(random.randint(int(2 * max_image_size/image_size_anchor), int(12 * max_image_size/image_size_anchor)), 1)
        line_width = max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        draw_scribble(
            to_draw       = vip_canvas,
            bbox_coor     = bbox_coor,
            mask_polygon  = all_polygons_union,
            outline_color = color_alpha,
            line_width    = line_width,
            max_image_size = max_image_size,
            image_size_anchor = image_size_anchor
        )
    elif shape == "mask contour":
        line_width = max(random.randint( int(1 *max_image_size/image_size_anchor), int(2 * max_image_size/image_size_anchor)), 1)
        line_width = max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        draw_mask_contour(
            to_draw    = vip_canvas,
            bbox_coor  = bbox_coor,
            segmentation_coor = segmentation,
            color      = color_alpha,
            width      = line_width,
        )
    elif shape == "mask":
        line_width = random.randint( int(0 *max_image_size/image_size_anchor), int(2 * max_image_size/image_size_anchor))
        line_width = max(int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        draw_mask(
            to_draw   = vip_canvas,
            bbox_coor = bbox_coor,
            segmentation_coor = segmentation,
            color     = color_alpha,
            width     = line_width
        )
    
    image = image.convert("RGBA")
    #Blend the visual prompt image with the original image
    image = Image.alpha_composite(image, vip_img)
    image = image.convert("RGB")
        
    return image