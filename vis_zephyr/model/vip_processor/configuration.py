# =================================================================================================
# File: vis_zephyr/model/vip_processorprocessor/conservation_organizer.py
# Description: Configuration for Visual Prompt - including supported datasets, visual prompt types, and styles.
# =================================================================================================


#VISUAL PROMPT SUPPORTED DATASETS
visual_prompt_config = dict(
    refcocog  = [ ["rectangle", "ellipse", "triangle", "point", "scribble", "mask contour", "mask", "arrow"], ''],
    vcr       = [ ["rectangle", "ellipse", "triangle", "scribble", "mask contour", "mask", "arrow"], ''],
    vg_rel    = [ ["rectangle", "ellipse",], ''],
    flickr30k = [ ["rectangle", "ellipse", "arrow"], ''],
    v7w       = [ ["rectangle"],'constant'],
    pointQA_twice = [ ["rectangle"], 'constant'],
)

visual_prompt_config_test = dict(
    vcr_qa  = [ ["point"], 'constant'],
    vcr_qar = [ ["point"], 'constant'],
)

#COLOR
color_pool = {
    'red'       : (255, 0, 0),
    'lime'      : (0, 255, 0),
    'blue'      : (0, 0, 255),
    'yellow'    : (255, 255, 0),
    'fuchsia'   : (255, 0, 255),
    'aqua'      : (0, 255, 255),
    'orange'    : (255, 165, 0),
    'purple'    : (128, 0, 128),
    'gold'      : (255, 215, 0),
}

#WORDINGs for different shapes -> text prompts
words_shape ={
    "rectangle"     : ["within", "rectangle"], 
    "ellipse"       : ["within", "ellipse"],
    "triangle"      : ["with", "triangle"],
    "point"         : ["at", "point"], 
    "scribble"      : ["with", "scribble"], 
    "mask contour"  : ["with", "mask contour"],
    "mask"          : ["with", "mask"],
    "arrow"         : ["pointed to by", "arrow"],
 }