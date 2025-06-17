#VISUAL PROMPT SUPPORTED DATASETS
visual_prompt_config = dict(
    refcocog  = [ ["rectangle", "ellipse", "triangle", "point", "scribble", "mask contour", "mask", "arrow"], ''],
    vcr       = [ ["rectangle", "ellipse", "triangle", "scribble", "mask contour", "mask", "arrow"], ''],
    vg_rel    = [ ["rectangle", "ellipse",], ''],
    flickr30k = [ ["rectangle", "ellipse", "arrow"], ''],
    v7w       = [ ["rectangle"],'constant'],
    pointQA_twice = [ ["rectangle"], 'constant'],
)