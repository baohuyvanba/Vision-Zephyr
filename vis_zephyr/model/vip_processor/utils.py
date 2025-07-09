# =================================================================================================
# File: vis_zephyr/model/vip_processorprocessor/utils.py
# Description: All utility functions for Visual Prompt processing in Vision-Zephyr models.
# =================================================================================================
import random
import collections

# ====================================================================================================================================
def build_prompt_from_multiple_choices(question, options):
    """
    Builds a string prompt from a question and multiple choice options.
    """
    if len(options) != 4:
        return "Error: Expected 4 options, got {}".format(len(options))
    
    #Options -> String
    options_str = "\n".join(["{}. {}".format(chr(65 + i), option) for i, option in enumerate(options)])

    #Prompt
    prompt = f"""{question}
{options_str}
Answer with the option's letter from the given choices directly."""
    
    return prompt

#====================================================================================================================================
# VCR UTILITIES
#====================================================================================================================================
def get_all_instance(all_corpus):
    """
    Extracts all unique instance indices from a list of corpora.
    """
    all_instance_index = []
    for corpus in all_corpus:
        for instance in corpus:
            if type(instance) == list:
                all_instance_index.extend(instance)
    
    all_instance_index = list(set(all_instance_index))
    return all_instance_index

def get_color_and_shape(all_instance_index, shapes_list, color_list):
    """
    Get a unique pair of Color and Shape for each Instance
    """
    #Random shape
    shapes = random.choice(shapes_list, k = len(all_instance_index))
    shape_counts      = collections.Counter(shapes)
    non_unique_shapes = {shape for shape, count in shape_counts.items() if count > 1}

    results         = {}
    shape_and_color = {}

    for i, instance in enumerate(all_instance_index):
        shape = shapes[i]

        #Add shape to dictionary (shape_and_color) if not already present
        if shape not in shape_and_color:
            shape_and_color[shape] = []

        #Shape already choisen/in not unique shapes
        if (shape in non_unique_shapes) or (shape in shape_and_color and shape_and_color[shape]):
            avail_colors = [color for color in color_list if color[0] not in shape_and_color[shape]]
            
            #Color still available
            if avail_colors:
                #Randomly choose a color
                color_name, color_rgb = random.choice(avail_colors)
                #Add to dict
                shape_and_color[shape].append(color_name)
            #No color available
            else:
                color_name = None
                color_rbg  = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        #Shape not already chosen/unique
        else:
            if random.choice([True, False]):
                color_name, color_rgb = random.choice(color_list)
                shape_and_color[shape].append(color_name)
            else:
                color_name = None
                color_rgb  = (random.randint(0,255), random.randint(0, 255), random.randint(0, 255))

        results[instance] = [color_name, color_rgb, shape]
    
    return results


