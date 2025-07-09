# =================================================================================================
# File: vis_zephyr/model/vip_processor/processor.py
# Description: Handles Visual Prompt processing for Vision-Zephyr models.
# =================================================================================================
import json

from configuration import visual_prompt_config, visual_prompt_config_test, color_pool, words_shape
from utils import get_all_instance, get_color_and_shape

def visual_prompt_process(
    source,
    image,
    image_size_anchor,
    data_args,
):
    dataset_maintype, dataset_subtype = source['id'].split('-')[0], source['id'].split('-')[1]

    #Visual Prompt Type choices
    if getattr(data_args, "visual_prompt_style", None) != None:
        #test dataset
        vip_shapes, vip_style = visual_prompt_config_test[data_args.visual_prompt_style]
    else:
        vip_shapes, vip_style = visual_prompt_config[dataset_maintype]

    #VCR
    if dataset_maintype in {'vcr'}:
        #Metadata
        source['meta_dir'] = source['meta_dir'].replace('./dataset', data_args.image_dir)
        metadata = json.load(open(source['meta_dir']))

        if getattr(data_args, "visual_prompt_style", None) == 'vcr_qa':
            shape_color_info, all_instance_index, conservation = create_question_qa(source, vip_shapes, color_list = list(color_pool.items()))

    
    conversation = source['conversation']
    return image, conversation

# VCR ================================================================================================================================
def create_question_qa(source, shapes_list, color_list):
    question = [source['question']]
    answer   = [source['answer']]
    
    #Get all instances from the question and answer
    all_corpus = question + answer
    all_instance_index = get_all_instance(all_corpus)

    #Get instances's shape+color 
    shape_and_color = get_color_and_shape(
        all_instance_index = all_instance_index,
        shapes_list        = shapes_list,
        color_list         = color_list
    )

    class_names = source['class_names']


