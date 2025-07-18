# =================================================================================================
# File: vis_zephyr/model/vip_processor/processor.py
# Description:
#    - Processes visual prompts for images, blending shapes and colors based on configurations.
#    - Generates conversations based on visual prompts for datasets like VCR.
# =================================================================================================
import json
import random
import numpy as np

from .conversation_generator import image_blending
from .configuration import visual_prompt_config, visual_prompt_config_test, color_pool, answer_map
from .utils import get_all_instance, get_all_question_answer, get_color_and_shape, build_prompt_from_multiple_choices, get_question, get_answer

def visual_prompt_process(
    source,
    image,
    image_size_anchor,
    data_args,
):
    dataset_maintype, dataset_subtype = source['id'].split('-')[0], source['id'].split('-')[1]

    #Visual Prompt Type choices
    if getattr(data_args, "visual_prompt_style", None) != None:
        #TESTING/VALIDATING
        vip_shapes, vip_style = visual_prompt_config_test[data_args.visual_prompt_style]
    else:
        vip_shapes, vip_style = visual_prompt_config[dataset_maintype]

    #VCR
    if dataset_maintype in {'vcr'}:
        #Metadata
        source['meta_dir'] = source['meta_dir'].replace('./dataset', data_args.image_folder)
        metadata = json.load(open(source['meta_dir']))

        if getattr(data_args, "visual_prompt_style", None) == 'vcr_qa':
            shape_color_info, all_instance_index, conversations = create_question_qa_direct(source, vip_shapes, color_list = list(color_pool.items()))
        elif getattr(data_args, "visual_prompt_style", None) == 'vcr_qar':
            shape_color_info, all_instance_index, conversations = create_question_qar_direct(source, vip_shapes, color_list = list(color_pool.items()))
        else:
            shape_color_info, all_instance_index, conversations = create_question_qa_qar(source, vip_shapes, color_list = list(color_pool.items()))
        
        #Get bboxes and segmentations
        source['bboxes']        = [metadata['boxes'][instance_index][:-1] for instance_index in all_instance_index]
        source['segmentations'] = []

        for instance_index in all_instance_index:
            segmentation_data = []
            for segmentation_idx in range(len(metadata['segms'][instance_index]) - 1, -1, -1):
                if len(metadata['segms'][instance_index][segmentation_idx]) >= 4:
                    segmentation_data.append(list(np.array(metadata['segms'][instance_index][segmentation_idx]).flatten()))
            
            if len(segmentation_data) > 0:
                source['segmentations'].append(segmentation_data)
            else:
                source['segmentations'].append(None)

    #Alpha Blending
    alpha = getattr(data_args, "alpha", None)
    for instance_index, (bbox, segmentation) in enumerate(zip(source['bboxes'], source['segmentations'])):
        color_name, color_rgb, shape = shape_color_info[instance_index]

        #Add ViP to Image
        image = image_blending(
            image             = image,
            shape             = shape,
            image_size_anchor = image_size_anchor,
            rgb_color         = color_rgb,
            bbox_coor         = bbox,
            segmentation      = segmentation,
            vip_style         = vip_style,
            alpha             = alpha,
        )
    
    #conversation = source['conversation']
    return image, conversations

# VCR ================================================================================================================================
def create_question_qa_direct(source, shapes_list, color_list):
    question = [source['question']]
    answer   = source['answer_choices']
    
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
    shape_and_color_vip_image_all = []

    #Question: Instance ID -> ViP
    question, shape_and_color_vip_img = get_all_question_answer(
        all_corpus      = question,
        shape_and_color = shape_and_color,
        class_names     = class_names,
        answer_type     = 'direct'
    )
    question = question[0]
    shape_and_color_vip_image_all.extend(shape_and_color_vip_img)

    #Answer: Instance ID -> ViP
    answer, shape_and_color_vip_img = get_all_question_answer(
        all_corpus      = answer,
        shape_and_color = shape_and_color,
        class_names     = class_names,
        answer_type     = 'direct'
    )
    shape_and_color_vip_image_all.extend(shape_and_color_vip_img)

    question_prompt = '<image>\n' + build_prompt_from_multiple_choices(question, answer)
    question_answer_prompt = answer_map[source['answer_label']]

    conversations = [
        {
            "from": "human",
            "value": question_prompt
        },
        {
            "from": "gpt",
            "value": question_answer_prompt
        },
    ]
    shape_and_color = [shape_and_color[instance_index] for instance_index in all_instance_index]

    return shape_and_color, all_instance_index, conversations

def create_question_qar_direct(source, shapes_list, color_list):
    question       = [source['question']]
    orginal_answer = [source['answer_choices'][source['answer_label']]]
    rationale      = source['rationale_choices']

    #Get all instances from the question and answer
    all_corpus = question + orginal_answer + rationale
    all_instance_index = get_all_instance(all_corpus)

    #Get instances's shape+color 
    shape_and_color = get_color_and_shape(
        all_instance_index = all_instance_index,
        shapes_list        = shapes_list,
        color_list         = color_list
    )

    class_names = source['class_names']
    shape_and_color_vip_image_all = []

    #Question: Instance ID -> ViP
    question, shape_and_color_vip_img = get_all_question_answer(
        all_corpus      = question,
        shape_and_color = shape_and_color,
        class_names     = class_names,
        answer_type     = 'direct'
    )
    question = question[0]
    shape_and_color_vip_image_all.extend(shape_and_color_vip_img)

    #Answer: Instance ID -> ViP
    original_answer, shape_and_color_vip_img = get_all_question_answer(
        all_corpus      = original_answer,
        shape_and_color = shape_and_color,
        class_names     = class_names,
        answer_type     = 'direct'
    )
    original_answer = original_answer[0]
    shape_and_color_vip_image_all.extend(shape_and_color_vip_img)

    #Rationale: Instance ID -> ViP
    rationale, shape_and_color_vip_img = get_all_question_answer(
        all_corpus      = rationale,
        shape_and_color = shape_and_color,
        class_names     = class_names,
        answer_type     = 'direct'
    )
    shape_and_color_vip_image_all.extend(shape_and_color_vip_img)

    question_prompt  = build_prompt_from_multiple_choices('', rationale)
    rationale_prompt = answer_map[source['rationale_label']]
    
    conversations = [
        {
            "from": "human",
            "value":  '<image>\n' + f'I give you a question and its answer, I need you to provide a rationale explaining why the answer is right. "{question}" The answer is "{original_answer}".What is the rationale for this decision?{question_prompt}' 
        },
        {
            "from": "gpt",
            "value": rationale_prompt
        },
    ]
    
    shape_and_color = [shape_and_color[instance_index] for instance_index in all_instance_index]

    return shape_and_color, all_instance_index, conversations

def create_question_qa_qar(source, shapes_list, color_list):
    # --- Data Augmentation --- by randomly choosing multiple choice or answer generation
    use_multiplechoice_q = random.random() < 0.5
    use_multiplechoice_r = random.random() < 0.5

    question = [source['question']]

    if not use_multiplechoice_q:
        #Get true answer
        answer = [ source['answer_choices'][source['answer_label']] ]
    else:
        #Get all answers
        answer = source['answer_choices']

    if not use_multiplechoice_r:
        #Get true rationale
        rationale = [ source['rationale_choices'][source['rationale_label']] ]
    else:
        #Get all rationales
        rationale = source['rationale_choices']
    
    #Get all instances from the question and answer
    all_corpus         = question + answer + rationale
    all_instance_index = get_all_instance(all_corpus)

    #Get instances's shape+color
    shape_and_color = get_color_and_shape(
        all_instance_index = all_instance_index,
        shapes_list        = shapes_list,
        color_list         = color_list
    )

    class_names = source['class_names']
    shape_and_color_vip_image_all = []

    #Question: Instance ID -> ViP
    question, shape_and_color_vip_img = get_all_question_answer(
        all_corpus      = question,
        shape_and_color = shape_and_color,
        class_names     = class_names,
    )
    question = question[0]
    shape_and_color_vip_image_all.extend(shape_and_color_vip_img)

    #Answer: Instance ID -> ViP
    answer, shape_and_color_vip_img = get_all_question_answer(
        all_corpus      = answer,
        shape_and_color = shape_and_color,
        class_names     = class_names,
    )
    shape_and_color_vip_image_all.extend(shape_and_color_vip_img)

    #Rationale: Instance ID -> ViP
    rationale, shape_and_color_vip_img = get_all_question_answer(
        all_corpus      = rationale,
        shape_and_color = shape_and_color,
        class_names     = class_names,
    )
    shape_and_color_vip_image_all.extend(shape_and_color_vip_img)

    # --- Build Conversation Prompts ---
    #Question for QA
    question_prompt = get_question(
        question             = question,
        all_choices          = answer,
        use_multiplechoice_q = use_multiplechoice_q
    )

    #Answer for Q
    answer_idx = source['answer_label'] if use_multiplechoice_q else 0
    q_answer_prompt = get_answer(
        choice               = answer_idx,
        content              = answer[answer_idx],
        use_multiplechoice_r = use_multiplechoice_q #multiple choice mode
    )

    #Question for QR
    rationale_prompt = get_question(
        question             = None,
        all_choices          = rationale,
        use_multiplechoice_q = use_multiplechoice_r,
        why_question         = True
    )

    #Answer for R
    r_answer_idx = source['rationale_label'] if use_multiplechoice_r else 0
    r_answer_prompt = get_answer(
        choice               = r_answer_idx,
        content              = rationale[r_answer_idx],
        use_multiplechoice_r = use_multiplechoice_r
    )

    conversations = [
        {
            "from": "human",
            "value": question_prompt
        },
        {
            "from": "gpt",
            "value": q_answer_prompt
        },
        {
            "from": "human",
            "value": rationale_prompt
        },
        {
            "from": "gpt",
            "value": r_answer_prompt
        }
    ]
    shape_and_color = [shape_and_color[instance_index] for instance_index in all_instance_index]

    return shape_and_color, all_instance_index, conversations