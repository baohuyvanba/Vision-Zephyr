# =================================================================================================
# File: vis_zephyr/model/vip_processorprocessor/utils.py
# Description: All utility functions for Visual Prompt processing in Vision-Zephyr models.
# =================================================================================================
import random
import collections

from .configuration import words_shape, answer_map, QUESTION_PREFIXES, OPTIONS_PREFIXES, WHY_QUESTIONS

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

def get_adjective():
    return random.choice(['The correct', 'The most accurate', 'The best', 'The ultimate', 'The final', 'The only', 'The ideal', 'The optimal', 'The most fitting', 'The definitive'])

def get_punctuation():
    return random.choice([':', '->', '→', '::', '—', ';', '|', '⇒'])

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
    shapes = random.choices(shapes_list, k = len(all_instance_index))
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
                color_rgb  = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
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

def get_all_question_answer(all_corpus, shape_and_color, class_names, answer_type = ''):
    """
    Replaces instances index by Visual Prompt information in the question and answer corpora.
    """
    all_text = []
    shape_and_color_vip_img = []

    for corpus in all_corpus:
        text = ''
        for instance_index, instance in enumerate(corpus):
            if type(instance) == list:
                #Replace instance index with Visual Prompt information
                for obj_index in range(len(instance)):
                    shape_color = shape_and_color[instance[obj_index]]
                    
                    if instance_index == 0 and obj_index == 0:
                        text += 'The '
                    else:
                        text += ' the '
                    
                    if class_names == None:
                        text += 'object'
                    elif random.random() < 0.5 and answer_type != 'direct':
                        text += random.choice(['object', 'instance'])
                    else:
                        text += class_names[instance[obj_index]]
                    
                    word_1, word_2 = words_shape[shape_color[2]]
                    #Word 1: prepositions - within, on, ...
                    text += ' ' + word_1 + ' '
                    #Word 2: shape - circle, square, ...
                    if random.random() < 0.5:
                        text += 'the'
                    if shape_color[0] is not None:
                        text += shape_color[0] + ' '
                    text += word_2

                    if obj_index != len(instance) - 1:
                        text += ' and'
                    
                    shape_and_color_vip_img.append(instance[obj_index])
            elif type(instance) == str:
                text += instance
            else:
                breakpoint()
            
            if instance_index != len(corpus) - 1 and type(corpus[instance_index + 1]) == str:
                if corpus[instance_index + 1] not in {'.', ',', '?', '!', ':', ';'}:
                    text += ' '
        
        all_text.append(text)

    return all_text, shape_and_color_vip_img

def get_question(question, all_choices, use_multiplechoice_q, why_question = False, no_image = False):
    """
    Randomly selects a question from a predefined list.
    """
    if why_question:
        question_prompt = random.choice(WHY_QUESTIONS)
    else:
        image_str = '' if no_image else '<image>\n'
        question_prompt = image_str + random.choice(QUESTION_PREFIXES) + question
    
    if use_multiplechoice_q:
        all_options = ''
        for choice_idx, choice in enumerate(all_choices):
            choice = '(' + answer_map[choice_idx] + ') ' + choice
            all_options += choice

            if choice_idx != len(all_choices) - 1:
                all_options += ' '
            else:
                all_options += ''
        
        question_prompt += " " + random.choice(OPTIONS_PREFIXES) + all_options
    
    return question_prompt

def get_answer(choice, content, use_multiplechoice_r):
    choice = answer_map[choice]
    choice_upper = choice.upper()

    if use_multiplechoice_r:
        content = content[0].lower() + content[1:] if content else content
        content = random.choice([
            f'({choice_upper})',
            f'({choice_upper})',
            f'{get_adjective()} answer is ({choice_upper})',
            f'{get_adjective()} answer is ({choice_upper})',
            f'({choice_upper}){get_punctuation()} {content}',
            f'({choice_upper}){get_punctuation()} {content}',
            f'{get_adjective()} answer is ({choice_upper}) — {content}',
            f'{get_adjective()} answer is ({choice_upper}) — {content}',
            f'({choice_upper}) — {get_adjective()} because {content}',
            f'({choice_upper}) — {get_adjective()} because {content}',
            f'Answer ({choice_upper}): {content}',
            f'Answer ({choice_upper}): {content}',
            f'Opt for ({choice_upper}) if {content}',
            f'Opt for ({choice_upper}) if {content}'
        ])
        return content.replace("\u2024", "-")
    else:
        return content