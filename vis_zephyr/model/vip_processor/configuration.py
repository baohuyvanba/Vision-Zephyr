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

#Answer Mapping
answer_map = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}

#Prefixes for question prompts
QUESTION_PREFIXES = [
    'Based on the provided source image, please answer this question: ',
    'In the context of the source image, can you answer: ',
    'With reference to the source image, please respond to the following query: ',
    "Considering the source image, what's your answer to: ",
    'Please provide an answer for the subsequent question, keeping the source image in mind: ',
    'Taking into account the source image, please answer: ',
    'After observing the source image, could you please answer the following: ',
    'Upon examining the source image, what would your answer be to: ',
    'Using the source image as a reference, please respond to: ',
    'In light of the source image, could you please answer: '
]

#MULTIPLE CHOICE PREFIXES
OPTIONS_PREFIXES = [
    'Available choices are as follows: ',
    'Select from the options below: ',
    'You may choose from the following: ',
    'Your choices include: ',
    'Here are your options: ',
    'Please pick one from the given possibilities: ',
    'The following options are available: ',
    'You have the following selections: ',
    'Which among these would you choose: ',
    'You can select from these alternatives: '
]


#Why Questions to ask model generate Rationale
WHY_QUESTIONS = [
    'why?',
    'why',
    "What's the rationale for your decision?",
    'What led you to that conclusion?',
    "What's the reasoning behind your opinion?",
    'Why do you believe that to be true?',
    'Can you explain the basis for your thinking?',
    'What factors influenced your perspective?',
    'How did you arrive at that perspective?',
    'What evidence supports your viewpoint?',
    'What makes you think that way?',
    "What's the logic behind your argument?",
    'Can you provide some context for your opinion?',
    "What's the basis for your assertion?",
    'Why do you hold that belief?',
    'What experiences have shaped your perspective?',
    'What assumptions underlie your reasoning?',
    "What's the foundation of your assertion?",
    "What's the source of your reasoning?",
    "What's the motivation behind your decision?",
    "What's the impetus for your belief?",
    "What's the driving force behind your conclusion?",
    'Why do you think that?',
    "What's your reasoning?",
    'What makes you say that?',
    'Why do you feel that way?',
    "What's the story behind that?",
    "What's your thought process?",
    "What's the deal with that?",
    "What's the logic behind it?",
    'Why do you believe that?',
    "What's the real deal here?",
    "What's the reason behind it?",
    "What's the thought process behind your decision?",
    "What's the rationale for your opinion?",
    'Why do you have that impression?',
    "What's the background to that?",
    "What's the evidence that supports your view?",
    "What's the explanation for that?"
]