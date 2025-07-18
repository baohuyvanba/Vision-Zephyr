# =================================================================================================
# File       : vis_zephyr/conversation.py
# Description: Manages conversation history and prompt templating.
# =================================================================================================

import dataclasses
from enum import auto, Enum
from typing import List, Tuple

class SeparatorStyle(Enum):
    """
    Different separator styles, used in conversation formatting.
    """
    ZEPHYR = auto()
    PLAIN  = auto()

@dataclasses.dataclass
class Conversation:
    """
    Keep track of conservation history
    """
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    separator_style: SeparatorStyle = SeparatorStyle.ZEPHYR
    separator_01: str = "</s>"
    separator_02: str = None
    version: str      = "Unknown"
    skip_next: bool   = False

    def get_prompt(self) -> str:
        """
        Get the formatted conversation prompt:
            <|system|>...</s><|user|>...</s><|assistant|>...</s>
        """
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is Tuple:
            messages = self.messages.copy()
            init_role, init_message = messages[0]
            
            #Ensure the first message contain image and also do not duplicate it
            init_message = init_message[0].replace("<image>", "").strip()
            messages[0]  = (init_role, "<image>\n" + init_message)

        if self.separator_style == SeparatorStyle.ZEPHYR:
            #Formating
            separators = [self.separator_01, self.separator_02]
            ret = f"<|system|>\n{self.system}{separators[0]}"      #<|system|>{message}</s>
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    
                    ret += f"<|{role}|>\n{message}{separators[0]}" #<|user|>{prompt}</s>
                else:
                    #Assistan's turn to respond
                    ret += f"<|{role}|>\n" #-> Get respone
            return ret
        else:
            raise ValueError(f"Unknown separator style: {self.separator_style}")
    
    def append_message(self, role, message):
        """Append new message to the conversation history"""
        self.messages.append([role, message])

    def copy(self):
        """Create a copy of the conversation."""
        return Conversation(
            system          = self.system,
            roles           = self.roles,
            messages        = [[r, m] for r, m in self.messages],
            offset          = self.offset,
            separator_style = self.separator_style,
            separator_01    = self.separator_01,
            separator_02    = self.separator_02,
            version         = self.version
        )

#system="A chat between a curious user and an artificial intelligence assistant. "
#       "The assistant gives helpful, detailed, and polite answers to the user's questions.",
conv_zephyr_v1 = Conversation(
    system = "You are an AI assistant specialized in Visual Commonsense Reasoning and able to understand the visual content that the user provides.\n"
             "Given an image and a question, your task is to provide an accurate answer, followed by a concise, logical explanation of your reasoning based on visual cues and common sense. Your response must clearly separate the answer and the explanation.",
    roles = ("user", "assistant"),
    messages = (),
    offset = 0,
    separator_style = SeparatorStyle.ZEPHYR,
    separator_01 = "</s>",
    separator_02 = None,
    version = "zephyr_v1",
)

conv_zephyr_vcr = Conversation(
    system = "You are an AI assistant specialized in Visual Commonsense Reasoning. Your task is to analyze the provided visual content along with a question. Subsequently, select the most appropriate answer from the given choices. Your answer must be in the format 'Answer is: {A, B, C or D}'.",
    roles = ("user", "assistant"),
    messages = (),
    offset = 0,
    separator_style = SeparatorStyle.ZEPHYR,
    separator_01 = "</s>",
    separator_02 = None,
    version = "zephyr_vcr",
)

conv_zephyr_plain = Conversation(
    system = "",
    roles = ("", ""),
    messages = (),
    offset = 0,
    separator_style = SeparatorStyle.PLAIN,
    separator_01 = "</s>",
    separator_02 = None,
    version = "plain",
)

#Default conversation
default_conversation = conv_zephyr_v1

#Update the templates dictionary
templates = {
    "default"   : conv_zephyr_v1,
    "zephyr_v1" : conv_zephyr_v1,
    "zephyr_vcr": conv_zephyr_vcr,
    "plain"     : conv_zephyr_plain,
}

if __name__ == "__main__":
    # For testing purposes
    conv = default_conversation.copy()
    conv.append_message("user", "What is in the image?")
    conv.append_message("assistant", "This is a test image.")
    print("--- Generated Prompt ---")
    print(conv.get_prompt())
    print("\n--- Messages ---")
    print(conv.messages)
    print("\n--- Version ---")
    print(conv.version)
