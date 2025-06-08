import dataclasses
from enum import auto, Enum
from typing import List, Tuple

class SeparatorStyle(Enum):
    """Different separator styles."""
    SINGLE = auto()
    TWO    = auto()

@dataclasses.dataclass
class Conversation:
    """Keep track of conservation history"""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    separator_style: SeparatorStyle = SeparatorStyle.TWO
    separator_01: str = " "
    separator_02: str = "</s>"
    version: str = "Unknown"
    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is Tuple:
            messages = self.messages.copy()
            init_role, init_message = messages[0]
            init_message = init_message[0].replace("<image>", "").strip()
            messages[0] = (init_role, "<image>\n" + init_message)

        if self.separator_style == SeparatorStyle.TWO:
            separators = [self.separator_01, self.separator_02]
            ret = self.system + separators[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is Tuple:
                        message, _, _ = message
                    ret += role + ": " + message + separators[i % 2]
                else:
                    ret += role + ": "
        elif self.separator_style == SeparatorStyle.SINGLE:
            ret = self.system + self.separator_01
            for role, message in messages:
                if message:
                    if type(message) is Tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.separator_01
                else:
                    ret += role + ": "
        else:
            raise ValueError(f"Unknown separator style: {self.separator_style}")
        
        return ret
    
    def append_message(self, role, message):
        """Append a message to the conversation."""
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
    system = "You are an AI assistant specialized in Visual Commonsense Reasoning."
             "Given an image and a question, your task is to provide an accurate answer, followed by a concise, logical explanation of your reasoning based on visual cues and common sense. Your response must clearly separate the answer and the explanation.",
    roles = ("USER", "ASSISTANT"),
    messages = (),
    offset = 0,
    separator_style = SeparatorStyle.TWO,
    separator_01 = " ",
    separator_02 = "</s>",
    version = "v1",
)

#Default conversation
default_conversation = conv_zephyr_v1

#Update the templates dictionary
templates = {
    "default"  : conv_zephyr_v1,
    "zephyr_v1": conv_zephyr_v1,
}

if __name__ == "__main__":
    # For testing purposes
    conv = default_conversation.copy()
    conv.append_message("USER", "What is in the image?")
    conv.append_message("ASSISTANT", "This is a test image.")
    print(conv.get_prompt())
    print(conv.messages)
    print(conv.version)
