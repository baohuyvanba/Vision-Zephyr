from vis_zephyr.train.train import train

#Flash-Attention 2
from vis_zephyr.train.zephyr_flash_attn_monkey_patch import replace_mistral_attn_with_flash_attn
replace_mistral_attn_with_flash_attn()

if __name__ == "__main__":
    train(
        attn_implementation = "flash_attention_2",
    )