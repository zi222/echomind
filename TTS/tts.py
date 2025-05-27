from .fish_speech.models.text2semantic.inference import generate_semantic_tokens
from .fish_speech.models.text2semantic.inference import load_model as load_text2semantic_model
from .fish_speech.models.vqgan.inference import decode_to_audio
from .fish_speech.models.vqgan.inference import load_model as load_vqgan_model

import numpy as np
from pathlib import Path
import torch


# 提前加载模型
semantic_model = None
vqgan_model = None
decode_one_token = None

def initialize_tts_models(half: bool = False):
    global semantic_model, vqgan_model, decode_one_token
    # 加载语义模型
    precision = torch.half if half else torch.bfloat16
    semantic_model, decode_one_token = load_text2semantic_model(
        checkpoint_path="/root/codespace/TTS/checkpoints/fish-speech-1.5",
        device="cuda",
        precision=precision,
        compile=False,
    )
    # 加载 VQGAN 模型
    vqgan_model = load_vqgan_model(
        config_name="firefly_gan_vq",
        checkpoint_path="/root/codespace/TTS/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        device="cuda"
    )
def text_to_speech(
    text: str,
    prompt_text = ["你的参考文本"],
    prompt_tokens_path = ["/root/codespace/TTS/fake.npy"],
    output_audio_path ="/root/codespace/TTS/output.wav",
) :
    """
    直接调用 Fish-Speech 的 Python API 生成语音
    """
    # 生成语义 token (codes_0.npy)
    generate_semantic_tokens(
        text=text,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens_path,
        num_samples=1,
        max_new_tokens=0,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        checkpoint_path="/root/codespace/TTS/checkpoints/fish-speech-1.5",
        device="cuda",
        compile=False,
        seed=42,
        half=False,
        iterative_prompt=True,
        chunk_length=100,
        output_dir="/root/codespace/TTS/temp",
        model=semantic_model,
        decode_one_token=decode_one_token,
    )
    
    # # 保存中间 token（假设需要）
    # np.save("codes_0.npy", semantic_tokens)
    
    # VQGAN 解码生成音频
    decode_to_audio(
        input_path=Path("/root/codespace/TTS/temp/codes_0.npy"),
        output_path=output_audio_path,
        model=vqgan_model,
        device="cuda",
    )
    
    # # 保存为 WAV 文件
    # torchaudio.save(output_audio_path, audio, sample_rate=24000)
if __name__ == "__main__":

    text_to_speech("秦始皇陵兵马俑是世界八大奇迹之一，位于中国陕西省西安市临潼区，是中国古代伟大的军事统帅秦始皇的陵墓。兵马俑是陪葬品，主要用于保护秦始皇的灵魂。兵马俑的发现改变了人们对古代中国的认识，展示了当时的军事、文化和艺术水平。")