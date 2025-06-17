import asyncio
import websockets
import os
import glob
from chat_with_wenwu.streamlit_app import get_qa_history_chain, gen_response, get_available_models
from TTS.tts import text_to_speech, initialize_tts_models
from pathlib import Path

# 缓存问答链
_qa_chain = None
_selected_model = "qwen2.5-coder-32b-instruct"

def clean_audio_data_folder():
    audio_files = glob.glob("data/*.wav")
    for f in audio_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

async def send_audio_to_cloud(audio_path):
    uri = "ws://182.92.128.19:8000/ws/upload"
    CHUNK_SIZE = 512 * 1024

    try:
        async with websockets.connect(uri, max_size=50 * 1024 * 1024) as websocket:
            file_size = os.path.getsize(audio_path)
            await websocket.send(f"{os.path.basename(audio_path)},{file_size}")

            with open(audio_path, "rb") as f:
                bytes_sent = 0
                while bytes_sent < file_size:
                    chunk = f.read(CHUNK_SIZE)
                    await websocket.send(chunk)
                    bytes_sent += len(chunk)

        print(f"✅ 音频已发送至服务器：{audio_path}")

        # 发送成功，删除音频
        os.remove(audio_path)
        print(f"🗑️ 已删除本地音频文件：{audio_path}")

    except Exception as e:
        print(f"❌ WebSocket 发送失败：{e}（音频保留以便重试）")

def load_qa_chain(model="qwen2.5-coder-32b-instruct", vectordb_path=None):
    global _qa_chain
    _qa_chain = get_qa_history_chain(model=model, vectordb_path=vectordb_path)
    print(f"QA chain loaded with model {model}")

def initialize_models():
    print("初始化 TTS 模型...")
    initialize_tts_models(half=True)
    clean_audio_data_folder()
    load_qa_chain(model=_selected_model)

async def process_text(text):
    # 生成问答结果（同步调用，这里如果gen_response支持异步请改成await）
    print(f"收到文本，生成回答：{text}")
    answer = gen_response(
        chain=_qa_chain,
        input=text,
        chat_history=[],
    )
    # 如果answer是生成器，合并成字符串
    if hasattr(answer, '__iter__') and not isinstance(answer, str):
        output = "".join(answer)
    else:
        output = answer

    print(f"生成回答: {output}")

    # 合成语音，输出路径固定
    audio_file_path = "data/audio.mp3"
    # 你这里可以加自定义音色判断，这里简单用默认音色
    prompt_tokens_path = ["/root/codespace/TTS/fake.npy"]

    text_to_speech(output, prompt_tokens_path=prompt_tokens_path, output_audio_path=audio_file_path)
    print(f"语音合成完成，路径：{audio_file_path}")

    # 发送音频文件
    await send_audio_to_cloud(audio_file_path)

async def receive_text_from_pi():
    uri = "ws://182.92.128.19:8000/ws/text"
    async with websockets.connect(uri) as websocket:
        print("连接树莓派文字输入 WebSocket 成功，等待消息...")
        try:
            while True:
                message = await websocket.recv()
                print(f"收到树莓派消息: {message}")

                # 收到文字后处理生成语音并发送
                await process_text(message)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"连接关闭: {e}")
        except Exception as e:
            print(f"异常: {e}")

if __name__ == "__main__":
    initialize_models()
    asyncio.run(receive_text_from_pi())