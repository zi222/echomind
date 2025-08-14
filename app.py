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
_chat_history = []

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
    except asyncio.TimeoutError:
        print("Sending keepalive ping...")
        await websocket.ping() # 发送心跳包，保持连接状态
    except Exception as e:
        print(f"❌ WebSocket 发送失败：{e}（音频保留以便重试）")

def get_available_pdfs():
    """获取knowledge_db目录下所有PDF文件"""
    knowledge_dir = Path("/root/codespace/data/knowledge_db")
    knowledge_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    return list(knowledge_dir.glob("*.pdf"))

def load_qa_chain(model="qwen2.5-coder-32b-instruct", vectordb_path=None):
    global _qa_chain, _vectordb_path
    _qa_chain = get_qa_history_chain(model=model, vectordb_path=vectordb_path)
    print(f"QA chain loaded with model {model}")

def initialize_models():
    print("初始化 TTS 模型...")
    # 设置内存优化配置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # 初始化TTS模型，关闭half精度以减少内存问题
    initialize_tts_models(half=False)
    clean_audio_data_folder()
    load_qa_chain(model=_selected_model, vectordb_path=None)

async def process_text(text):
    # 检查是否有新的PDF知识库
    pdf_files = get_available_pdfs()
    if pdf_files:
        pdf_file = pdf_files[0]  # 使用第一个PDF文件
        pdf_path = str(pdf_file)
        persist_dir = Path("/root/codespace/data/vector_db") / pdf_file.stem
        persist_dir.mkdir(parents=True, exist_ok=True)
        vectordb_path = (pdf_path, str(persist_dir))
        
            
        print(f"检测到知识库文件：{pdf_path}，更新QA链...")
        try:
            load_qa_chain(model=_selected_model, vectordb_path=vectordb_path)
        except Exception as e:
            print(f"加载QA链失败: {e}")
            print("将使用无知识库模式继续...")
            load_qa_chain(model=_selected_model, vectordb_path=None)
    else:
        print("未检测到知识库文件，使用现有QA链")

    # 生成问答结果（同步调用，这里如果gen_response支持异步请改成await）
    print(f"收到文本，生成回答：{text}")
    answer = gen_response(
        chain=_qa_chain,
        input=text,
        chat_history=_chat_history,
    )
    # 如果answer是生成器，合并成字符串
    if hasattr(answer, '__iter__') and not isinstance(answer, str):
        output = "".join(answer)
    else:
        output = answer
    _chat_history.append(("human", text))
    _chat_history.append(("ai", output))

    print(f"生成回答: {output}")

    # 合成语音，输出路径固定
    audio_file_path = "data/audio.mp3"
    # 检查是否有自定义音色
    user_voice_path = "/root/codespace/data/user_voice.npy"
    prompt_tokens_path = [user_voice_path if os.path.exists(user_voice_path) else "/root/codespace/TTS/fake.npy"]

    try:
        text_to_speech(output, prompt_tokens_path=prompt_tokens_path, output_audio_path=audio_file_path)
    except Exception as e:
        print(f"语音合成失败: {e}")
        # 清理CUDA缓存
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 重试一次
        try:
            text_to_speech(output, prompt_tokens_path=prompt_tokens_path, output_audio_path=audio_file_path)
        except Exception as e:
            print(f"语音合成重试失败: {e}")
            raise
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
        except asyncio.TimeoutError:
            print("Sending keepalive ping...")
            await websocket.ping() # 发送心跳包，保持连接状态
        except websockets.exceptions.ConnectionClosed as e:
            print(f"连接关闭: {e}")
        except Exception as e:
            print(f"异常: {e}")

if __name__ == "__main__":
    initialize_models()
    asyncio.run(receive_text_from_pi())
