import streamlit as st
import websockets, asyncio, threading
from pydub import AudioSegment

st.title("音频服务端")

# 设置WAV文件绝对路径（修改为你的实际路径）
WAV_FILE = "data/yinpin.wav"    # Linux路径示例

async def handler(websocket):
    while True:
        text = await websocket.recv()
        st.write(f"收到: {text}")
        
        # 从指定路径加载WAV文件
        audio = AudioSegment.from_wav(WAV_FILE)
        await websocket.send_bytes(audio.raw_data)

def start_server():
    asyncio.run(websockets.serve(handler, "localhost", 8501))

if st.button("启动服务"):
    threading.Thread(target=start_server, daemon=True).start()
    st.success(f"服务已启动，使用音频文件: {WAV_FILE}")