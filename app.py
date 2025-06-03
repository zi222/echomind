# main.py
import threading
import asyncio
from app_server import start_websocket_server
import os

# 启动 WebSocket 服务
def run_ws():
    asyncio.run(start_websocket_server())

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)  # 确保有 data 文件夹

    # 启动 WebSocket 线程
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # 启动 Streamlit
    os.system("streamlit run app_streamlit.py")
