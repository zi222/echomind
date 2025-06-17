
import asyncio
import websockets
import os

async def send_wav_file():
    uri = "ws://182.92.128.19:8000/ws/upload"  # 请替换为你阿里云 IP
    wav_file = "data/audio.wav"
    CHUNK_SIZE = 256 * 1024 

    if not os.path.exists(wav_file):
        print(f"Error: File {wav_file} not found!")
        return

    async with websockets.connect(uri, max_size=50 * 1024 * 1024) as websocket:
        file_size = os.path.getsize(wav_file)
        # 先发送 metadata（文件名和大小）
        await websocket.send(f"{os.path.basename(wav_file)},{file_size}")

        with open(wav_file, "rb") as f:
            bytes_sent = 0
            while bytes_sent < file_size:
                chunk = f.read(CHUNK_SIZE)
                await websocket.send(chunk)
                bytes_sent += len(chunk)
                print(f"Sending: {(bytes_sent/file_size)*100:.1f}%", end="\r")

        print(f"\nWAV 文件发送完毕：{wav_file} ({file_size} bytes)")

async def start_server():
    while True:
        try:
            await send_wav_file()
        except Exception as e:
            print(f"Error: {e}")
        await asyncio.sleep(5)  # 每隔 5 秒重试


async def send_audio_to_cloud(audio_path):
    uri = "ws://182.92.128.19:8000/ws/upload"
    CHUNK_SIZE = 256 * 1024

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

async def receive_text_from_pi():
    uri = "ws://182.92.128.19:8000/ws/text"  # 替换为你的阿里云IP
    
    async with websockets.connect(uri) as websocket:
        print("无问芯穹已连接，等待接收树莓派文字...")
        try:
            while True:
                message = await websocket.recv()
                print(f"📩 收到树莓派消息: {message}")
                
                # 这里可以添加业务逻辑，例如：
                # 1. 调用 NLP 处理
                # 2. 存储到数据库
                # 3. 返回响应给树莓派
                
        except websockets.exceptions.ConnectionClosed:
            print("连接已关闭")

# 运行接收端
asyncio.get_event_loop().run_until_complete(receive_text_from_pi())


# if __name__ == "__main__":
#     asyncio.run(start_server())
