# # websocket_server.py
# import asyncio
# import websockets
# import os

# async def send_wav(websocket):
#     wav_file = "data/audio.wav"
#     CHUNK_SIZE = 8192

#     if not os.path.exists(wav_file):
#         await websocket.send("ERROR:File not found")
#         return

#     file_size = os.path.getsize(wav_file)
#     await websocket.send(f"{os.path.basename(wav_file)},{file_size}")

#     with open(wav_file, "rb") as f:
#         while True:
#             chunk = f.read(CHUNK_SIZE)
#             if not chunk:
#                 break
#             await websocket.send(chunk)

# async def start_websocket_server():
#     async with websockets.serve(send_wav, "0.0.0.0", 8765, max_size=50 * 1024 * 1024):
#         print("WebSocket server running on ws://0.0.0.0:8765")
#         await asyncio.Future()  # 保持运行


import asyncio
import websockets
import os


async def send_wav(websocket):
    # 指定要发送的WAV文件路径
    wav_file = "data/audio.wav"  # 替换为你实际的WAV文件
    CHUNK_SIZE = 8192  # 每次传输8KB数据块

    if not os.path.exists(wav_file):
        print(f"Error: File {wav_file} not found!")
        await websocket.send("ERROR:File not found")
        return

    # 获取文件大小
    file_size = os.path.getsize(wav_file)

    # 发送文件名和文件大小(用逗号分隔)
    await websocket.send(f"{os.path.basename(wav_file)},{file_size}")

    # 分块读取并发送文件内容
    with open(wav_file, "rb") as f:
        bytes_sent = 0
        while bytes_sent < file_size:
            chunk = f.read(CHUNK_SIZE)
            await websocket.send(chunk)
            bytes_sent += len(chunk)

            # 显示进度(可选)
            progress = (bytes_sent / file_size) * 100
            print(f"Sending: {progress:.1f}%", end="\r")

    print(f"\nFile {wav_file} sent successfully ({file_size} bytes)")


async def main():
    # 设置最大传输大小为50MB
    async with websockets.serve(
            send_wav,
            "0.0.0.0",
            8080,
            max_size=50 * 1024 * 1024
    ):
        print("WebSocket server started on ws://0.0.0.0:8080")
        await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    asyncio.run(main())