import streamlit as st
from audio_recorder_streamlit import audio_recorder
import asyncio
import websockets
from threading import current_thread
from concurrent.futures import ThreadPoolExecutor
from chat_with_wenwu.streamlit_app import get_qa_history_chain, gen_response, get_available_models
from TTS.tts import text_to_speech, initialize_tts_models, save_uploaded_audio, train_voice_model
from weather_mcp.MCPClient import MCPClient
import os
import uuid
import glob
import time
from chat_with_wenwu.get_vector import get_vectordb
from pathlib import Path




# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# # 定义线程局部事件循环存储
# event_loops = {}

# def get_thread_loop():
#     thread_id = current_thread().ident
#     if thread_id not in event_loops:
#         event_loops[thread_id] = asyncio.new_event_loop()
#     return event_loops[thread_id]

# 设置当前线程的事件循环
# asyncio.set_event_loop(get_thread_loop())
# @st.cache_resource(show_spinner="加载中...")
@st.cache_resource(show_spinner="加载中...")
def load_qa_chain(model="qwen2.5-coder-32b-instruct",vectordb_path=None):
    return get_qa_history_chain(model=model, vectordb_path=vectordb_path)

@st.cache_resource(show_spinner="别急，小二马上就来...")
def load_tts_models():
    initialize_tts_models(half=True)

async def get_weather_info(city: str) -> str:
    """获取指定城市的天气信息"""
    client = MCPClient()
    try:
        # 设置超时为30秒
        await asyncio.wait_for(
            client.connect_to_server("weather_mcp/server.py"),
            timeout=30.0
        )
        weather_info = await asyncio.wait_for(
            client.process_query(f"{city}的天气"),
            timeout=30.0
        )
        return weather_info
    except asyncio.TimeoutError:
        return "天气查询超时，请稍后再试"
    except Exception as e:
        return f"获取天气信息失败: {str(e)}"
    finally:
        try:
            await asyncio.wait_for(client.cleanup(), timeout=5.0)
        except:
            pass  # 确保无论如何都能继续

def clean_audio_data_folder():
    audio_files = glob.glob("data/*.wav")
    for f in audio_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to delete {f}: {e}")
# 辅助函数 - 获取音频时长
def get_audio_duration(file_path):
    """获取音频文件时长（秒）"""
    try:
        import wave
        with wave.open(file_path, 'r') as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            duration = frames / float(rate)
            return duration
    except:
        # 如果是MP3文件，使用其他库
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  # 毫秒转秒
        except:
            return 0


def main():
    load_tts_models()
    clean_audio_data_folder()  # 每次运行清理 data/ 目录下旧音频
    st.markdown('### 欢迎使用知识库私人助手')
    
    # 添加模型选择框
    available_models = get_available_models()
    
    # 在侧边栏添加模型选择
    with st.sidebar:
        st.markdown("### 模型设置")
        selected_model = st.selectbox(
            "选择AI模型",
            available_models,
            index=0,  # 默认选择第一个模型
            help="选择不同的AI模型来获得不同的回答效果"
        )
    
        # 显示当前选择的模型信息
        st.info(f"当前使用模型: {selected_model}")
        
        # 添加清除对话历史的按钮
        if st.button("清除对话历史"):
            st.session_state.messages = []
            st.rerun()
    
        # 添加知识库上传组件
        st.markdown("### 自定义知识库")
        uploaded_file = st.file_uploader("上传你的知识库PDF文件", type=["pdf"])
        vectordb = None
        if uploaded_file is not None:
           # 保存上传的 PDF
            knowledge_dir = Path("/root/codespace/data/knowledge_db")
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = knowledge_dir / uploaded_file.name
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("正在处理知识库..."):
                # 构建持久化向量路径
                persist_dir = Path("/root/codespace/data/vector_db") / uploaded_file.name.replace(".pdf", "")
                persist_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.vectordb_path = (str(pdf_path), str(persist_dir))
                # 加载或构建向量数据库
                get_vectordb(str(pdf_path), str(persist_dir))
                st.success("知识库上传成功！AI助手将使用你的知识库回答问题")
            
        else:
            st.session_state.vectordb_path = None
            st.warning("未上传知识库，AI将使用现有知识回答问题")

        # 添加自定义音色上传组件
        st.markdown("### 自定义AI助手音色") #  
    
        # 创建选项卡：上传文件或录制音频
        upload_option = st.radio("选择音色输入方式：", 
                                ["上传音频文件", "录制音频样本"],
                                horizontal=True)   #  
    
        voice_sample = None
        recording_file = "recording.wav"    #  
    
        if upload_option == "上传音频文件":
           # 文件上传器
            voice_sample = st.file_uploader(
               "上传你的音色样本 (WAV或MP3格式)", 
               type=["wav", "mp3"],
               help="上传10-30秒的清晰语音样本，用于训练AI音色"
           )
        else:
           # 录音组件
            st.info("请录制10-30秒的清晰语音")
           
           # 添加录音说明
           # 添加录音说明
            with st.expander("录音提示"):
                st.markdown("""
                - 找一个安静的环境录制
                - 保持麦克风距离嘴部15-20厘米
                - 用自然的声音说话（不要刻意改变音调）
                - 录制完成后可以回放确认质量
                - 请朗读以下文本：
                """)
                # 添加推荐朗读文本
                st.code("""
                大家好，我是[你的名字]，现在正在为我的AI助手录制声音样本。
                今天天气真不错，阳光明媚，风和日丽。
                我希望训练出一个自然、流畅的语音助手。
                     12345，上山打老虎。
                     """)


            # 自定义录音按钮
            audio_bytes = audio_recorder(
                text="点击开始录音",
                recording_color="#e87070",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=30,  # 最长录制30秒
            )
           
            # 显示录音结果并提供播放功能
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                # 保存录音文件
                with open(recording_file, "wb") as f:
                    f.write(audio_bytes)
               
               # 添加确认按钮
                if st.button("使用此录音样本"):
                   if os.path.exists(recording_file):
                       voice_sample = recording_file
                       st.session_state.voice_sample = recording_file
                       st.session_state.voice_sample_confirmed = True
                       st.success("录音样本已确认！")
                   else:
                       st.error("录音文件不存在，请重新录制")
                elif st.button("重新录制"):
                   # 重置状态
                   voice_sample = None
                   if os.path.exists(recording_file):
                       os.remove(recording_file)
                   st.rerun()   
    
        # 添加训练按钮
        if st.button("训练AI音色模型"):
            # 检查所有可能的样本来源
            voice_sample = (voice_sample or 
                          st.session_state.get('voice_sample') or
                          (st.session_state.get('voice_sample_confirmed') and recording_file))
            
            if voice_sample and (isinstance(voice_sample, st.runtime.uploaded_file_manager.UploadedFile) or 
                               (isinstance(voice_sample, str) and os.path.exists(voice_sample))):
                # 处理上传文件的情况
                if isinstance(voice_sample, st.runtime.uploaded_file_manager.UploadedFile):
                    voice_path = save_uploaded_audio(voice_sample)
                # 处理录音文件的情况
                else:
                    voice_path = voice_sample
                
                # 验证音频长度
                duration = get_audio_duration(voice_path)
                if duration < 5:
                    st.error("音频太短（小于5秒），请提供至少10秒的样本")
                elif duration > 60:
                    st.error("音频太长（超过60秒），请缩短至30秒以内")
                else:
                    # 调用音色训练函数
                    with st.spinner("正在训练AI音色，这可能需要几分钟..."):
                        success = train_voice_model(voice_path)
                    
                    if success:
                        st.success("音色训练完成！AI助手将使用你的音色")
                        st.session_state.custom_voice = True
                        st.audio(voice_path, format='audio/wav')
                        
                    else:
                        st.error("音色训练失败，请检查音频质量后重试")
            else:
                st.warning("请先提供音色样本（上传文件或录制音频）")    #  
    
        # 显示当前音色状态
        if st.session_state.get("custom_voice"):
            st.success("当前使用你的自定义音色")
        else:
            st.info("使用默认AI音色") #  
    

        # # 添加自定义音色上传组件
        # st.markdown("### 自定义AI助手音色")
        # voice_sample = st.file_uploader(
        #     "上传你的音色样本 (WAV或MP3格式)", 
        #     type=["wav","mp3"],
        #     help="上传10-30秒的清晰语音样本，用于训练AI音色"
        # )
        
        # # 添加训练按钮
        # if st.button("训练AI音色模型"):
        #     if voice_sample is not None:
        #         # 保存上传的音频文件
        #         voice_path = save_uploaded_audio(voice_sample)
                
        #         # 调用音色训练函数
        #         with st.spinner("正在训练AI音色，这可能需要几分钟..."):
        #             success = train_voice_model(voice_path)
                
        #         if success:
        #             st.success("音色训练完成！AI助手将使用你的音色")
        #             st.session_state.custom_voice = True
        #             st.audio(voice_path, format='audio/wav')
        #         else:
        #             st.error("音色训练失败，请检查音频格式后重试")
        #     else:
        #         st.warning("请先上传音色样本")
        
        # # 显示当前音色状态
        # if st.session_state.get("custom_voice"):
        #     st.info("当前使用自定义音色")
        # else:
        #     st.info("使用默认AI音色")
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vectordb_path" in st.session_state:
        st.session_state.qa_history_chain = load_qa_chain(
        model=selected_model,
        vectordb_path=st.session_state.vectordb_path
    )
    else:
        st.session_state.qa_history_chain = load_qa_chain(
        model=selected_model,
        vectordb_path=None
    )
    # 存储当前选择的模型
    if "current_model" not in st.session_state:
        st.session_state.current_model = available_models[0]
    
    # 检查模型是否发生变化，如果变化则重新加载问答链
    if st.session_state.current_model != selected_model:
        st.session_state.current_model = selected_model
        # 清除缓存并重新加载问答链
        load_qa_chain.clear()
        st.session_state.qa_history_chain = load_qa_chain(model=selected_model, vectordb_path=st.session_state.vectordb_path)
        st.success(f"已切换到模型: {selected_model}")
    
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = load_qa_chain(model=selected_model, vectordb_path=st.session_state.vectordb_path)
    
    # 建立容器 高度为500 px
    messages = st.container(height=500)
    
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
                st.write(message[1]) # 打印内容
    
    if prompt := st.chat_input("请输入您的问题..."):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        

        # 简单的中文城市名到英文映射
        city_mapping = {
            "北京": "Beijing",
            "上海": "Shanghai",
            "广州": "Guangzhou",
            "深圳": "Shenzhen",
            "杭州": "Hangzhou",
            "成都": "Chengdu"
        }
        
        # 检查是否询问天气
        if "天气" in prompt:
            # 尝试提取城市名
            city_cn = next((c for c in city_mapping if c in prompt), "北京")
            city_en = city_mapping[city_cn]
            
            try:
                # 使用线程池处理异步调用
                with ThreadPoolExecutor() as executor:
                    weather_info = executor.submit(
                        lambda: asyncio.run(get_weather_info(city_en))
                    ).result()
                
                # 将天气信息合并到输入中
                prompt = f"{prompt}\n当前{city_cn}天气信息:\n{weather_info}"
            except Exception as e:
                print(f"天气查询异常: {e}")
                prompt = f"{prompt}\n[天气查询服务暂时不可用]"

        # 生成回复（带知识库检索）
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages,
            # vectors=st.session_state.get("vectors")
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入 st.session_state.messages
        st.session_state.messages.append(("ai", output))

        # 生成语音文件
        audio_file_path = "data/audio.wav" 
        # 如果用户训练过音色，则用用户音色的npy路径，否则用默认
        user_voice_path = "/root/codespace/data/user_voice.npy"
        prompt_tokens_path = [user_voice_path if st.session_state.get("custom_voice") and os.path.exists(user_voice_path) else "/root/codespace/TTS/fake.npy"]

        text_to_speech(output,prompt_tokens_path=prompt_tokens_path, output_audio_path=audio_file_path)
        st.audio(audio_file_path)

        #asyncio.run(send_audio_to_cloud(audio_file_path))
if __name__ == "__main__":
    main()
# import streamlit as st
# from audio_recorder_streamlit import audio_recorder

# audio_bytes = audio_recorder(
#     text="点击录音",
#     recording_color="#e87070",
#     neutral_color="#6aa36f",
#     icon_name="microphone",
#     icon_size="2x",
# )

# if audio_bytes:
#     st.audio(audio_bytes, format="audio/wav")
#     # 保存录音文件
#     with open("recording.wav", "wb") as f:
#         f.write(audio_bytes)
