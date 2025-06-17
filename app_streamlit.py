import streamlit as st
import asyncio
import websockets
from threading import current_thread
from chat_with_wenwu.streamlit_app import get_qa_history_chain, gen_response, get_available_models
from TTS.tts import text_to_speech, initialize_tts_models, save_uploaded_audio, train_voice_model
import os
import uuid
import glob
import time
from chat_with_wenwu.get_vector import get_vectordb
from pathlib import Path
from app_server import send_audio_to_cloud




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

def clean_audio_data_folder():
    audio_files = glob.glob("data/*.wav")
    for f in audio_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to delete {f}: {e}")


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
        st.markdown("### 自定义AI助手音色")
        voice_sample = st.file_uploader(
            "上传你的音色样本 (WAV或MP3格式)", 
            type=["wav","mp3"],
            help="上传10-30秒的清晰语音样本，用于训练AI音色"
        )
        
        # 添加训练按钮
        if st.button("训练AI音色模型"):
            if voice_sample is not None:
                # 保存上传的音频文件
                voice_path = save_uploaded_audio(voice_sample)
                
                # 调用音色训练函数
                with st.spinner("正在训练AI音色，这可能需要几分钟..."):
                    success = train_voice_model(voice_path)
                
                if success:
                    st.success("音色训练完成！AI助手将使用你的音色")
                    st.session_state.custom_voice = True
                    st.audio(voice_path, format='audio/wav')
                else:
                    st.error("音色训练失败，请检查音频格式后重试")
            else:
                st.warning("请先上传音色样本")
        
        # 显示当前音色状态
        if st.session_state.get("custom_voice"):
            st.info("当前使用自定义音色")
        else:
            st.info("使用默认AI音色")
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

        asyncio.run(send_audio_to_cloud(audio_file_path))
if __name__ == "__main__":
    main()