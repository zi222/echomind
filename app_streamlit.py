import streamlit as st
import asyncio
from threading import current_thread
from chat_with_wenwu.streamlit_app import get_qa_history_chain, gen_response, get_available_models
from TTS.tts import text_to_speech, initialize_tts_models
import os
import uuid
import glob


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
def load_qa_chain(model="qwen1.5-14b-chat"):
    return get_qa_history_chain(model=model)

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
        uploaded_file = st.file_uploader("上传PDF知识库", type=["pdf"])
        # if uploaded_file is not None:
        #     # 读取PDF内容
        #     pdf_text = read_pdf_file(uploaded_file)
        #     # 文本向量化
        #     vectors = get_vector(pdf_text)
        #     # 存储向量化数据
        #     st.session_state.vectors = vectors
        #     st.session_state.knowledge_base = uploaded_file
        #     st.success("知识库已上传并完成向量化")
        # else:
        #     st.session_state.vectors = None
        #     st.warning("未上传知识库，AI将使用现有知识回答问题")
    
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 存储当前选择的模型
    if "current_model" not in st.session_state:
        st.session_state.current_model = available_models[0]
    
    # 检查模型是否发生变化，如果变化则重新加载问答链
    if st.session_state.current_model != selected_model:
        st.session_state.current_model = selected_model
        # 清除缓存并重新加载问答链
        load_qa_chain.clear()
        st.session_state.qa_history_chain = load_qa_chain(model=selected_model)
        st.success(f"已切换到模型: {selected_model}")
    
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = load_qa_chain(model=selected_model)
    
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
        audio_file_path = f"data/{uuid.uuid4().hex}.wav"  # 使用 UUID 生成唯一文件名
        os.makedirs("data", exist_ok=True)  # 确保 data 目录存在


        text_to_speech(output, output_audio_path=audio_file_path)
        st.audio(audio_file_path)
if __name__ == "__main__":
    main()
