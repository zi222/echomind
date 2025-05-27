import streamlit as st
import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
# sys.path.append("./data_base") # 将父目录放入系统路径中
from .get_vector import get_vectordb
from .model_to_llm import model_to_llm

from threading import Thread, current_thread
import sqlite3

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# 定义线程局部事件循环存储
event_loops = {}

def get_thread_loop():
    thread_id = current_thread().ident
    if thread_id not in event_loops:
        event_loops[thread_id] = asyncio.new_event_loop()
    return event_loops[thread_id]

# 设置当前线程的事件循环
asyncio.set_event_loop(get_thread_loop())

# OPENAI API 访问密钥配置
GENSTUDIO_API_KEY = "sk-bvzk5vkwe4jxejv2"
DEFAULT_BASE_URL = "https://cloud.infini-ai.com/maas/v1/"

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain(model:str="qwen1.5-14b-chat"):
    llm = model_to_llm(model=model, temperature=0.2, API_KEY=GENSTUDIO_API_KEY, DEFAULT_BASE_URL=DEFAULT_BASE_URL)

    condense_question_system_template = (
    "将用户问题转换为最简短的完整问句，要求：\n"
    "1. 消除所有指代不明的词汇（这/那/它）\n" 
    "2. 技术名词保留原词（不要简化'随机森林'为'那个算法'）\n"
    "3. 用主谓宾结构重组句子（例如：'怎么做红烧肉？'）\n"
    "4. 总长度控制在20字以内\n"
)
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
    vectordb = get_vectordb("/root/codespace/chat_with_wenwu/data_base/knowledge_db/national_treasure.pdf","/root/codespace/chat_with_wenwu/data_base/vector_db/chroma")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
    "你是一个会说话的私人知识库助手，回答需满足语音播报要求：\n"
    "1. 使用最简短的日常口语，避免复杂句式\n"
    "2. 每句话不超过15个字，用逗号自然分隔\n" 
    "3. 重要数字要放句首（如：'准确率82%，这个模型表现不错'）\n"
    "4. 遇到步骤说明时用'然后'连接（例如：先热锅，然后放油）\n"
    "5. 知识库内容优先，找不到时说：'根据我的记录...'\n"
    "6. 适当加入语气词（比如'呢'、'呀'）保持自然\n"
    "当前知识库：\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain

# 获取可用模型列表
def get_available_models():
    return [
        "qwen1.5-14b-chat",
        "megrez-3b-instruct", 
        "chatglm3",
        "llama-2-7b-chat",
        "deepseek-r1",
        "qwen3-30b-a3b",
        "glm-4-9b-chat",
        "yi-1.5-34b-chat",
        "qwen2-72b-instruct",
        "qwen2.5-7b-instruct"
        "qwen2.5-coder-32b-instruct"
    ]

def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]

def main():
    st.markdown('### 欢迎使用博物馆文物助手')
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
                st.write(message[1]) # 打印内容
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))
if __name__ == "__main__":
    main()
