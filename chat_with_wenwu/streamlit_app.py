import streamlit as st
import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from weather_mcp.MCPClient import MCPClient
# sys.path.append("./data_base") # 将父目录放入系统路径中
from .get_vector import get_vectordb
from .model_to_llm import model_to_llm

from threading import Thread, current_thread
import sqlite3
from datetime import datetime

current_time = datetime.now().strftime("%Y年%m月%d日 %A %H:%M:%S")

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

def get_qa_history_chain(model:str="qwen2.5-coder-32b-instruct",vectordb_path=None):
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
    system_prompt = (
    "你是一个温柔甜美的私人知识库助手，当前日期是{current_time}。\n"
    "1. 你的语气温柔甜美，使用最简短的日常口语，避免复杂句式\n"
    "2. 每句话尽量简短，尽量不超出20字\n" 
    "3. 如果存在知识库，优先使用知识库内容回答，如果使用知识库，就说，根据您上传的知识库记录；如果不用知识库就不用说'\n"
    "4.调用mcp服务没有使用知识库，所以不用说根据您上传的知识库记录'\n"
    "5. 适当加入语气词保持自然\n"
    "当前知识库：\n{context}。"
    )
    vectordb = None
    if vectordb_path:
        try:
            pdf_path, persist_dir = vectordb_path
            vectordb = get_vectordb(pdf_path, persist_dir)
        except Exception as e:
            st.error(f"加载向量数据库失败: {e}")

    if vectordb is not None:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        
        # 定义带检索的文档获取流程
        retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )       
        
        # 定义组合文档的函数（当有检索器时使用）
        def combine_docs_with_retriever(x):
            docs = retrieve_docs.invoke(x)
            return "\n\n".join(doc.page_content for doc in docs)

    else:
        print(f"无法初始化向量数据库，将不使用检索功能")
        
        # 定义空文档组合函数（当没有检索器时使用）
        def combine_docs_with_retriever(x):
            return ""


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs_with_retriever)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context=combine_docs_with_retriever, 
        ).assign(answer=qa_chain)
    return qa_history_chain

# 获取可用模型列表
def get_available_models():
    return [
        "qwen2.5-coder-32b-instruct",
        "megrez-3b-instruct", 
        "llama-2-7b-chat",
        "deepseek-r1",
        "qwen3-30b-a3b",
        "glm-4-9b-chat",
        "yi-1.5-34b-chat",
        "qwen2-72b-instruct",
        "qwen2.5-7b-instruct",
        "qwen2.5-coder-32b-instruct"
    ]

async def get_weather_info(city: str) -> str:
    """获取指定城市的天气信息"""
    client = MCPClient()
    try:
        # 设置超时为10秒
        await asyncio.wait_for(
            client.connect_to_server("weather_mcp/server.py"),
            timeout=10.0
        )
        weather_info = await asyncio.wait_for(
            client.process_query(f"{city}的天气"),
            timeout=10.0
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

def gen_response(chain, input, chat_history):
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
    if "天气" in input:
        # 尝试提取城市名
        city_cn = next((c for c in city_mapping if c in input), "北京")
        city_en = city_mapping[city_cn]
        
        try:
            # 获取当前事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环已在运行，创建新线程处理异步调用
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    weather_info = executor.submit(
                        lambda: asyncio.run(get_weather_info(city_en))
                    ).result()
            else:
                # 否则直接运行
                weather_info = loop.run_until_complete(get_weather_info(city_en))
            
            # 将天气信息合并到输入中
            input = f"{input}\n当前{city_cn}天气信息:\n{weather_info}"
        except Exception as e:
            print(f"天气查询异常: {e}")
            input = f"{input}\n[天气查询服务暂时不可用]"

    response = chain.stream({
        "input": input,
        "chat_history": chat_history,
        "current_time": current_time
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
