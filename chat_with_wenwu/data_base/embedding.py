from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modelscope import snapshot_download
import re
import os
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import Chroma



from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI

from langchain_openai import ChatOpenAI
# from langchain.chat_models import ChatOpenAI


# # 加载环境变量
# load_dotenv()
# API_KEY = os.getenv("GENSTUDIO_API_KEY")
# DEFAULT_BASE_URL = os.getenv("DEFAULT_BASE_URL")

# # 实例化chatopenai类
# chat = ChatOpenAI(openai_api_key=API_KEY, openai_api_base=DEFAULT_BASE_URL,model="deepseek-r1", streaming=False, temperature=0.0)

PDF_PATH="/root/codespace/chat_with_wenwu/data_base/knowledge_db/national_treasure.pdf"
DEFAULT_PERSIST_PATH = "/root/codespace/chat_with_wenwu/data_base/vector_db/chroma"

def get_embedding_function():
    local_model_dir = "/root/codespace/chat_with_wenwu/models/BAAI/bge-base-zh-v1.5"
    remote_model_name = "BAAI/bge-base-zh-v1.5"

    if os.path.exists(local_model_dir):
        print(f"加载本地模型: {local_model_dir}")
        embedding_function = HuggingFaceEmbeddings(model_name=local_model_dir)
    else:
        print(f"本地模型不存在，尝试加载远程模型: {remote_model_name}")
        embedding_function = HuggingFaceEmbeddings(model_name=remote_model_name)
    return embedding_function


# 数据处理与索引构建模块
def create_db(files=PDF_PATH, persist_directory=DEFAULT_PERSIST_PATH):
    # 加载PDF文档
    loader = PyMuPDFLoader(files)
    pdf_pages = loader.load()

    for pdf_page in pdf_pages:
        # 文本清洗处理
        content = pdf_page.page_content
        # 处理跨行连接的中文字符
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        content = re.sub(pattern, lambda m: m.group(0).replace('\n', ''), content)
        # 去除特殊字符
        content = content.replace('•', '').replace(' ', '').replace('\n', '')
        
    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,  
        chunk_overlap=20,
        separators=['。', '！', '？', '；', '，'],
        keep_separator=False
    )
    # split_docs = text_splitter.split_text(pdf_page.page_content)  #split_text方法应该接受一个字符串参数，然后将这个字符串分割成多个块，返回字符串列表。
    split_docs = text_splitter.split_documents(pdf_pages)   #处理整个文档列表

    embedding = get_embedding_function()
    vector_db = Chroma.from_documents(documents=split_docs, embedding=embedding, persist_directory=persist_directory)
    #vector_db.persist()
    return vector_db

def load_knowledge_db(path):
    """
    该函数用于加载向量数据库。

    参数:
    path: 要加载的向量数据库路径。

    返回:
    vectordb: 加载的数据库。
    """

    vector_db =  Chroma(
        persist_directory=path,
        embedding_function = get_embedding_function()
    )
    return vector_db

def augment_prompt(query: str):
  
    # 获取top3的文本片段
    results = create_db(documents).similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    # 构建prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    contexts:
    {source_knowledge}

    query: {query}"""
    return augmented_prompt

# 主程序
if __name__ == "__main__":

    question = "讲一下司母戊鼎的历史"
    docs = load_knowledge_db("./data_base/vector_db/chroma").similarity_search(question,k=3)
    print(f"检索到的内容数：{len(docs)}")
    for i, doc in enumerate(docs):
        print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")


#     query = "四羊方尊是什么年代的?"
#     # result = create_db(documents).similarity_search(query ,k = 3)
#     # print(f"相似度查询结果：{result}")
#     # 创建prompt
#     prompt = HumanMessage(
#         content=augment_prompt(query)
#     )
#     messages = [
#     SystemMessage(content="你是一个文物助手."),
#     HumanMessage(content="你知道四羊方尊吗."),
#     # AIMessage(content="你是谁?"),
#     # HumanMessage(content="橙子"),

# ]
#     messages.append(prompt)

#     res = chat(messages)

#     print(res.content)
