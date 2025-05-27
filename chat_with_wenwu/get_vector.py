import os
from .data_base.embedding import create_db, load_knowledge_db

def get_vectordb(file_path:str=None, persist_path:str=None):
    if os.path.exists(persist_path):  #持久化目录存在
        contents = os.listdir(persist_path)
        if len(contents) == 0:  #但是下面为空
            #print("目录为空")
            vectordb = create_db(file_path, persist_path)
            #presit_knowledge_db(vectordb)
            vectordb = load_knowledge_db(persist_path)
        else:
            #print("目录不为空")
            vectordb = load_knowledge_db(persist_path)
    else: #目录不存在，从头开始创建向量数据库
        vectordb = create_db(file_path, persist_path)
        #presit_knowledge_db(vectordb)
        vectordb = load_knowledge_db(persist_path)

    return vectordb

if __name__ == "__main__":

    question = "讲一下司母戊鼎的历史"
    docs = get_vectordb("./data_base/knowledge_db/national_treasure.pdf", "./data_base/vector_db/chroma").similarity_search(question,k=3)
    print(f"检索到的内容数：{len(docs)}")
    for i, doc in enumerate(docs):
        print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")
