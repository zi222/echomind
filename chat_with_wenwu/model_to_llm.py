from langchain_openai import ChatOpenAI

def model_to_llm(model:str=None, temperature:float=0.0, API_KEY:str=None, DEFAULT_BASE_URL:str=None):

    # Initialize the ChatOpenAI model with streaming enabled
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=DEFAULT_BASE_URL,
        model=model,
        temperature=temperature,
    )
    return llm
