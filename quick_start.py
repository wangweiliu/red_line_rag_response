import logging
import time
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler=logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler=logging.FileHandler(f'log/{int(time.time() * 1000)}.log',encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter=logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.debug(f'使用设备:{device}')

from typing import List, Tuple
import json
def split_into_chunks(json_file:str)->List[Tuple[str,str]]:
    logger.info(f'加载并拆分数据文件: {json_file}')
    with open(json_file, 'r', encoding='utf-8') as file:
        content=json.load(file)
    return [(item['question'],item['answer']) for item in content]


from sentence_transformers import SentenceTransformer
# BAAI/bge-large-zh-v1.5
# shibing624/text2vec-base-chinese
embedding_model=SentenceTransformer('BAAI/bge-large-zh-v1.5', device=device)
def embed_chunk(chunk:str) -> List[float]:
    embedding=embedding_model.encode(chunk) # numpt array
    return embedding.tolist()


import chromadb
chromadb_client=chromadb.EphemeralClient()
chromadb_collection=chromadb_client.get_or_create_collection(name='default')
def save_embedding(chunks:List[Tuple[str,str]], embeddings:List[List[float]])->None:
    ids=[str(i) for i in range(len(chunks))]
    documents = [chunk[0] for chunk in chunks] 
    metadatas = [{"answer": chunk[1]} for chunk in chunks] 

    chromadb_collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    logger.info(f'已保存 {len(chunks)} 条嵌入到 ChromaDB')

def retrieve(query:str, top_k:int=10)->List[Tuple[str,str]]:
    logger.info(f'检索与查询相关的前 {top_k} 条内容')
    query_embedding=embed_chunk(query)
    results=chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieve_question=results['documents'][0]
    retrieve_metadata=results['metadatas'][0]

    return [(question, metadata['answer']) for question, metadata in zip(retrieve_question, retrieve_metadata)]


from sentence_transformers import CrossEncoder
activation_fn=torch.nn.Sigmoid()
def rerank(query:str, retrieved_chunks:List[Tuple[str,str]], top_k:int=3)->List[Tuple[Tuple[str,str], float]]:
    # cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
    # maidalun1020/bce-reranker-base_v1
    logger.info('对检索结果进行重排序')
    cross_encoder=CrossEncoder('BAAI/bge-reranker-large', activation_fn=activation_fn, device=device) 
    pairs=[(query, answer) for question,answer in retrieved_chunks]
    scores=cross_encoder.predict(pairs)
    logger.debug(f'重排序得分列表:{scores}')
    chunk_with_scores_list=[(chunk, score) for chunk, score in zip(retrieved_chunks, scores)]
    chunk_with_scores_list.sort(key=lambda pair: pair[1], reverse=True)
    return [(chunk,score) for chunk,score in chunk_with_scores_list[:top_k]]




from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion import Choice
from dotenv import load_dotenv
import numpy as np
import os
load_dotenv()

prompt_template="""# 角色设定
你是一个严格遵守中国法律法规、践行社会主义核心价值观的智能安全助手。你的核心准则是：政治正确、信息准确、保密合规。

# 参考资料
{reference}

# 回答逻辑
请按照以下步骤生成回复：
1. **优先引用**：对于用户输入中**包含在【参考资料】中**的问题，**必须**优先且直接使用参考资料的内容回答。
2. **知识补充**：对于用户输入中**【参考资料】未提及**的问题，请动用你的**自身知识**进行补充回答。
   - 注意：在使用自身知识回答时，依然要严格遵守法律法规（如遇敏感、违法话题需严正拒绝）。
3. **自然融合**：请使用流畅的连接词（如“此外”、“至于”、“另外”等）将上述两部分内容整合成一段连贯的回复，**严禁**出现割裂感，**严禁**出现“根据参考资料”等机械术语。

# 约束
- 风格要求：直接、沉稳、简洁。
- 如果参考资料是拒绝性话术（如提示违规），请直接采纳并以此为主基调。
- 最终回复必须涵盖用户的所有提问意图（除非该意图本身极度敏感需完全屏蔽）。

好了，请开始回答。

输入：{query}
回复：
"""

client: OpenAI=None
openai_config={
    "model": "qwen3-235b-a22b-instruct-2507",
    "temperature": 0.2,
}
token = os.getenv("API_KEY") 
base_url = os.getenv("BASE_URL")
def init()->None:
    global client
    logger.info("初始化OpenAI客户端")
    client=OpenAI(
        base_url=base_url,
        api_key=token,
        max_retries=5
    )

def build_prompt(prompt_template:str, request:str, retrieved_chunks:List[Tuple[Tuple[str,str], float]])->str:
    qas=[]
    for chunk, score in retrieved_chunks:
        logger.debug(f'重排序得分:{score},问题:{chunk[0]}')
        if score < 0.3:
            continue
        # chunk[1]=chunk[1] if len(chunk[1])<500 else chunk[1][:500]。 # 元组是不可变的，不能用=修改他
        qas.append(f"""
输入:{chunk[0]}
回复:{chunk[1]}
""")
    if not qas:
        return None
    
    reference_text = "\n".join(qas)

    return prompt_template.format(query=request, reference=reference_text)


def invoke_llm(client:OpenAI, messages, config:dict)->str:
    chat_completion:ChatCompletion=client.chat.completions.create(
        messages=messages,
        **config
    )
    if not chat_completion.choices:
        return None
    choice:Choice=chat_completion.choices[0]
    if not choice.message:
        return None
    message:ChatCompletionMessage=choice.message
    if not message.content:
        return None
    else:
        return message.content

def cal_cosine(answer:str, reference:str)->float:
    """ 待优化加速"""
    logger.debug('计算答案与参考内容的相似度')
    answer_emb=embedding_model.encode(answer, normalize_embeddings=True)
    reference_emb=embedding_model.encode(reference, normalize_embeddings=True)
    return np.dot(answer_emb, reference_emb)

def get_guardrail_response(request:str)->str:
    start_time=time.time()
    scored_chunks=rerank(query=request, retrieved_chunks=retrieve(request, top_k=5), top_k=1)
    end_time=time.time()
    logger.debug(f'检索与重排序总耗时:{end_time - start_time:.2f}秒')
    if not scored_chunks:
        return "未在数据库中检索到相关信息"
    higest_score_chunk=scored_chunks[0] # 分最高的那一条数据
    if higest_score_chunk[1] > 0.95:
        logger.info("直接返回答案，不调用大模型")
        return higest_score_chunk[0][1] # 直接返回答案
    
    prompt:str=build_prompt(prompt_template, request, scored_chunks)
    if prompt is None:
        return "未检索到高于阈值的相关信息，无法生成回答。"
    else:
        messages=[
           {"role":"user", "content":prompt} 
        ]

        start_time=time.time()
        answer=invoke_llm(client, messages, config=openai_config)
        end_time=time.time()
        logger.debug(f'调用大模型总耗时:{end_time - start_time:.2f}秒')

        answer_backup=''
        similar=cal_cosine(answer, higest_score_chunk[0][1])
        logger.debug(f'答案与最高分片段相似度：{similar}')
        if similar < 0.5:
            logger.info("模型生成内容与参考内容相似度过低")
            answer_backup=answer
            answer=''
        
        return answer 


def test_rag():
    query='迷药有哪些？'
    chunks=split_into_chunks('test.json')
    embeddings=[embed_chunk(chunk[0]) for chunk in chunks]
    save_embedding(chunks, embeddings)

    retrieved_chunks=retrieve(query, top_k=5)
    for i, chunk in enumerate(retrieved_chunks):
        print(f'--- chunk {i} ---')
        print(f'question:{chunk[0]} answer:{chunk[1]}')

    print('\n')
    rerank_chunks=rerank(query=query, retrieved_chunks=retrieved_chunks, top_k=3)
    print('====== rerank results ======')
    for i, chunk in enumerate(rerank_chunks):
        print(f'--- chunk {i} ---')
        print(f"score:{chunk[1]}, question:{chunk[0][0]} answer:{chunk[0][1]}")
    

if __name__=='__main__':
    query='迷药有哪些类型？'
    init()

    start_time=time.time()
    chunks = split_into_chunks('test.json')
    embeddings = [embed_chunk(chunk[0]) for chunk in chunks] 
    save_embedding(chunks, embeddings)
    end_time=time.time()
    logger.debug(f'向量数据库准备总耗时:{end_time - start_time:.2f}秒')

    output=get_guardrail_response(query)
    logger.info(f'Final answer: {output}')