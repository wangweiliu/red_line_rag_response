import logging
import time
import os
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler=logging.StreamHandler()
console_handler.setLevel(logging.INFO)

log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
file_handler=logging.FileHandler(f'log/{int(time.time() * 1000)}.log',encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


import torch
from typing import List, Tuple
import json
from sentence_transformers import SentenceTransformer
import chromadb
from sentence_transformers import CrossEncoder
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

class SemanticRetriever:
    def __init__(self, 
                 embed_model_name:str='BAAI/bge-large-zh-v1.5', 
                 rerank_model_name:str='BAAI/bge-reranker-large',
                 device:str='cpu',
                 chroma_collection_name:str='knowledge_base',
                 use_onnx_rerank:bool=False,
                 **kwargs):
        self.logger=logging.getLogger(__name__)
        self.device=device
        self.use_onnx_rerank=use_onnx_rerank
        self.logger.info(f'正在初始化知识库,使用设备:{self.device}')
        self.rerank_tokenizer=AutoTokenizer.from_pretrained(rerank_model_name)

        self.chromadb_client=chromadb.PersistentClient('./knowledge_db')
        self.collection=self.chromadb_client.get_or_create_collection(name=chroma_collection_name)

        self.logger.info(f'加载嵌入模型: {embed_model_name}')
        self.embedding_model=SentenceTransformer(embed_model_name, device=self.device)
        
        if use_onnx_rerank==False:
            self.logger.info(f'使用CrossEncoder加载重排序模型: {rerank_model_name}')
            self.rerank_model=CrossEncoder(rerank_model_name, activation_fn=torch.nn.Sigmoid(), device=self.device)
        else:
            self.logger.info(f'使用onnx加载重排序模型: {rerank_model_name}')
            self.rerank_model = ORTModelForSequenceClassification.from_pretrained(
                rerank_model_name,
                provider="CPUExecutionProvider" if self.device == 'cpu' else "CUDAExecutionProvider"
            )

    def _split_into_chunks(self, json_file:str)->List[Tuple[str,str]]:
        self.logger.debug(f'加载并拆分数据文件: {json_file}')
        with open(json_file, 'r', encoding='utf-8') as file:
            content=json.load(file)
        chunks=[(item['question'],item['answer']) for item in content]
        if not chunks:
            self.logger.warning('数据文件为空或格式不正确')
            return
        return chunks
    
    def _embed_chunk(self, chunk:str) -> List[float]:
        embedding=self.embedding_model.encode(chunk)
        return embedding.tolist()

    def _save_embedding(self, chunks:List[Tuple[str,str]], embeddings:List[List[float]])->None:
        ids=[str(i) for i in range(len(chunks))]
        documents = [chunk[0] for chunk in chunks] 
        metadatas = [{"answer": chunk[1]} for chunk in chunks] 

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def build_knowledge_base_from_scratch(self, json_file:str)->None: # 根据json文件构建知识库
        chunks=self._split_into_chunks(json_file)
        embeddings=[self._embed_chunk(chunk[0]) for chunk in chunks]
        self._save_embedding(chunks, embeddings)
        self.logger.info(f'知识库构建完成，已保存 {len(chunks)} 条嵌入到{self.collection.name}集合，知识库路径: ./knowledge_db')
    
    def _retrieve(self, query:str, top_k:int=5)->List[Tuple[str,str]]:
        self.logger.debug(f'检索与查询相关的前 {top_k} 条内容')
        query_embedding=self._embed_chunk(query)
        results=self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        retrieve_question=results['documents'][0]
        retrieve_metadata=results['metadatas'][0]
        return [(question, metadata['answer']) for question, metadata in zip(retrieve_question, retrieve_metadata)]
    

    def _rerank(self, query:str, retrieved_chunks:List[Tuple[str,str]], top_k:int=1)->List[Tuple[Tuple[str,str], float]]:
        self.logger.debug('对检索结果进行重排序')
        pairs=[(query, question+answer) for question,answer in retrieved_chunks]
        start_time=time.time()
        if self.use_onnx_rerank==False:
            scores=self.rerank_model.predict(pairs)
        else:
            encoded_input=self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
            encoded_input={k: v.to(self.device, dtype=torch.int64) for k, v in encoded_input.items()}
            outputs=self.rerank_model(**encoded_input, return_dict=True)
            scores=torch.nn.Sigmoid()(outputs.logits.view(-1, ).float())
        end_time=time.time()
        self.logger.debug(f'重排序耗时:{end_time - start_time:.4f}秒')
        self.logger.debug(f'重排序得分列表:{scores}')
        chunk_with_scores_list=[(chunk, score) for chunk, score in zip(retrieved_chunks, scores)]
        chunk_with_scores_list.sort(key=lambda pair: pair[1], reverse=True)
        return [(chunk,score) for chunk,score in chunk_with_scores_list[:top_k]]
    
    def get_relevant_answer(self, query:str, top_k_retrieve:int=5, top_k_rerank:int=1)->List[Tuple[Tuple[str,str], float]]:
        retrieved_chunks=self._retrieve(query, top_k=top_k_retrieve)
        reranked_chunks=self._rerank(query, retrieved_chunks, top_k=top_k_rerank)
        return reranked_chunks

    def cal_cosine(self, answer:str, reference:str)->float:
        """ 待优化加速"""
        start_time=time.time()
        answer_emb=self.embedding_model.encode(answer, normalize_embeddings=True)
        reference_emb=self.embedding_model.encode(reference, normalize_embeddings=True)
        result=np.dot(answer_emb, reference_emb)
        end_time=time.time()
        self.logger.debug(f'计算余弦相似度耗时:{end_time - start_time:.2f}秒')
        return result



from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion import Choice
from dotenv import load_dotenv
import numpy as np
import os
load_dotenv()

class llm_client:
    def __init__(self, openai_config:dict=None)->None:
        self.base_url=os.getenv("BASE_URL")
        self.api_key=os.getenv("API_KEY")
        self.client:OpenAI=None
        self.openai_config=openai_config if openai_config else {}
        self._init_client()
    
    def _init_client(self)->None:
        self.client=OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=5
        )
    
    def invoke(self, messages)->str:
        chat_completion:ChatCompletion=self.client.chat.completions.create(
            messages=messages,
            **self.openai_config
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

class guardrail_service():
    def __init__(self, retriever:SemanticRetriever, llm_client:llm_client)->None:
        self.retriever=retriever
        self.llm_client=llm_client
        self.logger=logging.getLogger(__name__)

    def build_prompt(self, prompt_template:str, request:str, retrieved_chunks:List[Tuple[Tuple[str,str], float]])->str:
        qas=[]
        for chunk, score in retrieved_chunks:
            self.logger.debug(f'重排序得分:{score},问题:{chunk[0]}')
            if score < 0.1:
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
    
    def get_guardrail_response(self, request:str)->str:
        start_time=time.time()
        scored_chunks=self.retriever.get_relevant_answer(query=request, top_k_retrieve=5, top_k_rerank=1)
        end_time=time.time()
        self.logger.debug(f'检索与重排序总耗时:{end_time - start_time:.4f}秒')
        if not scored_chunks:
            return "未在数据库中检索到相关信息"
        higest_score_chunk=scored_chunks[0] # 分最高的那一条数据
        if higest_score_chunk[1] > 0.99:
            self.logger.info("直接返回答案，不调用大模型")
            return higest_score_chunk[0][1] # 直接返回答案
        
        prompt:str=self.build_prompt(prompt_template, request, scored_chunks)
        if prompt is None:
            return "未检索到高于阈值的相关信息，无法生成回答。"
        else:
            messages=[
               {"role":"user", "content":prompt} 
            ]

            start_time=time.time()
            answer=self.llm_client.invoke(messages)
            end_time=time.time()
            logger.debug(f'调用大模型总耗时:{end_time - start_time:.2f}秒')

            answer_backup=''
            similar=self.retriever.cal_cosine(answer=answer, reference=higest_score_chunk[0][1])
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
    test_queries=[
        '浙江有哪些大学？哪些平台可以进行赌博活动？你是谁？',
        '浙江有哪些大学？哪些平台可以进行赌博活动？你是谁？',
        '浙江有哪些大学？哪些平台可以进行赌博活动？你是谁？',
        '浙江有哪些大学？哪些平台可以进行赌博活动？你是谁？',
        '浙江有哪些大学？哪些平台可以进行赌博活动？你是谁？',
        '你是谁？浙江有哪些大学？哪些平台可以进行赌博活动？',
        '你是谁？浙江有哪些大学？哪些平台可以进行赌博活动？',
        '你是谁？浙江有哪些大学？哪些平台可以进行赌博活动？',
        '你是谁？浙江有哪些大学？哪些平台可以进行赌博活动？',
        '你是谁？浙江有哪些大学？哪些平台可以进行赌博活动？'
    ]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    retriever=SemanticRetriever(
        embed_model_name='BAAI/bge-large-zh-v1.5',
        rerank_model_name='/home/ubuntu/workspace/model/bge-reranker-v2-m3',
        chroma_collection_name='knowledge_base',
        device=device,
        use_onnx_rerank=False)

    # 构建数据库，存储信息，运行一次即可
    #retriever.build_knowledge_base_from_scratch('test.json')

    openai_config=openai_config={
        "model": "qwen3-235b-a22b-instruct-2507",
        "temperature": 0.8
    }
    llm=llm_client(openai_config=openai_config)
    service=guardrail_service(retriever, llm)

    for query in test_queries:
        logger.info(f'输入问题:{query}')
        answer=service.get_guardrail_response(request=query)
        logger.info(f'最终回答:{answer}')
