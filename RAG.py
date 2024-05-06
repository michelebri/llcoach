from abc import ABC, abstractmethod
import os
from typing import List

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI

from action import Action
from action_embedding import ActionEmbeddings
class RAG:

    def __init__(self, actions_dir):
        
        Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-mpnet-base-v2")
        Settings.chunk_size = 256
        Settings.chunk_overlap = 25
        Settings.llm = None

        # key_file is located in the directory of the main script
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "key.txt")
        # Verify if the file "key.txt exists" else raise an exception
        if not os.path.exists(key_file):
            raise Exception("OpenAI key file not found")

        # Load the openai key from a local file called "key.txt"
        with open(key_file, "r") as file:
            self.openai_key = file.read().strip()

        self.client = OpenAI(api_key=self.openai_key)

        self.actions_index = self.load_actions(actions_dir)

        self.retriever = self.initialize_retriever()
        self.query_engine = self.initialize_query_engine()

    def add_directory_actions_directory(self, directory_path):
        self.action_sources.append(SimpleDirectoryReader(directory_path))

    def load_actions(self, directory) -> VectorStoreIndex:
        action_nodes = []
        for action_file in os.listdir(directory):
            if action_file.endswith(".txt"):
                    action = Action.load_from_file(os.path.join(directory, action_file))

                    # Create a Document instance for each action
                    action_nodes.append(TextNode(text=action.description, id_=action.action_id))
                    
        return VectorStoreIndex(action_nodes)


    def initialize_retriever(self, top_k: int = 3) -> VectorIndexRetriever:
        return VectorIndexRetriever(
            index=self.actions_index,
            similarity_top_k=top_k,
        )

    def initialize_query_engine(self, similarity_cutoff: float = 0.2) -> RetrieverQueryEngine:
        return RetrieverQueryEngine(
            retriever=self.retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
        )


    def extract_usable_actions(self, planning_goal: str, planning_domain: str) -> List[str]:
        action_extraction_query = planning_goal
        usable_actions = self.query_engine.retrieve(action_extraction_query)
        print("AAAAAAAAAAAAAAAA")
        print(usable_actions)
        return usable_actions


    def query_llm(self, system_prompt, request_prompt, temperature=0.9, max_tokens=500, top_p=0.9, frequency_penalty=0.6):

        prompt = request_prompt 

        print("\n##################################\n\n\nSYSTEM PROMPT\n"+system_prompt)
        print("\n##################################\n\n\n\n\nREQUEST PROMPT\n"+request_prompt)

        agent = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty
        )

        return agent.choices[0].message.content
    
