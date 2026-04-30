import os
import json
import pandas as pd
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

# --- Structured Output Schemas ---
class TagUpdate(BaseModel):
    """Schema for hierarchical tag generation."""
    tags: List[str] = Field(description="Full list of tags including inherited parent tags and 1-2 new complex tags.")

class QAResponse(BaseModel):
    """Schema for MCQs."""
    question: str = Field(description="The scientific/complex question based on tags.")
    choices: List[str] = Field(description="4 distinct multiple choice answers numbered using digits (0, 1, 2, 3).")
    correct_choice_idx: int = Field(description="Index (0, 1, 2, or 3) of the correct answer.")

# --- The Agent ---
class DatasetAgent:
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self._init_llm()

    def _init_llm(self):
        try:
            self.llm = ChatOpenAI(model=self.model_name, temperature=0.7, openai_api_key=self.api_key)
            self.tag_generator = self.llm.with_structured_output(TagUpdate)
            self.qa_generator = self.llm.with_structured_output(QAResponse)
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI LLM: {e}")
            self.llm = None
            self.tag_generator = None
            self.qa_generator = None

    def update_model(self, model_name: str, api_key: Optional[str] = None):
        if model_name != self.model_name or api_key != self.api_key:
            self.model_name = model_name
            self.api_key = api_key
            self._init_llm()

    def generate_blueprint(self, topic: str, depth: int, breadth: int, model_name: str = None, api_key: str = None) -> pd.DataFrame:
        """
        Phase 1: Build the graph structure with tags.
        """
        if model_name or api_key:
            self.update_model(model_name or self.model_name, api_key)
            
        if not self.tag_generator:
            raise ValueError("OpenAI API key is missing or invalid. Please provide it in the configuration.")
            
        blueprint = []
        
        # 1. Generate Root Tags
        prompt_root = f"Generate a list of 2-3 core conceptual tags for the topic: {topic}."
        root_res = self.tag_generator.invoke(prompt_root)
        root_id = "r1"
        blueprint.append({"id": root_id, "parent_id": None, "tags": root_res.tags})
        
        # 2. Recursive Children
        self._build_depth(blueprint, root_id, root_res.tags, 1, depth, breadth, topic)
        
        # Format tags as strings for UI compatibility
        df = pd.DataFrame(blueprint)
        df['tags'] = df['tags'].apply(lambda x: json.dumps(x))
        return df

    def _build_depth(self, blueprint, parent_id, parent_tags, current_depth, max_depth, breadth, topic):
        if current_depth >= max_depth:
            return
        
        for i in range(breadth):
            child_id = f"{parent_id}_c{i+1}"
            prompt = (
                f"Topic: {topic}\n"
                f"Parent Tags: {parent_tags}\n"
                "Task: Generate a new list of tags that inherits ALL parent tags but adds 1-2 NEW tags "
                "to make the conceptual complexity slightly higher for a sub-question."
            )
            res = self.tag_generator.invoke(prompt)
            blueprint.append({"id": child_id, "parent_id": parent_id, "tags": res.tags})
            self._build_depth(blueprint, child_id, res.tags, current_depth + 1, max_depth, breadth, topic)

    def generate_qa_batch(self, df: pd.DataFrame, model_name: str = None, api_key: str = None) -> pd.DataFrame:
        """
        Phase 2: Generate Questions and Answers based on the blueprint tags.
        """
        if model_name or api_key:
            self.update_model(model_name or self.model_name, api_key)
            
        if not self.qa_generator:
            raise ValueError("OpenAI API key is missing or invalid. Please provide it in the configuration.")
            
        questions = []
        choices_list = []
        correct_indices = []
        
        for _, row in df.iterrows():
            tags = row['tags'] if isinstance(row['tags'], list) else json.loads(row['tags'])
            prompt = (
                f"Create a multiple choice question based on these technical tags: {tags}.\n"
                "Ensure the question is not too challenging but requires understanding of the specific sub-topics."
            )
            res = self.qa_generator.invoke(prompt)
            questions.append(res.question)
            choices_list.append(json.dumps(res.choices))
            correct_indices.append(res.correct_choice_idx)
            
        df['question'] = questions
        df['choices'] = choices_list
        df['correct_idx'] = correct_indices
        return df

import tempfile

# --- File Handling Helpers ---
def save_to_json(df: pd.DataFrame, prefix: str) -> str:
    """
    Saves a DataFrame to a temporary JSON file. 
    NamedTemporaryFile(delete=False) ensures the file persists for Gradio to handle the download.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix=f"{prefix}_") as tmp:
        path = tmp.name
        df.to_json(path, orient="records", indent=4)
    return path

def load_from_json(file_path) -> pd.DataFrame:
    try:
        if hasattr(file_path, 'name'):
            file_path = file_path.name
        return pd.read_json(file_path)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return pd.DataFrame()
