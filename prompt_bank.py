from ollama import chat
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from functools import partial
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from tool_config import tool_description

MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", "24"))
api_version = "2024-12-01-preview"


class LLM_agent:
    def __init__(self, batch_size, LLM_model = 'phi4', api_id = 0):
        self.batch_size = batch_size
        self.LLM_model = LLM_model
        self.openai_api = False
        self.azure_openai_api = False
        self.azure_inference_api = False
        self.responses_api = False
        self.request_timeout = float(os.environ.get("OPENAI_TIMEOUT", "60"))
        self.azure_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", LLM_model)
        self.azure_inference_model = os.environ.get("AZURE_AI_INFERENCE_MODEL", "Phi-4")
        del api_id

        if LLM_model == 'phi4api':
            endpoint = os.environ.get("AZURE_AI_INFERENCE_ENDPOINT")
            api_key = os.environ.get("AZURE_AI_INFERENCE_KEY")
            if not endpoint or not api_key:
                raise ValueError("phi4api requires AZURE_AI_INFERENCE_ENDPOINT and AZURE_AI_INFERENCE_KEY.")
            self.client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
            self.azure_inference_api = True
        elif os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"):
            self.client = AzureOpenAI(
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", api_version),
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                timeout=self.request_timeout,
                max_retries=1,
            )
            self.azure_openai_api = True
        elif LLM_model != 'phi4':
            client_kwargs = {
                "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
                "timeout": self.request_timeout,
                "max_retries": 1,
            }
            if os.environ.get("OPENAI_BASE_URL"):
                client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
            self.client = OpenAI(**client_kwargs)
            self.openai_api = True
            self.responses_api = os.environ.get("OPENAI_USE_RESPONSES", "0") == "1" or LLM_model.startswith("gpt-5.6")
                


    def agent_model(self, instruction, mode = 're'):
        if mode == 're':
            system_content = "Evaluate recommendation tools for the item ranking task."
        elif mode == 'ps':
            system_content = "Summarize the user's preference. Do not use bullet points or numbered lists."
        elif mode == 'rc':
            system_content = "Recall and rerank items from candidate lists"
        elif mode == 'llmrank':
            system_content = "Predict the next item based on user profile and purchase history."
        try:
            if self.LLM_model == 'phi4':
                response = chat(model=self.LLM_model, 
                                messages=[
                                    {"role": "system", "content": system_content},
                                    {'role': 'user','content': instruction},],
                                options={"no_history": True})
                return response['message']['content']
            elif self.azure_inference_api:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_content),
                        UserMessage(content=instruction)
                    ],
                    max_tokens=2048,
                    model=self.azure_inference_model)
                return response.choices[0].message.content           
            elif self.responses_api and hasattr(self.client, "responses"):
                response = self.client.responses.create(
                    model=self.LLM_model,
                    input=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": instruction},
                    ],
                    max_output_tokens=2048,
                    store=False,
                    timeout=self.request_timeout,
                )
                if getattr(response, "output_text", None):
                    return response.output_text
                for output in getattr(response, "output", []):
                    for content in getattr(output, "content", []):
                        if getattr(content, "type", None) == "output_text":
                            return content.text
                return str(response)
            elif self.azure_openai_api:
                response = self.client.chat.completions.create(
                    stream=False,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": instruction},
                    ],
                    max_tokens=2048,
                    model=self.azure_model,
                )
                return response.choices[0].message.content
            elif self.openai_api:
                response = self.client.chat.completions.create(
                    stream=False,
                    messages=[
                        {
                            "role": "system",
                            "content": system_content,
                        },
                        {
                            "role": "user",
                            "content": instruction,
                        }
                    ],
                    max_tokens=2048,
                    model=self.LLM_model,
                )
                return response.choices[0].message.content
        except Exception as e:
            return str(e)



    def parallel_ranking_evaluation(self, instruction_list, mode = 're'):
        """
        Executes ranking_evaluation_model in parallel for a batch of instructions.
        
        Args:
            instruction_list (list): A list of instruction strings (length = batch_size).
        
        Returns:
            List of responses from ranking_evaluation_model.
        """
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            agent_model = partial(self.agent_model, mode = mode)
            results = list(executor.map(agent_model, instruction_list))
        if mode == 're':
            results = np.array(results).reshape(self.batch_size, len(instruction_list)//self.batch_size)
        return results
    
    def user_profile_summary_model(self, item_descripitions):
        
        instruction = f"""[Instruction]
        Summarize the user's preference in THREE~FIVE sentences based on descriptions of items he clicked.

        [Item Descrition List]
        {item_descripitions}
        """
        return self.agent_model(instruction, mode='ps')
        
    def generate_user_profile_parallel(self, batch_item_descripitions):
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            batch_user_profile = list(executor.map(self.user_profile_summary_model, batch_item_descripitions))
        return batch_user_profile
    
    def llmrank_item_predict(self, item_descripitions, user_profile, item_type):
        if user_profile:
            instruction = f"""[Instruction]
            Predict the next {item_type} the user prefers according to his user profile and the descriptions of items he purchased before.

            [Item Descrition List]
            {item_descripitions}

            [User Profile]
            {user_profile}
            """
        else:
            instruction = f"""[Instruction]
            Predict the next {item_type} the user prefers according to the descriptions of items he purchased before.

            [Item Descrition List]
            {item_descripitions}
            """
        return self.agent_model(instruction, mode='llmrank')
    def llmrank_item_predict_parallel(self, batch_user_profile, batch_item_descripitions, item_type):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            llmrank_item_predict = partial(self.llmrank_item_predict, item_type = item_type)
            batch_predicted_items = list(executor.map(llmrank_item_predict, batch_item_descripitions, batch_user_profile))
        return batch_predicted_items
    
class PromptBank:
    def __init__(self, item_num, top_k = 10):
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'
        self.top_k = top_k
        self.item_num = item_num
        self.rec_tool_info = {}
    def rec_tool_evaluation(self, user_profile, predicted_ranking, item_description, rec_tool):
        prompt = f"""[Instruction]
        According to the predicted item ranking of the recommendation tool, evaluate if this recommendation tool is suitable for the user. 
        
        [User Profile]
        {user_profile}
        
        [Item Description]
        {item_description}

        [Predicted Item Ranking]
        {predicted_ranking} out of {self.item_num}
        
        [Recommendation tool]
        {self.rec_tool_info.get(rec_tool, tool_description(rec_tool))}

        DO NOT output any explanation or reasoning.
        The output must be ONE SINGLE WORD in [{self.BOS}POOR{self.EOS}, {self.BOS}FAIR{self.EOS}, {self.BOS}GOOD{self.EOS}, {self.BOS}GREAT{self.EOS}].
        """
        return prompt

    def get_rec_tool_evaluation_prompt(self, batch_user_profiles, batch_item_descriptions, batch_predicted_ranking_tool_pairs):
        """
        Parallel execution of rec_tool_evaluation for multiple user-item pairs with different ranking-tool pairs.
        
        batch_user_profiles: List of batch_size user profiles (one per user-item pair)
        batch_item_descriptions: List of batch_size item descriptions (one per user-item pair)
        batch_predicted_ranking_tool_pairs: List of 3 (predicted_ranking, rec_tool) pairs per user-item pair
        """
        tasks = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for user_profile, item_description, ranking_tool_pairs in zip(batch_user_profiles, batch_item_descriptions, batch_predicted_ranking_tool_pairs):
                for predicted_ranking, rec_tool in ranking_tool_pairs:
                    tasks.append(
                        executor.submit(self.rec_tool_evaluation, user_profile, predicted_ranking, item_description, rec_tool)
                    )
            
            # Gather all results
            results = [task.result() for task in tasks]
        
        return results
    
    def recall_item_comparison(self, item_desc, rec_top_item_desc):
        labels = [chr(ord('A') + idx) for idx in range(len(rec_top_item_desc))]
        group_blocks = "\n\n".join(
            f"[Group {label} Descriptions]\n{descriptions}"
            for label, descriptions in zip(labels, rec_top_item_desc)
        )
        options = ", ".join(f"{self.BOS}{label}{self.EOS}" for label in labels)
        prompt = f"""[Instruction]
        According to the descriptions of items in the candidate groups, evaluate which group this target item is most relevant to.

        {group_blocks}

        [Target Item Description]
        {item_desc}


        DO NOT output any explanation or reasoning.
        The output must be ONE SINGLE CHARACTER in [{options}].
        """
        return prompt

    def get_recall_item_comparison_prompt(self, batch_item_desc, batch_rec_top_item_desc): 
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(self.recall_item_comparison, batch_item_desc, batch_rec_top_item_desc))
        
        return results

    def item_relavant_evaluation(self, user_profile, item_desc, rec_top_item_desc):
        prompt = f"""[Instruction]
        According to the user profile and recalled item descriptions of this user, evaluate if this target item is relevant with this user. 
        
        [User Profile]
        {user_profile}
        
        [Recalled Item Descriptions]
        {rec_top_item_desc}

        [Target Item Description]
        {item_desc}


        DO NOT output any explanation or reasoning.
        The output must be ONE SINGLE WORD in [{self.BOS}IRRELEVANT{self.EOS}, {self.BOS}NEUTRAL{self.EOS}, {self.BOS}RELEVANT{self.EOS}].
        """
        return prompt
    
    def get_item_relavant_evaluation_prompt(self, batch_user_profiles, batch_item_desc, batch_rec_top_item_desc):

        tasks = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for user_profile, item_description, all_rec_top_item_desc in zip(batch_user_profiles, batch_item_desc, batch_rec_top_item_desc):
                for rec_top_item_desc in all_rec_top_item_desc:
                    tasks.append(
                        executor.submit(self.item_relavant_evaluation, user_profile, item_description, rec_top_item_desc)
                    )
            
            # Gather all results
            results = [task.result() for task in tasks]
        
        return results
    
    def rerank_candidate(self, user_profile, item_desc, candidate_desc):
        prompt = f"""[Instruction]
        According to the user profile and his historical item descriptions, rank top-{self.top_k} items this user may prefer from the candidate item list. 
        
        [User Profile]
        {user_profile}
        
        [Historical Item Descriptions]
        {item_desc}

        [Candidate Item List in format of (ID, description)]
        {candidate_desc}

        The output MUST be a list of candidate item IDs with length of {self.top_k}, with items seperated by lines. DO NOT output any explanation or reasoning.
        """
        return prompt
    
    def get_reranked_item_prompt(self, batch_user_profiles, batch_item_desc, batch_candidate_desc):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(self.rerank_candidate, batch_user_profiles, batch_item_desc, batch_candidate_desc))
        
        return results
