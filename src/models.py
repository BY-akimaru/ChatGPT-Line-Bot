from typing import List, Dict
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent

import requests


class ModelInterface:
    def check_token_valid(self) -> bool:
        pass

    def chat_completions(self, messages: List[Dict], model_engine: str) -> str:
        pass

    def audio_transcriptions(self, file, model_engine: str) -> str:
        pass

    def image_generations(self, prompt: str) -> str:
        pass


class OpenAIModel(ModelInterface):
    def __init__(self, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key
        self.api_key = api_key
        self.base_url = 'https://api.openai.com/v1'
        self.llm = OpenAI(model_name="gpt-3.5-turbo")
        self.tools = load_tools(["google-search"], llm=self.llm)
        self.agent = initialize_agent(self.tools, self.llm, agent="zero-shot-react-description", verbose=True)
        
    def _request(self, method, endpoint, body=None, files=None):
        self.headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        try:
            if method == 'GET':
                r = requests.get(f'{self.base_url}{endpoint}', headers=self.headers)
            elif method == 'POST':
                if body:
                    self.headers['Content-Type'] = 'application/json'
                r = requests.post(f'{self.base_url}{endpoint}', headers=self.headers, json=body, files=files)
            r = r.json()
            if r.get('error'):
                return False, None, r.get('error', {}).get('message')
        except Exception:
            return False, None, 'OpenAI API 系統不穩定，請稍後再試'
        return True, r, None

    def check_token_valid(self):
        return self._request('GET', '/models')

    def chat_completions(self, messages, model_engine) -> str:
        json_body = {
            'model': model_engine,
            'messages': messages
        }
        return self._request('POST', '/chat/completions', body=json_body)

    def audio_transcriptions(self, file_path, model_engine) -> str:
        files = {
            'file': open(file_path, 'rb'),
            'model': (None, model_engine),
        }
        return self._request('POST', '/audio/transcriptions', files=files)

    def image_generations(self, prompt: str) -> str:
        json_body = {
            "prompt": prompt,
            "n": 1,
            "size": "512x512"
        }
        return self._request('POST', '/images/generations', body=json_body)
