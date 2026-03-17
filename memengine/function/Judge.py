from abc import ABC, abstractmethod
import json
import re
from memengine.function.LLM import *
from langchain.prompts import PromptTemplate

class BaseJudge(ABC):
    """
    Assess given observations or intermediate messages on certain aspects.
    """
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LLMJudge(BaseJudge):
    """
    Judge vias large language models.
    """
    def __init__(self, config):
        super().__init__(config)

        self.llm = eval(config.LLM_config.method)(config.LLM_config)

    def __extract_json_obj__(self, text):
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        stack = []
        start_idx = None
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start_idx = i
                stack.append(ch)
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        snippet = text[start_idx:i + 1]
                        try:
                            return json.loads(snippet)
                        except Exception:
                            start_idx = None
                            continue
        return None

    def __parse_scale_score__(self, res):
        default_score = float(getattr(self.config, 'default_score', 5.0))
        text = str(res).strip()

        # Priority 1: strict JSON or embedded JSON object.
        json_obj = self.__extract_json_obj__(text)
        if isinstance(json_obj, dict):
            for key in ('score', 'value'):
                if key in json_obj:
                    try:
                        return float(json_obj[key])
                    except Exception:
                        pass

        # Priority 2: backward-compatible eval parsing with protection.
        try:
            return float(eval(text))
        except Exception:
            pass

        # Priority 3: regex extracts the first number from text.
        number_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if number_match is not None:
            try:
                return float(number_match.group(0))
            except Exception:
                pass

        # Final fallback.
        return default_score

    def __post_scale__(self, res):
        score = self.__parse_scale_score__(res)
        if hasattr(self.config, 'post_scale'):
            return score/self.config.post_scale
        return score
    
    def __post_bool__(self, res):
        if res == 'True':
            return True
        elif res == 'False':
            return False
        else:
            return "LLM Judge Parse Error for Boolean"

    def __call__(self, input_dict, post_process = 'scale'):
        prompt_template = PromptTemplate(
            input_variables=self.config.prompt.input_variables,
            template=self.config.prompt.template
        )
        prompt = prompt_template.format(**input_dict)
        if post_process == 'scale':
            prompt += '\nPlease return STRICT JSON only in one object: {\"score\": <number>}.'
        res = self.llm.fast_run(prompt)

        if post_process == 'scale':
            return self.__post_scale__(res)
        elif post_process == 'bool':
            return self.__post_bool__(res)
        else:
            raise "Judge Post Process Type Error!"
    
