import json
import re
from typing import Dict, List, Sequence, Union

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall,
                                              InitialDeltaToolCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser)
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
import traceback

logger = init_logger(__name__)


class Llama31ToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        
        #self.model_tokenizer = self.model_tokenizer.tokenizer

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent = False
        self.current_tool_initial_sent: bool = False
        self.streamed_args_for_tool: List[str] = [
        ]  # map what has been streamed for each tool so far to a list

        self.tool_call_start_token: str = "<|python_tag|>"
        self.tool_call_end_token: str = "<|eom_id|>"

        #Llama 3.1 is being told to return only JSON but it may or may not prefix with the <|python_tag|> token
        #and it may or may not end with the <|eom_id|> token or the <|eot_id|> token. 
        #I suspect this is a training error because the docs are fairly clear that we should see an eot_id or eom_id token
        #at the end of a function call but the docs themselves have a typo and the model seems to omit these tokens. 
        #So we're just going to give the model the option to output prefix and suffix tokens and if we see 
        #anything that looks like a JSON function call with a name and parameters, wether or not it's wrapped in these tokens,
        #we'll process it. Gotta love the bleeding edge of AI dev.
        self.tool_call_regex = re.compile(
            r"(?:<\|[^\|]+\|>)?(\{.*?[\"\']name[\"\'].*?[\"\']parameters[\"\'].*?{.*?}.*?})(?:<\|[^\|]+\|>)?", re.DOTALL)

        if not self.model_tokenizer: #Set in the superclasses constructor
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

    def extract_tool_calls(self,
                           model_output: str) -> ExtractedToolCallInformation:
        print("TOOL PARSER model_output", model_output)
        # sanity check; avoid unnecessary processing
        
        if(not (first_match := self.tool_call_regex.search(model_output)) ):
            print("No tool calls found")
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        else:
            try:
                print("FIRST MATCH: ", first_match)
                first_match_idx = first_match.start()
                print("FIRST MATCH IDX: ", first_match_idx)
                function_call_tuples = (
                    self.tool_call_regex.findall(model_output)
                    )
                
                raw_function_calls = []
                for match in function_call_tuples:
                    print("MATCH: ", match)
                    dat = json.loads(match)
                    raw_function_calls.append(dat)

                
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(function_call["parameters"])))
                    for function_call in raw_function_calls
                ]

                content = model_output[:first_match_idx] #everything before the first tool call
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None)

            except Exception as e:
                logger.error("Error in extracting tool call from response %s", e)
                logger.error(traceback.format_exc())
                
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        logger.error("Streaming tool calls not implemented for Llama31")
        raise NotImplementedError("Streaming tool calls not implemented for Llama31")
        