import json
import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Any, Callable, Generator, List, Optional, Type, Union

import pandas
import plotly
import streamlit
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import StructuredTool, Tool
from PIL import Image

from financial_insights.src.function_calling import (ConversationalResponse,
                                                     FunctionCallingLlm)
from financial_insights.src.tools import get_conversational_response
from financial_insights.src.tools_filings import retrieve_filings
from financial_insights.src.tools_stocks import (get_financial_summary,
                                                 get_historical_price,
                                                 get_stock_info,
                                                 retrieve_symbol_list,
                                                 retrieve_symbol_quantity_list)
from financial_insights.src.tools_yahoo_news import scrape_yahoo_finance_news

logging.basicConfig(level=logging.INFO)
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

# tool mapping of available tools
TOOLS = {
    'get_stock_info': get_stock_info,
    'get_historical_price': get_historical_price,
    'retrieve_symbol_list': retrieve_symbol_list,
    'retrieve_symbol_quantity_list': retrieve_symbol_quantity_list,
    'scrape_yahoo_finance_news': scrape_yahoo_finance_news,
    'get_financial_summary': get_financial_summary,
    'get_conversational_response': get_conversational_response,
    'retrieve_filings': retrieve_filings,
}

TEMP_DIR = 'financial_insights/streamlit/cache/'
SOURCE_DIR = 'financial_insights/streamlit/cache/sources/'
CONFIG_PATH = 'financial_insights/config.yaml'


# @contextmanager
# def st_capture(output_func: Callable[[Any], None]) -> None:
#     """
#     context manager to catch stdout and send it to an output streamlit element

#     Args:
#         output_func (function to write terminal output in

#     Yields:
#         Generator:
#     """
#     with StringIO() as stdout, redirect_stdout(stdout):
#         old_write = stdout.write

#         def new_write(string: str) -> int:
#             ret = old_write(string)
#             output_func(stdout.getvalue())
#             return ret

#         stdout.write = new_write
#         yield


@contextmanager
def st_capture(output_func: Callable[[Any], Any]) -> Generator[None, None, None]:
    """
    Context manager to catch stdout and send it to an output function.

    Args:
        output_func (Callable[[str], None]): Function to write terminal output to.

    Yields:
        None: A generator that redirects stdout to the output_func.
    """
    stdout = StringIO()
    with redirect_stdout(stdout):
        try:
            yield
        finally:
            output_func(stdout.getvalue())


def set_fc_llm(
    tools: List[str],
    default_tool: Optional[Union[StructuredTool, Tool, Type[BaseModel]]] = ConversationalResponse,
) -> None:
    """
    Set the FunctionCallingLlm object with the selected tools

    Args:
        tools (list): list of tools to be used
    """
    set_tools = [TOOLS[name] for name in tools]
    streamlit.session_state.fc = FunctionCallingLlm(tools=set_tools, default_tool=default_tool)


def handle_userinput(user_question: Optional[str], user_query: Optional[str]) -> Optional[Any]:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.
    """
    output = streamlit.empty()

    with streamlit.spinner('Processing...'):
        with st_capture(output.code):
            tool_messages, response = streamlit.session_state.fc.function_call_llm(
                query=user_query,
                max_it=streamlit.session_state.max_iterations,
                debug=True,
            )

    streamlit.session_state.chat_history.append(user_question)
    streamlit.session_state.chat_history.append(response)

    with streamlit.chat_message('user'):
        streamlit.write(f'{user_question}')

    stream_response_object(response)

    return response


def stream_chat_history() -> None:
    for ques, ans in zip(
        streamlit.session_state.chat_history[::2],
        streamlit.session_state.chat_history[1::2],
    ):
        with streamlit.chat_message('user'):
            streamlit.write(ques)

        stream_response_object(ans)


def stream_response_object(response: Any) -> Any:
    # Convert JSON string to dictionary
    try:
        # Try to convert the string to a dictionary
        response_dict = json.loads(response)
        # Check if the result is a dictionary
        if isinstance(response_dict, dict):
            response = response_dict

    except (json.JSONDecodeError, TypeError):
        # If JSON decoding fails, return the original string
        pass

    if isinstance(response, str):
        stream_single_response(response)

    elif isinstance(response, list):
        for item in response:
            stream_single_response(item)

    elif isinstance(response, dict):
        for key, value in response.items():
            stream_single_response(value)

    elif isinstance(response, tuple):
        for element in response:
            if isinstance(element, str):
                stream_single_response(element)
            elif isinstance(element, list):
                for item in element:
                    stream_single_response(item)
            elif isinstance(element, dict):
                for key, value in element.items():
                    stream_single_response(value)
    else:
        raise Exception('Invalid response type')


def stream_single_response(response: Any) -> None:
    """Streamlit chatbot response."""
    # If response is a string
    if isinstance(response, str):
        with streamlit.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            if not response.endswith('.png'):
                streamlit.write(response)
            else:
                # Load the image
                image = Image.open(response)

                # Display the image
                streamlit.image(image, use_column_width=True)
    # If response is a figure
    elif isinstance(response, plotly.graph_objs.Figure):
        # Display the image
        streamlit.image(response, use_column_width=True)

    # If response is a dataframe, display its head
    elif isinstance(response, pandas.DataFrame):
        streamlit.write(response.head())
