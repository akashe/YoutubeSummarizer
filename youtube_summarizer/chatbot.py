import pdb
import json
import asyncio

import streamlit as st
from openai import OpenAI
import redirect as rd

import time
import os
from copy import deepcopy

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

from process_videos import process_videos
from process_channels import process_channels
from process_clips import create_clips_for_video
from process_single_transcript import process_single_transcript

import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval

from utils import is_valid_openai_api_key, ui_spacer, process_html_string, get_usage_in_dollars, get_refresh_time, \
    chars_processed_dict_for_failed_cases_with_some_processing, \
    chars_processed_dict_for_failed_cases_with_no_processing

from function_definitions import function_definitions

from db import *

daily_community_dollars = 0.5

# Initialize the db if not done yet
create_db_and_table()

with open("youtube_summarizer/html_code_default_play.html","r") as f:
    html_code_default_play = f.read()

with open("youtube_summarizer/html_code_default_pause.html","r") as f:
    html_code_default_pause = f.read()

def check_run_status(run_to_check, thread_id):
    while run_to_check.status == "queued" or run_to_check.status == "in_progress":
        time.sleep(1)
        run_to_check = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_to_check.id
        )

    return run_to_check


def return_assistant_messages(run_to_check, thread_id):

    total_input_text_len = 0
    total_output_text_len = 0

    messages = client.beta.threads.messages.list(
        thread_id=thread_id
    )
    for m in messages:
        content = m.content[0].text.value
        if m.run_id == run_to_check.id and m.role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)
            # logger.info(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
            total_output_text_len += len(content)
        else:
            total_input_text_len += len(content)
    if st.session_state.using_community_tokens:
        input_tokens_processed = int(total_input_text_len / 4)
        output_tokens_processed = int(total_output_text_len / 4)
        save_or_update_tokens(input_tokens_processed, output_tokens_processed)


available_functions = {
        "process_channels": process_channels,
        "process_videos": process_videos,
        "create_clips_for_video": create_clips_for_video,
        "process_single_transcript": process_single_transcript
    }

possible_errors = ["The sought topics are not discussed in the video",
                   "Transcripts not available"]

st.set_page_config(
    page_title="YoutubeBuddy",
    # page_icon="icon.png",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if 'processing' not in st.session_state:
    st.session_state.processing = False

if 'welcome_message_shown' not in st.session_state:
    st.session_state.welcome_message_shown = False

if 'using_community_tokens' not in st.session_state:
    st.session_state.using_community_tokens = False

if 'percent_community_token' not in st.session_state:
    st.session_state.percent_community_token = 0

screen_width = streamlit_js_eval(js_expressions='screen.width',
                                 want_output=True,
                                 key='SCR')
if screen_width is None:
    component_dim = 410
elif screen_width <= 780:
    component_dim = 205
elif screen_width <= 1024:
    component_dim = 310
else:
    component_dim = 410


with st.sidebar:
    st.markdown(f"""
    ## YouTube Buddy
    """)
    st.write("Made by [Akash Kumar](https://www.linkedin.com/in/akashkumar2/).", unsafe_allow_html=True)
    st.markdown('Do share your [feedback](https://f7f6zk74dit.typeform.com/to/kYw7Y8y7).')

st.subheader("YouTube Buddy: Streamline Your YouTube Experience")

ui_spacer(2)

if not st.session_state.welcome_message_shown:
    with st.chat_message("assistant"):
        msg = "👋 Welcome to YouTube Buddy!\n\n"\
                "Try 'Summarize this youtube video for me [url]'\n\n" \
                "Summarize all the videos released by this channel in past 2 weeks\n\n" \
                "Summarize what was discussed about Trump in this video [url] \n\n" \
                "How demand and supply shape economics as discussed in [url] \n\n" \
                "Can you shorten this podcast and create clips for me around this topic? \n\n" \
                "If you like buddy consider [buying me a coffee](https://buymeacoffee.com/akashe) 🤗"
        st.write(msg)

community_openai_api_key = None
personal_openai_api_key = None
openai_api_key = None

t1,t2 = st.tabs(['Community version','Enter your own API key'])
with t1:
    input_tokens_used, output_token_used = get_today_token_usage()
    logger.info(f"Input tokens used till now: {input_tokens_used}")
    logger.info(f"Output tokens used till now: {output_token_used}")
    dollars_used = get_usage_in_dollars(input_tokens_used, output_token_used)
    pct = dollars_used/daily_community_dollars*100
    if pct<=5:
        pct=5
    st.session_state.percent_community_token = pct
    st.write(f'Community tokens used: :{"green" if pct else "red"}[{int(pct)}%]')
    st.progress((pct if pct <=100 else 100)/100)
    if pct < 100.0:
        community_openai_api_key = st.secrets["openai_api_key"]
        #st.write(
        #    'Please consider using your own API key for helping the community.')
    if pct > 100.0:
        st.write(
            f'Community tokens over for today. Refresh in: {get_refresh_time()}. '
            f'Get your own OpenAI API key [here](https://platform.openai.com/account/api-keys)')


with t2:
    personal_openai_api_key = st.text_input('OpenAI API key', type='password', value=None)

if personal_openai_api_key is not None:
    openai_api_key = personal_openai_api_key
    st.session_state.using_community_tokens = False

if st.session_state.percent_community_token < 100.0 and personal_openai_api_key is None:
    openai_api_key = community_openai_api_key
    st.session_state.using_community_tokens = True


if openai_api_key and is_valid_openai_api_key(openai_api_key):

    os.environ["OPENAI_API_KEY"]=openai_api_key
    client = OpenAI(
        api_key=openai_api_key
    )

    assistant = client.beta.assistants.create(
        name="Youtube Assistant",
        description="You an eager helpful assistant. You can help people in answering"
                    "questions they might have about a youtube video by getting the transcript of the video and"
                    "using the transcript to answer the question asked. Don't use summary to answer a question, use"
                    "returned transcripts. Whenever asked to summarize videos use the inbuilt function to process video"
                    "and not rely on returned transcripts to generate summaries"
                    ,
        model="gpt-4o-mini",
        tools=function_definitions
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if not message.get("html_code"):
                st.markdown(message["content"])
            else:
                html = message["content"]
                components.html(html, height=component_dim)

    if st.session_state.thread_id is None:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

    if not st.session_state.welcome_message_shown:
        with st.chat_message("assistant"):
            msg = "👋 Welcome to YouTube Buddy!\n\n"\
                  "Try 'Summarize this youtube video for me [url]'\n\n" \
                  "Summarize all the videos released by this channel in past 2 weeks\n\n" \
                  "Summarize what was discussed about Trump in this video [url] \n\n" \
                  "How demand and supply shape economics as discussed in [url] \n\n" \
                  "Can you shorten this podcast and create clips for me around this topic? \n\n" \
                  "If you like buddy consider [buying me a coffee](https://buymeacoffee.com/akashe) 🤗"

            st.write(msg)

        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.session_state.welcome_message_shown = True

    prompt = st.chat_input("Enter prompt")

    def prompt_processing(prompt):
        st.session_state.processing = True

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=prompt
        )

        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=assistant.id)

        run = check_run_status(run, st.session_state.thread_id)

        if run.status in ["cancelled", "failed", "cancelling", "expired"]:
            with st.chat_message("assistant"):
                st.markdown("I think my servers are having a fever. Can you retry again?")

        if run.status == "requires_action":
            required_action = run.required_action

            tool_calls = required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                tool_call_id = tool_call.id
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                try:
                    if function_name != "process_single_transcript":
                        # all other functions other than process_single_transcript print results to create html
                        # elements and need separate code to decide what to store in chat context
                        with st.chat_message("assistant"):
                            with rd.stdout(to=st.empty(), format="markdown"):
                                chars_processed, function_response = asyncio.run(
                                    function_to_call(
                                        **function_args
                                    )
                                )

                                if function_name == "create_clips_for_video" and \
                                        function_response not in possible_errors:
                                    new_html_code = deepcopy(html_code_default_play).replace('{{VIDEOS_JSON}}',
                                                                                              function_response)
                                    new_html_code = process_html_string(new_html_code)
                                    components.html(new_html_code, height=component_dim)

                        if function_name != "create_clips_for_video":
                            st.session_state.messages.append({"role": "assistant", "content": function_response})
                        if function_name == "create_clips_for_video" and \
                                function_response in possible_errors:
                            st.session_state.messages.append({"role": "assistant", "content": function_response})
                        if function_name == "create_clips_for_video" and \
                                function_response not in possible_errors:
                            new_html_code = deepcopy(html_code_default_pause).replace('{{VIDEOS_JSON}}',
                                                                                      function_response)
                            new_html_code = process_html_string(new_html_code)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": new_html_code, "html_code": True})
                    else:
                        chars_processed, function_response = function_to_call(
                            **function_args
                        )
                except Exception as e:
                    logger.error(e)
                    function_response = "Something wrong with the request. Please try again :)"
                    chars_processed = chars_processed_dict_for_failed_cases_with_no_processing
                    # TODO: add chars used here
                    with st.chat_message("assistant"):
                        with rd.stdout(to=st.empty(), format="markdown"):
                            print(function_response)
                    st.session_state.messages.append({"role": "assistant", "content": function_response})

                tool_outputs.append({
                    "tool_call_id": tool_call_id,
                    "output": function_response
                })
                #logger.info(function_response)

                # Update token usage using the rough rule of 1 token = 4 chars of english
                if st.session_state.using_community_tokens:
                    input_tokens_processed = int(chars_processed["input_chars"] / 4)
                    output_tokens_processed = int(chars_processed["output_chars"] / 4)
                    save_or_update_tokens(input_tokens_processed, output_tokens_processed)

                # Append the results of tool call
                results_append_run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=st.session_state.thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs)

                results_append_run = check_run_status(results_append_run, st.session_state.thread_id)

                if function_name == "process_single_transcript":
                    return_assistant_messages(results_append_run, st.session_state.thread_id)
        else:
            return_assistant_messages(run, st.session_state.thread_id)

        st.session_state.processing = False

    #####
    #   is the user enters a prompt while another prompt is running then existing execution stops
    #   we create a new thread in which processing happens. This looses the context
    #####
    if prompt and not st.session_state.processing:
        logger.info(prompt)
        prompt_processing(prompt)
    elif prompt:
        logger.info(prompt)
        with st.chat_message("assistant"):
            msg = "Execution stopped! Please wait till processing is complete.\n\n"\
                  "Check the 'Running' tab at the top!"
            st.write(msg)

        st.session_state.messages.append({"role": "assistant", "content": msg})

        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

        prompt_processing(prompt)


else:
    st.error("Please enter a valid OpenAI key in the 'Enter your own API key' tab to continue.")

ui_spacer(2)
