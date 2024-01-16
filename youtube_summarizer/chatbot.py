import pdb
import json
import asyncio

import streamlit as st
from openai import OpenAI
import redirect as rd

import time
import os
from copy import deepcopy

from process_videos import process_videos
from process_channels import process_channels
from process_clips import create_clips_for_video
from process_single_transcript import process_single_transcript

import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval

from utils import is_valid_openai_api_key, ui_spacer, process_html_string
from function_definitions import function_definitions

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
    messages = client.beta.threads.messages.list(
        thread_id=thread_id
    )
    for m in messages:
        if m.run_id == run_to_check.id and m.role == "assistant":
            content = m.content[0].text.value
            with st.chat_message("assistant"):
                st.markdown(content)

            st.session_state.messages.append({"role": "assistant", "content": content})


available_functions = {
        "process_channels": process_channels,
        "process_videos": process_videos,
        "create_clips_for_video": create_clips_for_video,
        "process_single_transcript": process_single_transcript
    }

possible_errors = ["The sought topics are not discussed in the video",
                   "Transcripts not available"]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if 'processing' not in st.session_state:
    st.session_state.processing = False

if 'welcome_message_shown' not in st.session_state:
    st.session_state.welcome_message_shown = False


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
    st.markdown('Source code can be found [here](https://github.com/akashe/YoutubeSummarizer/tree/dev).')

st.subheader("YouTube Buddy: Streamline Your YouTube Experience")


openai_api_key = st.secrets["openai_api_key"]

if openai_api_key and is_valid_openai_api_key(openai_api_key):

    os.environ["OPENAI_API_KEY"]=openai_api_key
    client = OpenAI(
        api_key=openai_api_key
    )

    assistant = client.beta.assistants.create(
        name="Youtube Assistant",
        description="You an eager helpful assistant. You job is to simply experience of Youtube for people."
                    "You can help people in summarizing one or many youtube videos. You can retrieve and summarize"
                    "latest videos from youtube channel from last 3 weeks. You can help people in answering"
                    "questions they might have about a youtube video by getting the transcript of the video and"
                    "using the transcript to answer the question asked. Don't use summary to answer a question, use"
                    "returned transcripts."
                    ,
        model="gpt-4-1106-preview",
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
                  "What is the recent India Maldives row discussed in [url] \n\n" \
                  "Can you shorten this podcast and create clips for me around this topic?\n\n" \
                  "Summarize all the videos released by this channel in past 2 weeks\n\n"
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

            try:
                required_action = run.required_action

                tool_calls = required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for tool_call in tool_calls:
                    tool_call_id = tool_call.id
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name != "process_single_transcript":
                        with st.chat_message("assistant"):
                            with rd.stdout(to=st.empty(), format="markdown"):
                                function_response = asyncio.run(
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
                        function_response = function_to_call(
                            **function_args
                        )

                    tool_outputs.append({
                        "tool_call_id": tool_call_id,
                        "output": function_response
                    })

                    # Append the results of tool call
                    results_append_run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=st.session_state.thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs)

                    results_append_run = check_run_status(results_append_run, st.session_state.thread_id)

                    if function_name == "process_single_transcript":
                        return_assistant_messages(results_append_run, st.session_state.thread_id)
            except Exception as e:
                msg = "Oops! Something is wrong with the request please retry."
                print(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            return_assistant_messages(run, st.session_state.thread_id)

        st.session_state.processing = False

    #####
    #   is the user enters a prompt while another prompt is running then existing execution stops
    #   we create a new thread in which processing happens. This looses the context
    #####
    if prompt and not st.session_state.processing:
        prompt_processing(prompt)
    elif prompt:
        with st.chat_message("assistant"):
            msg = "Execution stopped! Please wait till processing is complete.\n\n"\
                  "Check the 'Running' tab at the top!"
            st.write(msg)

        st.session_state.messages.append({"role": "assistant", "content": msg})

        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

        prompt_processing(prompt)


else:
    st.error("Please enter a valid OpenAI key in the Configuration tab to continue.")