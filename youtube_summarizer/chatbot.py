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

from utils import is_valid_openai_api_key, ui_spacer, process_html_string
from function_definitions import function_definitions

with open("youtube_summarizer/html_code_default_play.html","r") as f:
    html_code_default_play = f.read()

with open("youtube_summarizer/html_code_default_pause.html","r") as f:
    html_code_default_pause = f.read()

import streamlit.components.v1 as components

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
        "create_clips_for_video": create_clips_for_video
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

with st.sidebar:
    st.markdown(f"""
    ## YouTube Buddy
    """)
    st.write("Made by [Akash Kumar](https://www.linkedin.com/in/akashkumar2/).", unsafe_allow_html=True)
    st.markdown('Source code can be found [here](https://github.com/akashe/YoutubeSummarizer/tree/dev).')

st.subheader("YouTube Buddy: Streamline Your YouTube Experience")

ui_spacer(2)

st.markdown(
    """
    👋 Welcome to YouTube Buddy!

🔗 Enter a valid OpenAI access key.

⭐️ Interact with videos using chat.

💡 Summarize videos, generate short clips from them, get latest videos released by your favourite channels.

🎯 Get the gist quickly and start navigating YouTube smarter, not harder!
    """
)
ui_spacer(2)

with st.expander("Configuration"):

    model_name = "gpt-4-1106-preview"
    # TODO: if the person has access to this model

    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

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
                    "questions they might have in a youtube video."
                    "Politely refuse for requests from user that are not possible for you.",
        model="gpt-4-1106-preview",
        tools=function_definitions
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if not message.get("html_code"):
                st.markdown(message["content"])
            else:
                html = message["content"]
                components.html(html, height=800)

    if st.session_state.thread_id is None:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

    if prompt := st.chat_input("Enter prompt"):

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

                with st.chat_message("assistant"):
                    with rd.stdout(to=st.empty(), format="markdown"):
                        function_response = asyncio.run(
                            function_to_call(
                                **function_args
                            )
                        )

                        if function_name == "create_clips_for_video":
                            new_html_code = deepcopy(html_code_default_play).replace('{{VIDEOS_JSON}}', function_response)
                            new_html_code = process_html_string(new_html_code)
                            #TODO: Remove extra spaces that appear in chat
                            components.html(new_html_code, height=800)

                if function_name != "create_clips_for_video":
                    st.session_state.messages.append({"role": "assistant", "content": function_response})
                if function_name == "create_clips_for_video":
                    new_html_code = deepcopy(html_code_default_pause).replace('{{VIDEOS_JSON}}', function_response)
                    new_html_code = process_html_string(new_html_code)
                    st.session_state.messages.append({"role": "assistant", "content": new_html_code, "html_code": True})

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
        else:
            return_assistant_messages(run, st.session_state.thread_id)

        #TODO: handle openai.BadRequestError: Error code: 400 when you add text message when a run is active

else:
    st.error("Please enter a valid OpenAI key in the Configuration tab to continue.")