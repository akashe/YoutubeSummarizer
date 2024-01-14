import os
import streamlit as st
import validators
import redirect as rd
import asyncio
import openai
from copy import deepcopy

from process_clips import create_clips_for_video

import streamlit.components.v1 as components

from utils import is_valid_openai_api_key, ui_spacer, process_html_string

with open("youtube_summarizer/html_code_default_play.html","r") as f:
    html_code = f.read()

st.session_state.show_player = False

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
    👋 Welcome to YouTube Insight!

🔗 Paste the URL of a video.

⭐️ Create clips from a youtube video.

💡 No need to watch the whole video. Just see the most important parts of it.

🎯 Get the gist quickly and start navigating YouTube smarter, not harder!
    """
)
ui_spacer(2)

'''
with st.expander("Configuration"):
    model_name = st.selectbox(
        'Which LLM you prefer to use?',
        ('GPT-3.5-turbo-16k: Cost effective', 'GPT-4-1106-Preview: Precise but costly'))

    model_name = model_name.split(":")[0].lower()

    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
'''
model_name = 'GPT-4-1106-Preview: Precise but costly'.split(":")[0].lower()
openai_api_key = st.secrets["openai_api_key"]

with st.form("Analysis"):
    try:
        video_links = st.text_input("Enter Comma-Separated YouTube video urls",
                                    placeholder="https://www.youtube.com/watch?v=dBH3nMNKtaQ, "
                                                "https://youtu.be/pGsbEd6w7PI?si=QyK7UMybPx1_Was2")

        video_links = video_links.strip().split(",")
        video_links = [video.strip() for video in video_links if video != ""]
        video_links = [video.split("&")[0] for video in video_links]

        print(video_links)

        url_check = [validators.url(video) for video in video_links]
        video_check = ["watch" in video or "youtu.be" in video for video in video_links]

        if not all(url_check) or not all(video_check):
            raise Exception

    except Exception as e:
        st.error('Please enter valid youtube video urls separated by comma')

    search_terms = st.text_input("Enter Topic(s) For Custom Summary (leave blank for general summary)",
                                 placeholder="nutrition, OpenAI, Israel",
                                 help="Input topics, separated by commas, this will gather all related mentions from the "
                                      "videos for a focused summary.\n Try using GPT-4 for more than 1 topic.")

    submitted = st.form_submit_button("Submit")

    to_out = st.empty()

    if submitted and not openai_api_key:
        st.error("Please add your OpenAI key in the Configuration tab to continue.")

    if submitted and openai_api_key and not is_valid_openai_api_key(openai_api_key):
        st.error("Please enter a valid OpenAI key in the Configuration tab to continue.")

    if submitted and openai_api_key and is_valid_openai_api_key(openai_api_key):

        st.session_state.show_player = True
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        if not search_terms == "":
            search_terms = search_terms.split(",")
            search_terms = [term.strip() for term in search_terms]
        else:
            search_terms = None

        with rd.stdout(to=to_out, format="markdown"):

            if len(video_links) == 0:
                print("Generating summaries using default options")
                video_links = ["https://www.youtube.com/watch?v=dBH3nMNKtaQ",
                               "https://youtu.be/pGsbEd6w7PI?si=QyK7UMybPx1_Was2"]

            if st.session_state.get('show_player', True):
                try:
                    videos_json = asyncio.run(
                        create_clips_for_video(youtube_video_links=video_links,
                                               model_name=model_name)
                    )

                    new_html_code = deepcopy(html_code).replace('{{VIDEOS_JSON}}', videos_json)

                    components.html(new_html_code, height=800)
                except Exception as e:
                    msg = "Oops! Something is wrong with the request please retry."
                    print(msg)
