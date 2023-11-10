import os
import streamlit as st
from process_channels import process_channels
import validators
import redirect as rd
import asyncio
import openai

from utils import is_valid_openai_api_key, ui_spacer

with st.sidebar:
    st.markdown(f"""
    ## YouTube Insight
    """)
    st.write("Made by [Akash Kumar](https://www.linkedin.com/in/akashkumar2/).", unsafe_allow_html=True)
    st.markdown('Source code can be found [here](https://github.com/akashe/YoutubeSummarizer/tree/dev).')

st.subheader("YouTube Insight: Streamline Your YouTube Experience")

ui_spacer(2)

st.markdown(
    """
    👋 Welcome to YouTube Insight!

🔗 Paste the URL of a channel.

📆 Select a timeframe for channels. Options range from 1 to 3 weeks.
  
⭐️ Expect a general summary by default, outlining content from videos released in that time.

💡 Enter search terms to shift from general to specific, topic-focused summaries.

🔥 Process multiple channels at once for insightful content overviews.

🎯 Get the gist quickly and start navigating YouTube smarter, not harder!
    """
)
ui_spacer(2)

with st.expander("Configuration"):
    model_name = st.selectbox(
        'Which LLM you prefer to use?',
        ('GPT-3.5-turbo-16k: Cost effective', 'GPT-4-1106-Preview: Precise but costly'))

    model_name = model_name.split(":")[0].lower()

    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

with st.form("YoutubeSummary"):
    try:
        youtube_channels = st.text_input("Enter Comma-Separated YouTube channel urls",
                                         placeholder="https://www.youtube.com/c/lexfridman, https://www.youtube.com/@hubermanlab")

        youtube_channels = youtube_channels.strip().split(",")
        youtube_channels = [channel.strip() for channel in youtube_channels if channel != ""]

        url_check = [validators.url(channel) for channel in youtube_channels]
        youtube_urls = ["youtube" in url for url in youtube_channels]

        if not all(url_check) or not all(youtube_urls):
            raise Exception
    except Exception as e:
        st.error('Please enter valid urls separated by comma')

    last_n_weeks = st.selectbox("Gather videos from the channels for last how many weeks?",
                                (1, 2, 3))

    search_terms = st.text_input("Enter Topic(s) For Custom Summary (leave blank for general summary)",
                                 placeholder="nutrition, OpenAI, Israel",
                                 help="Input topics, separated by commas, this will gather all related mentions from the "
                                      "videos for a focused summary.\n Try using GPT-4 for more than 1 topic.")

    return_sources = st.toggle("Return sources",
                               help="Get source urls in the combined summary")

    submitted = st.form_submit_button("Submit")

    to_out = st.empty()

    if submitted and not openai_api_key:
        st.error("Please add your OpenAI key in the Configuration tab to continue.")

    if submitted and openai_api_key and not is_valid_openai_api_key(openai_api_key):
        st.error("Please enter a valid OpenAI key in the Configuration tab to continue.")

    if submitted and openai_api_key and is_valid_openai_api_key(openai_api_key):

        openai.api_key = openai_api_key

        if not search_terms == "":
            search_terms = search_terms.split(",")
            search_terms = [term.strip() for term in search_terms]
        else:
            search_terms = None

        with rd.stdout(to=to_out, format="markdown"):

            to_out.empty()

            if len(youtube_channels) == 0:
                print("Generating summaries using default options")
                youtube_channels = ["https://www.youtube.com/c/lexfridman",
                                    "https://www.youtube.com/@hubermanlab"]
                last_n_weeks = 3

            _ = asyncio.run(
                process_channels(youtube_channels,
                                 last_n_weeks,
                                 search_terms,
                                 return_sources,
                                 model_name)
            )
