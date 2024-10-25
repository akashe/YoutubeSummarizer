from datetime import datetime, timedelta
import json
from typing import List
import openai
import streamlit as st
import random


from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


def get_adjusted_iso_date_time(summary_of_n_weeks: int) -> str:
    current_date = datetime.now()
    adjusted_date = current_date - timedelta(weeks=summary_of_n_weeks)

    return adjusted_date.strftime('%Y-%m-%dT%H:%M:%SZ')


def get_transcript_from_xml(transcript_in_xml: dict) -> str:
    text_transcript = "\n".join([i["text"] for i in transcript_in_xml])

    return text_transcript


def check_supported_models(model_name: str) -> bool:
    with open("model_config.json","r") as f:
        supported_models = json.load(f)

    return model_name.lower() in supported_models


def get_transcripts(video_ids: List[str], video_titles: List[str]) -> List[List[dict]]:

    username = st.secrets["proxy_username"]
    password = st.secrets["proxy_password"]

    ports = ["10001", "10002", "10003", "10004", "10005", "10006", "10007", "10008", "10009", "10010"]
    port = random.choice(ports)

    proxy = f"http://{username}:{password}@gate.smartproxy.com:{port}"
    logger.info(f'proxy: {proxy}')

    proxies = {
        'http': proxy,
        'https': proxy
    }
    
    transcripts = []
    for video_id, video_title in zip(video_ids, video_titles):
        try:
            json_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-GB'], proxies=proxies)
            transcripts.append(json_transcript)
        except (TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound) as e:
            logger.info(f'Subtitle error {e}')
            logger.info(f'Subtitles unavailable for the video "{video_title}"')
            print("\n")
            print(f'English transcripts unavailable for the video "{video_title}"')

    return transcripts


def http_connection_decorator(func):
    # Legacy code to maintain same HTTP clients for connections.
    # Not needed for new version of openai python lib
    async def inner(*args, **kwargs):
        try:
            from aiohttp import ClientSession
            openai.aiosession.set(ClientSession())

            return await func(*args, **kwargs)

        except Exception as e:
            print("Something bad happened with the request. Please retry :)")
        finally:
            await openai.aiosession.get().close()

    return inner


def is_valid_openai_api_key(api_key: str) -> bool:
    openai.api_key = api_key
    try:
        openai.models.list()
    except openai.AuthenticationError as e:
        return False
    else:
        return True


def ui_spacer(n=2, line=False, next_n=0):
    for _ in range(n):
        st.write('')
    if line:
        st.tabs([' '])
    for _ in range(next_n):
        st.write('')


def process_html_string(html_string):
    # Replace multiple newlines with a single one, or remove them entirely
    processed_string = ''.join(html_string.splitlines())
    processed_string = processed_string.replace("\t", "").replace("    ", "")
    return processed_string


# HTML and JavaScript with dynamic sizing and f-string for videos
html_code_default_play = """
<div id="videoContainer" style="width: 100%;"></div>
<script src="https://www.youtube.com/iframe_api"></script>
<script>
var player;
var currentVideoIndex = 0;
var videos = {videos_json};

function onYouTubeIframeAPIReady() {{
    loadVideo(currentVideoIndex);
}}

function loadVideo(index) {{
    if (index < videos.length) {{
        var video = videos[index];
        var container = document.getElementById('videoContainer');
        var width = container.offsetWidth;
        var height = width * (9/16); // Maintain a 16:9 aspect ratio
        player = new YT.Player('videoContainer', {{
            height: height,
            width: width,
            videoId: video.id,
            playerVars: {{
                'autoplay': 1,
                'start': video.start,
                'end': video.end,
                'controls': 1
            }},
            events: {{
                'onReady': onPlayerReady,
                'onStateChange': onPlayerStateChange
            }}
        }});
    }}
}}

function onPlayerReady(event) {{
    event.target.playVideo();
}}

function onPlayerStateChange(event) {{
    if (event.data == YT.PlayerState.ENDED) {{
        currentVideoIndex++;
        if (currentVideoIndex < videos.length) {{
            player.destroy(); // Destroy the current player
            loadVideo(currentVideoIndex); // Load the next video
        }}
    }}
}}

window.onresize = function() {{
    if (player) {{
        var container = document.getElementById('videoContainer');
        var width = container.offsetWidth;
        var height = width * (9/16); // Maintain a 16:9 aspect ratio
        player.setSize(width, height);
    }}
}};
</script>
"""

html_code_default_pause = """
<div id="videoContainer" style="width: 100%; cursor: pointer;"></div>
<script src="https://www.youtube.com/iframe_api"></script>
<script>
var player;
var currentVideoIndex = 0;
var videos = {videos_json};

function onYouTubeIframeAPIReady() {{
    loadVideo(currentVideoIndex);
}}

function loadVideo(index) {{
    if (index < videos.length) {{
        var video = videos[index];
        var container = document.getElementById('videoContainer');
        var width = container.offsetWidth;
        var height = width * (9/16); // Maintain a 16:9 aspect ratio
        player = new YT.Player('videoContainer', {{
            height: height,
            width: width,
            videoId: video.id,
            playerVars: {{
                'start': video.start,
                'end': video.end,
                'controls': 1
            }},
            events: {{
                'onStateChange': onPlayerStateChange
            }}
        }});
    }}
}}

function onPlayerStateChange(event) {{
    if (event.data == YT.PlayerState.ENDED) {{
        currentVideoIndex++;
        if (currentVideoIndex < videos.length) {{
            player.destroy(); // Destroy the current player
            loadVideo(currentVideoIndex); // Load the next video
        }}
    }}
}}

document.getElementById('videoContainer').addEventListener('click', function() {{
    if (!player.getPlayerState() || player.getPlayerState() == YT.PlayerState.CUED) {{
        player.playVideo();
    }}
}});

window.onresize = function() {{
    if (player) {{
        var container = document.getElementById('videoContainer');
        var width = container.offsetWidth;
        var height = width * (9/16); // Maintain a 16:9 aspect ratio
        player.setSize(width, height);
    }}
}};
</script>
"""


def get_usage_in_dollars(input_tokens, output_tokens):
    input_tokens_cost = input_tokens*0.15/1000000
    output_tokens_cost = output_tokens*0.60/1000000

    return input_tokens_cost + output_tokens_cost


def get_refresh_time():
    now = datetime.now()

    start_of_next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    time_to_wait = start_of_next_day - now

    hours, remainder = divmod(time_to_wait.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours} hours, {minutes} minutes"


chars_processed_dict_for_failed_cases_with_some_processing = {
    "input_chars": 1024,
    "output_chars": 0
}

chars_processed_dict_for_failed_cases_with_no_processing = {
    "input_chars": 0,
    "output_chars": 0
}
