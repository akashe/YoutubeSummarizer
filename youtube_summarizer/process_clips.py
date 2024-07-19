from get_chain import acompletion_with_retry
from youtube.get_information import YoutubeConnect
from utils import get_transcripts
from copy import deepcopy

import json
import pdb
import re
from typing import List, Dict

import logging

from utils import chars_processed_dict_for_failed_cases_with_some_processing, \
    chars_processed_dict_for_failed_cases_with_no_processing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

re_pattern = r'\[\s*(\d+(\.\d+)?)\s*,\s*(\d+(\.\d+)?)\s*\]'

improve_performance_system_prompt = "It is a Monday in October, most productive day of the year. Think step by step." \
                                    "I will tip you 200$ for every request you answer right. Take a deep breadth." \
                                    "YOU CAN DO THIS TASK. "

"""
summarize_prompt = {
    "system": "You are a news reporter whose job is to find the key items discussed in a video. "
              "A transcript of a video in a list format. It contains the start time of words expressed and what was  "
              "said in the video. Given the transcript, identify the most important things discussed during that time.",
    "user": "Transcript: {context}"
}

get_timestamp_prompt = {
    "system": "You are a very peculiar mind which likes to solve nitty-gritty things with an eye on detail."
              "Given a timestamped transcript of a youtube video and the most important points discussed in the video,Your job is "
              "to suggest a range of timestamps. You have to refer the most important points of video and identify where they are present in the video."
              "The transcript is a list of dict containing the start time for a phrase or sentence and the phrase or sentence itself."
              "The timestamp ranges would be used to create clips from the video. The output will exclusively comprise lists of timestamp"
              " ranges, with each element of the list presented in the following manner: [start_time, end_time]. This format should be strictly adhered to for consistency and clarity."
              "No explanatory text is needed only the ranges in the above format will suffice."
              " Your selection process will be meticulous to ensure that viewers get the most informative and comprehensive understanding of the video's content from the highlighted segments.",
    "user": "Transcript: {context} \n Most important points: {summary}"
}
"""
summarize_prompt = {
    "system": "You are a news reporter whose job is to find the key items discussed in a video. "
              "A transcript of a video in a list format. It contains the start time of words expressed and what was  "
              "said in the video. Given the transcript, identify the most important things discussed during that time."
              "Just give no more than {bullet_points} the bullet points of most important points and don't generate a summary.",
    "user": "Transcript: {context}"
}

summarize_prompt_specific_terms = {
    "system": "You are a news reporter whose job is to find the key items discussed in a video. "
              "A transcript of a video in a list format. It contains the start time of words expressed and what was  "
              "said in the video. Given the transcript, identify things discussed about the topic(s) {search_terms} during that time."
              "If the topic(s) {search_terms} are not discussed, return '-1'",
    "user": "Transcript: {context}"
}

get_timestamp_prompt = {
    "system": "You are a very peculiar mind which likes to solve nitty-gritty things with an eye on detail."
              "Given a timestamped transcript of a youtube video and the most important points discussed in the video,Your job is "
              "to suggest a range of timestamps. You have to refer the most important points of video and identify where they are present in the video."
              "The transcript is a list of dict containing the start time for a phrase or sentence and the phrase or sentence itself."
              "The timestamp ranges would be used to create clips from the video. The output will exclusively comprise lists of timestamp"
              " ranges, with each element of the list presented in the following manner: [start_time, end_time]. This format should be strictly adhered to for consistency and clarity."
              " No explanatory text is needed only the ranges in the above format will suffice."
              " Your selection process will be meticulous to ensure that viewers get the most informative and comprehensive understanding of the video's content from the highlighted segments."
              " The output list of time stamps should not contain more than {len_range_items} items."
              " DON'T CREATE RANGES WITH LESS THAN 10 SECOND DIFFERENCE BETWEEN start_time and end_time."
              " Do not create ranges for the timestamps that discuss advertisement, endorsements or paid content information",
    "user": "Transcript: {context} \n Most important points: {summary}"
}

get_timestamp_prompt_specific_terms = {
    "system": "You are a very peculiar mind which likes to solve nitty-gritty things with an eye on detail."
              "Given a timestamped transcript of a youtube video and things discussed about the topic(s) {search_terms} in the video,Your job is "
              "to suggest a range of timestamps. You have to refer the things discussed about the topic(s) {search_terms} in the video and identify where they are present in the video."
              "The transcript is a list of dict containing the start time for a phrase or sentence and the phrase or sentence itself."
              "The timestamp ranges would be used to create clips from the video. The output will exclusively comprise lists of timestamp"
              " ranges, with each element of the list presented in the following manner: [start_time, end_time]. This format should be strictly adhered to for consistency and clarity."
              " No explanatory text is needed only the ranges in the above format will suffice."
              " Your selection process will be meticulous to ensure that viewers get the most informative and comprehensive understanding of the topic(s) {search_terms} from the highlighted segments."
              " The output list of time stamps should not contain more than {len_range_items} items."
              " DON'T CREATE RANGES WITH LESS THAN 10 SECOND DIFFERENCE BETWEEN start_time and end_time."
              " Do not create ranges for the timestamps that discuss advertisement, endorsements or paid content information",
    "user": "Transcript: {context} \n things discussed about the topic(s): {summary}"
}

MAX_TIME_PER_LLM_CALL = 60*10*3

def get_waypoints_for_video_len(video_total_len_estimate):

    default_limit_per_llm_call = 60*10
    default_len_range_items = 10
    default_bullet_points = 5

    # Parse no more than 40 minutes of video content to get good performance from LLM
    if video_total_len_estimate > MAX_TIME_PER_LLM_CALL:
        factor = int(video_total_len_estimate/default_limit_per_llm_call)
    else:
        factor = int(video_total_len_estimate/default_limit_per_llm_call) + 1

    return factor*default_bullet_points, factor*default_len_range_items, factor*default_limit_per_llm_call


def get_videos_and_ranges(ranges):

    result = []
    for id_ in ranges:
        for el in ranges[id_]:
            result.append(
                {"id": id_, "start": el[0], "end": el[1]}
            )
    return result


def ranges_in_float_from_llm_response(response:str):

    result = []
    hits = re.findall(re_pattern, response, flags=re.MULTILINE)

    for hit in hits:
        range_element = []
        assert len(hit) == 4
        range_element.append(int(float(hit[0])))
        range_element.append(int(float(hit[2])))
        result.append(range_element)

    return result


def parse_captions(text_captions: List[List[dict]],
                   time_limit_per_llm_call: int = 60 * 10):

    assert len(text_captions) > 0
    alert_message_given = False
    transcripts = []
    total_seconds_of_conversation = 0
    last_start = 0
    for i in text_captions:

        duration = i['duration']
        total_seconds_of_conversation += duration
        # total time should be more than 5 mins and the total time in subtitles also greater than 5 mins
        # total_seconds_of_conversation forces the system to gather text for videos which contains less conversation
        if total_seconds_of_conversation > time_limit_per_llm_call and (i['start']-last_start) > time_limit_per_llm_call:
            if not alert_message_given and time_limit_per_llm_call >= MAX_TIME_PER_LLM_CALL:
                print("\nProcessing a long video, this may take more than a minute...\n")
                alert_message_given = True
            yield transcripts
            transcripts = []
            total_seconds_of_conversation = 0
            last_start = i['start']

        # Try with the duration also
        transcripts.append({"start": i['start'], "text": i['text']})

    if len(transcripts) > 0:
        yield transcripts


async def get_time_stamp_ranges(video_titles: List[str],
                                video_ids: List[str],
                                transcripts: List[List[dict]],
                                search_terms: List[str] = None,
                                model_name: str = "gpt-4o-mini") -> (dict, Dict[str, List[List[float]]]):
    input_char_len_processed = 0
    output_char_len_processed = 0
    ranges = {}
    for video_title, video_id, transcript in zip(video_titles, video_ids, transcripts):

        video_total_len_estimate = transcript[-1]["start"]
        n_bullet_points, n_len_range_items, time_limit_per_llm_call = get_waypoints_for_video_len(
            video_total_len_estimate)

        ranges[video_id] = []
        parsed_transcripts = parse_captions(transcript, time_limit_per_llm_call)
        for fixed_duration_transcript in parsed_transcripts:
            # First we create a summary of the video

            if search_terms:
                terms = " ".join(search_terms)
                prompt_dict = deepcopy(summarize_prompt_specific_terms)
                prompt_dict['system'] = improve_performance_system_prompt + prompt_dict['system'].format(
                    bullet_points=n_bullet_points,
                    search_terms=terms)
            else:
                prompt_dict = deepcopy(summarize_prompt)
                prompt_dict['system'] = improve_performance_system_prompt + prompt_dict['system'].format(
                    bullet_points=n_bullet_points)
            prompt_dict['user'] = prompt_dict['user'].format(context=fixed_duration_transcript)
            max_tokens = 1024

            chars_processed_for_bulled_points, key_points = await acompletion_with_retry(model_name=model_name,
                                                      prompt_dict=prompt_dict,
                                                      max_tokens=max_tokens,
                                                      stream=False)

            if key_points == "-1":
                # topic not discussed in the video
                print(f"\nThe sought topics are not discussed in the video '{video_title}'\n")
                continue

            if search_terms:
                terms = " ".join(search_terms)
                prompt_dict = deepcopy(get_timestamp_prompt_specific_terms)
                prompt_dict['system'] = improve_performance_system_prompt + prompt_dict['system'].format(
                    len_range_items=n_len_range_items,
                    search_terms=terms)
            else:
                prompt_dict = deepcopy(get_timestamp_prompt)
                prompt_dict['system'] = improve_performance_system_prompt + prompt_dict['system'].format(
                    len_range_items=n_len_range_items)
            prompt_dict["user"] = prompt_dict["user"].format(context=fixed_duration_transcript, summary=key_points)
            max_tokens = 1024

            chars_processed_for_ranges, fixed_duration_range_str = await acompletion_with_retry(model_name=model_name,
                                                                    prompt_dict=prompt_dict,
                                                                    max_tokens=max_tokens,
                                                                    stream=False)

            fixed_duration_range = ranges_in_float_from_llm_response(fixed_duration_range_str)
            ranges[video_id].extend(fixed_duration_range)

            input_char_len_processed += (chars_processed_for_bulled_points["input_chars"] +
                                         chars_processed_for_ranges["input_chars"])
            output_char_len_processed += (chars_processed_for_bulled_points["output_chars"] +
                                          chars_processed_for_ranges["output_chars"])

    total_char_len_processed = {
        "input_chars": input_char_len_processed,
        "output_chars": output_char_len_processed
    }
    return total_char_len_processed, ranges


async def create_clips_for_video(youtube_video_links: List[str],
                                 search_terms: List[str] = None,
                                 model_name: str = "gpt-4o-mini"):

    youtube_connect = YoutubeConnect()

    try:
        video_ids = []
        for link in youtube_video_links:
            if "m.youtube" in link:
                video_id = link.split("v=")[-1].split("&")[0]
            elif "youtu.be" in link:
                video_id = link.split("/")[-1].split("?")[0]
            else:
                video_id = link.split("?v=")[1].split('&')[0]
            video_ids.append(video_id)
    except Exception as e:
        msg = "Enter valid urls"
        print(msg)
        logger.error(msg)
        return chars_processed_dict_for_failed_cases_with_no_processing, msg

    logger.info(f"Analyzing {len(video_ids)} videos")
    print("\n")
    if len(video_ids) > 1:
        print(f"Analyzing {len(video_ids)} videos")
    else:
        print(f"Analyzing {len(video_ids)} video")

    print("\n")
    video_titles = []
    for id_, video_id in enumerate(video_ids):
        video_title = youtube_connect.get_video_title(video_id)
        video_titles.append(video_title)
        print(f"{id_ + 1}. [{video_title}](https://www.youtube.com/watch?v={video_id})")

    transcripts = get_transcripts(video_ids, video_titles)
    if len(transcripts) == 0:
        logger.error("Transcripts not available")
        return chars_processed_dict_for_failed_cases_with_no_processing, "Transcripts not available"

    # get time stamp ranges
    chars_processed, ranges = await get_time_stamp_ranges(video_titles, video_ids, transcripts, search_terms, model_name)

    if len(ranges) == 0:
        logger.error("The sought topics are not discussed in the video")
        return chars_processed, "The sought topics are not discussed in the video"

    # embed html code with those ranges
    ranges_with_video_ids = get_videos_and_ranges(ranges)

    # return the video ranges
    videos_json = json.dumps(ranges_with_video_ids)

    return chars_processed, videos_json

