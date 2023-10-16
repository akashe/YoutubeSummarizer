import pdb

import argparse
from typing import List

from youtube.get_information import YoutubeConnect
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound

from utils import get_adjusted_iso_date_time, get_transcript_from_xml, check_supported_models
from get_chain import get_summary_of_each_video, get_documents, get_summary_with_keywords

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from prompts import get_per_document_with_keyword_prompt_template, \
    get_combine_document_prompt_template, \
    get_per_document_prompt_template, \
    get_combine_document_with_source_prompt_template


#["https://www.youtube.com/@BeerBiceps", "https://www.youtube.com/@hubermanlab","https://www.youtube.com/@MachineLearningStreetTalk"]
#["AGI", "history", "spirituality", "human pyschology", "new developments in science"]
def process_videos(
    youtube_video_links: List[str] = ["https://www.youtube.com/watch?v=MVYrJJNdrEg", "https://www.youtube.com/watch?v=e8qJsk1j2zE"],
    search_terms: List[str] = None,
    get_source: bool = False,
    model_name: str = "gpt-4"
) -> str:

    # TO do check each link for correctness
    video_ids = [video.split("?v=")[1] for video in youtube_video_links]

    logger.info(f"Analyzing a total of {len(video_ids)} videos")
    print("\n")
    print(f"Analyzing a total of {len(video_ids)} videos")
    transcripts = []
    for video_id in video_ids:
        try:
            json_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-GB'])
            #clean_transcript = get_transcript_from_xml(json_transcript)
            transcripts.append((video_id, json_transcript))
        except (TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound):
            logger.info(f'Subtitles unavailable for the video https://www.youtube.com/watch?v={video_id}')
            print("\n")
            print(f'English transcripts unavailable for the video https://www.youtube.com/watch?v={video_id}')

    documents = get_documents(transcripts, model_name)
    try:
        if search_terms:
            per_document_template = get_per_document_with_keyword_prompt_template(model_name)
            combine_document_template = get_combine_document_with_source_prompt_template(model_name) if get_source \
                else get_combine_document_prompt_template(model_name)
            result = get_summary_with_keywords(documents, search_terms, per_document_template, combine_document_template, model_name)
        else:
            per_document_template = get_per_document_prompt_template(model_name)
            result = get_summary_of_each_video(documents, per_document_template, model_name)
        #result = get_chain_for_summary(documents, search_terms)
    except Exception as e:
        print(e)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('youtube_video_links', nargs='+', metavar='y',
                        help='a list youtube channel links that you want to process')
    parser.add_argument('-s', '--search_terms', type=str, nargs='*',
                        help="design the summary around your topics of interest. If not given,"
                             "a general summary will be created.")
    parser.add_argument('--return_sources', action='store_true', default=False,
                        help="To return sources of information in the final summary.")
    parser.add_argument('--model_name', default='gpt-4',
                        help="model to use for generating summaries.")

    args = parser.parse_args()

    assert check_supported_models(args.model_name), "Model not available in config"

    process_videos(args.youtube_channel_links, args.summary_of_n_weeks, args.search_terms, args.return_sources)

