import pdb

import argparse
from typing import List

from youtube.get_information import YoutubeConnect
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound

from utils import get_adjusted_iso_date_time, get_transcript_from_xml, check_supported_models, get_transcripts
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
    youtube_video_links: List[str] = ["https://www.youtube.com/watch?v=MVYrJJNdrEg", "https://www.youtube.com/watch?v=e8qJsk1j2zE", "https://youtu.be/m8LnEp-4f2Y?si=ZgwnRQGp_DeztHdC", "https://m.youtube.com/watch?si=ZgwnRQGp_DeztHdC&v=m8LnEp-4f2Y&feature=youtu.be"],
    search_terms: List[str] = None,
    get_source: bool = False,
    model_name: str = "gpt-4"
) -> str:

    youtube_connect = YoutubeConnect()

    # TO do check each link for correctness
    try:
        video_ids = []
        for link in youtube_video_links:
            if "youtu.be" in link:
                video_id = link.split("/")[-1].split("?")[0]
            elif "m.youtube" in link:
                video_id = link.split("&v=")[-1].split("&")[0]
            else:
                video_id = link.split("?v=")[1]
            video_ids.append(video_id)
    except Exception as e:
        print("Enter valid urls")
        return "-1"

    video_titles = []
    for video_id in video_ids:
        video_titles.append(youtube_connect.get_video_title(video_id))

    logger.info(f"Analyzing {len(video_ids)} videos")
    print("\n")
    if len(video_ids) > 1:
        print(f"Analyzing {len(video_ids)} videos")
    else:
        print(f"Analyzing {len(video_ids)} video")

    transcripts = get_transcripts(video_ids)

    documents = get_documents(video_ids, video_titles, transcripts, model_name)
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

