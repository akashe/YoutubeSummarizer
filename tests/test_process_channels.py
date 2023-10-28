import pytest
import os

import sys
sys.path.append(os.path.join(os.getcwd(), "youtube_summarizer"))

pytest_plugins = ('pytest_asyncio',)

from youtube_summarizer.process_channels import process_channels

youtube_channel_links = ["https://www.youtube.com/@BeerBiceps", "https://www.youtube.com/@hubermanlab","https://www.youtube.com/@MachineLearningStreetTalk"]
search_terms = ["AGI", "history", "spirituality", "human pyschology", "new developments in science"]
summary_of_n_weeks = 1
get_source = True
model_name = "gpt-3.5-turbo-16k"


@pytest.mark.asyncio
async def test_process_channels_with_youtube_video_link():
    assert "OPENAI_API_KEY" in os.environ

    result = await process_channels(["https://www.youtube.com/watch?v=MVYrJJNdrEg"],
                                    summary_of_n_weeks,
                                    search_terms,
                                    get_source,
                                    model_name)

    assert result is None


@pytest.mark.asyncio
async def test_process_channels_with_faulty_weeksk():
    assert "OPENAI_API_KEY" in os.environ

    result = await process_channels(youtube_channel_links,
                                    "abc",
                                    search_terms,
                                    get_source,
                                    model_name)

    assert result is None


@pytest.mark.asyncio
async def test_process_channels_with_faulty_model_name():
    assert "OPENAI_API_KEY" in os.environ

    result = await process_channels(["https://www.youtube.com/@BeerBiceps"],
                                    summary_of_n_weeks,
                                    search_terms,
                                    get_source,
                                    "llama_2")

    assert result is None


@pytest.mark.asyncio
async def test_process_channels_without_search_term():
    assert "OPENAI_API_KEY" in os.environ

    result = await process_channels(["https://www.youtube.com/@BeerBiceps"],
                                    summary_of_n_weeks,
                                    None,
                                    get_source,
                                    model_name)

    assert result is not None
    assert result != "-1"
    assert len(result) > 10


@pytest.mark.asyncio
async def test_process_channels_with_search_term():
    assert "OPENAI_API_KEY" in os.environ

    result = await process_channels(["https://www.youtube.com/@BeerBiceps"],
                                    summary_of_n_weeks,
                                    ["India"],
                                    get_source,
                                    model_name)

    assert result is not None
    assert result != "-1"
    assert len(result) > 10
    assert "India" in result


@pytest.mark.asyncio
async def test_process_channels_with_combined_summary_search_term():
    assert "OPENAI_API_KEY" in os.environ

    result = await process_channels(["https://www.youtube.com/@BeerBiceps"],
                                    5,
                                    ["India"],
                                    get_source,
                                    model_name)

    assert result is not None
    assert result != "-1"
    assert len(result) > 10
    assert "India" in result