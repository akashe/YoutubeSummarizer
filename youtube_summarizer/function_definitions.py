function_definitions = [
    {
        "type": "function",
        "function": {
            "name": "process_videos",
            "description": "Get the summary of one or multiple videos. You get a general summary if no keywords"
                           "are given. If keywords are given then a summary around those keywords is returned."
                           "This function only returns a summary and cant be used to answer questions about a video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "youtube_video_links": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "List of youtube video url in string format to process by the function.",
                    },
                    "search_terms": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "List of search terms in string format to create summary around a specific terms.",
                    },
                    "get_source": {
                        "type": "boolean",
                        "description": "Boolean value to return the source video ids in the final summaries",
                    },
                },
                "required": ["youtube_video_links"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_channels",
            "description": "Get the summary of one or multiple youtube channels. The uesr selects the number of weeks"
                           "they want summary for. The function retrieves all the video released on the channel in that"
                           "time frame. The function also takes input for keywords. You get a general summary if no keywords"
                           "are given. If keywords are given then a summary around those keywords is returned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "youtube_channel_links": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "List of youtube channel urls in string format to process by the function.",
                    },
                    "summary_of_n_weeks": {
                        "type": "number",
                        "description": "Number of total last weeks for which the youtube videos will be retrieved"
                                       "from the channel. Minimum value is 1 week and maximum is 3 weeks",
                        "default": 1
                    },
                    "search_terms": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "List of search terms in string format to create summary around a specific terms.",
                    },
                    "get_source": {
                        "type": "boolean",
                        "description": "Boolean value to return the source video ids in the final summaries",
                    },
                },
                "required": ["youtube_channel_links"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_clips_for_video",
            "description": "Generate smaller clips or videos from a youtube video. Made for people who don't have time "
                           "to watch videos and want to watch the most important points discussed in the video. The out"
                           "put of the function is json object containing a list of clips. Each clip is identified by"
                           "the video id and start and stop second for that video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "youtube_video_links": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "List of youtube video url in string format to process by the function.",
                    },
                    "search_terms": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "List of search terms in string format to create clips around a specific terms.",
                    },
                },
                "required": ["youtube_video_links"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_single_transcript",
            "description": "Given a youtube video url this function returns the english transcript of the video which"
                           "can be used to answer any question about a video. If the model returns that "
                           "'Sorry! English transcripts unavailable for the video' then an answer can't be generated ",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_url": {
                        "type": "string",
                        "description": "The youtube video link for the video whose transcripts are needed."
                    },
                },
                "required": ["video_url"],
            },
        },
    }
]
