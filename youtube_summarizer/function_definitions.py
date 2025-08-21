function_definitions = [
    {
        "type": "function",
        "function": {
            "name": "process_videos",
            "description": "**PRIMARY FUNCTION FOR VIDEO SUMMARIES** - Use this function to create summaries of one or multiple YouTube videos. This is the ONLY function that should be used when users ask to 'summarize a video' or 'give me a summary'. You get a general summary if no keywords are given. If keywords are given then a summary around those keywords is returned. DO NOT use process_single_transcript for summaries.",
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
            "description": "Get the summary of one or multiple youtube channels for recent uploads. The user selects the number of weeks they want summary for (1-3 weeks). The function retrieves all videos released on the channel in that time frame. You get a general summary if no keywords are given. If keywords are given then a summary around those keywords is returned.",
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
                        "description": "Number of total last weeks for which the youtube videos will be retrieved from the channel. Minimum value is 1 week and maximum is 3 weeks",
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
            "description": "Generate shorter clips from a YouTube video for users who want to watch only the most important segments. The output is a JSON object containing a list of clips, each identified by video ID and start/stop timestamps.",
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
            "description": "**FOR SPECIFIC QUESTIONS ONLY** - Use this function ONLY when users ask specific questions about a video's content (e.g., 'What did they say about X?', 'At what time do they mention Y?'). This returns the full English transcript which you can use to answer detailed questions. DO NOT use this function for general summaries - use process_videos instead. If the function returns 'Sorry! English transcripts unavailable for the video' then an answer cannot be generated.",
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