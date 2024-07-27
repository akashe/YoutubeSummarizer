## YouTube Buddy: Streamline Your YouTube Experience

[Click here to try it out!](https://youtubebuddy.streamlit.app/)

Get information from youtube videos the way you want it.

1. **Video Summary**: Generate a concise summary of videos.
2. **Keyword Specific Summary**:
   - Enter a specific keyword such as "conflict" to obtain a targeted summary focused on that topic.
   - Ideal when a video covers multiple topics of interest like "war", "AGI", "Psychology"; the app will produce summaries relevant to each topic.
3. **Channel Summary**: If you don't have time to watch every video from your favorite channel, use our tool to generate a summary of all videos released in the past few weeks across multiple channels. A handy way to stay updated with what’s interesting.
4. **Keyword Specific Channel Summary**: Tailor your channel summary based on specific topics of interest. For instance, if you are interested in "market fluctuation", get a precise summary of videos discussing this topic over the past weeks. Save your time and benefit from focused viewing.
5. **Ask question about video**: Want to understand topics mentioned in the video, just ask the bot to answer it for you.
6. **Create clips**: Create smaller video clips from video for general summary of the video or you can ask to create clips for specific topics also.

### Obtain your OpenAI API key
This application uses OpenAI models. To use the app, you need an API key.
Click [here](https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt) to learn how to create your OpenAI API key.

### Steps to Run the App Locally

Running the app locally involves the following steps:

1. [Set up a Google cloud project in Google workspaces.](https://developers.google.com/workspace/guides/get-started)
2. [Turn on Youtube Data API v3 for your Google project.](https://www.youtube.com/watch?v=fN8WwVQTWYk)
3. [Create OAuth configuration for your project.](https://developers.google.com/workspace/guides/configure-oauth-consent)
4. [Generate access credentials for your Google cloud project.](https://developers.google.com/workspace/guides/create-credentials)
5. [Download 'credentials.json' from your project and keep it locally.](https://techiejackieblogs.com/how-to-create-google-mail-api-credentials-json/)
6. Execute 'youtube_summarizer/process_videos.py'.
7. On initial execution, authenticate the project via a Gmail account.
8. This will locally save a 'token.json' file, containing the access token for future access without having to authenticate again.


#### Entry points for code

- **chatbot.py**: Streamlit UI to get information from youtube using a chatbot.
- **process_videos**: Use this entry point to process a single or multiple videos, employing or ignoring keywords.
- **process_channels**: Use this to process single or multiple channels with or without search terms.
- **process_a_playlist**: This entry point processes all the videos in a selected playlist.
- **process_clips**: This entry point creates clips for one or multiple videos.
