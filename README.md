## YouTube Insight: Streamline Your YouTube Experience

[Click here to try it out!](https://youtubeinsight.streamlit.app/)

In the modern world, we find ourselves overwhelmed with an overabundance of content on a daily basis. Furthermore, staying on top of the latest videos, podcasts, and interviews could be an uphill task, especially in fields such as AI. 

To make matters easier, we introduce to you Youtube Summarizer, a tool designed to provide summarized recaps of your favorite videos and channels. The features of our App include:

1. **Video Summary**: Generate a concise summary of videos.
2. **Keyword Specific Summary**:
   - Enter a specific keyword such as "conflict" to obtain a targeted summary focused on that topic.
   - Ideal when a video covers multiple topics of interest like "war", "AGI", "Psychology"; the app will produce summaries relevant to each topic.
3. **Channel Summary**: If you don't have time to watch every video from your favorite channel, use our tool to generate a summary of all videos released in the past few weeks across multiple channels. A handy way to stay updated with what’s interesting.
4. **Keyword Specific Channel Summary**: Tailor your channel summary based on specific topics of interest. For instance, if you are interested in "market fluctuation", get a precise summary of videos discussing this topic over the past weeks. Save your time and benefit from focused viewing.

### Obtain your OpenAI API key
This application uses OpenAI models. To use the app, you need an API key.
Click [here](https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt) to learn how to create your OpenAI API key.



### Choosing between GPT4 and GPT3.5:

The summarization quality is significantly influenced by the Language Learning Model (LLM) choice:

- GPT3.5 provides you with fast but succinct information. 
- For a more comprehensive and detailed summary, resort to GPT-4. However, consider slower processing due to usage rate limits. 

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

- **process_videos**: Use this entry point to process a single or multiple videos, employing or ignoring keywords.
- **process_channels**: Use this to process single or multiple channels with or without search terms.
- **process_a_playlist**: This entry point processes all the videos in a selected playlist.