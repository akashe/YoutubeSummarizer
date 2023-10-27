import pdb
import time
import sys
from copy import deepcopy

import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import create_base_retry_decorator
from langchain.schema import Document
from typing import List, Tuple, Any, Callable



from utils import get_model_max_len, get_model_max_tokens

import tiktoken
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_time_encoded_transcripts(transcript: List[dict],
                                 model_name: str) -> Tuple[bool, float, float, str]:
    enc = tiktoken.encoding_for_model(model_name)
    model_max_token_len = get_model_max_len(model_name)

    sentences = []
    length_till_now = 0
    all_start = 0.0
    was_transcript_splitted = False

    for dialogue in transcript:
        text = dialogue["text"]
        start = dialogue["start"]
        duration = dialogue["duration"]

        # To include for tokens added by joining all sentences.
        enc_text = enc.encode(text + "\n")

        if len(enc_text) + length_till_now > model_max_token_len:
            was_transcript_splitted = True
            yield was_transcript_splitted, all_start, start + duration, "\n".join(sentences)
            sentences = [text]
            all_start = start + duration
            length_till_now = len(enc_text)

        else:
            sentences.append(text)
            length_till_now += len(enc_text)

    yield was_transcript_splitted, all_start, start + duration, "\n".join(sentences)


def get_documents(video_ids: List[str],
                  video_titles: List[str],
                  transcripts: List[List[dict]],
                  model_name: str) -> List[Document]:
    documents = []

    for video_id, video_title, transcript in zip(video_ids, video_titles, transcripts):
        for did_split_happen, start, end, text in get_time_encoded_transcripts(transcript, model_name):
            start_min, start_sec = int(start / 60), int(start % 60)
            end_min, end_sec = int(end / 60), int(end % 60)
            video_start = abs(int(start))

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "https://www.youtube.com/watch?v=" + video_id,
                        "video_start": video_start,
                        "start_min": start_min,
                        "start_sec": start_sec,
                        "end_min": end_min,
                        "end_sec": end_sec,
                        "did_split_happen": did_split_happen,
                        "title": video_title
                    }
                )
            )

    return documents


def divide_big_summary_into_parts(summary: str, model_name: str) -> List[str]:
    enc = tiktoken.encoding_for_model(model_name)
    model_max_token_len = get_model_max_len(model_name)

    encoded_summary = enc.encode(summary)

    smaller_summaries = []
    if len(encoded_summary) > model_max_token_len:
        for i in range(0, len(encoded_summary), model_max_token_len):
            smaller_summaries.append(
                enc.decode(encoded_summary[i:i + model_max_token_len])
            )
    else:
        smaller_summaries = [summary]

    return smaller_summaries


def get_updated_prompts(original_prompt_dict: dict,
                        context: str,
                        summary_keywords: Any = None) -> dict:

    prompt_dict = deepcopy(original_prompt_dict)

    if prompt_dict["summary_keywords"]:
        assert summary_keywords is not None
        prompt_dict["system"] = prompt_dict["system"].format(summary_keywords=summary_keywords)

    prompt_dict["user"] = prompt_dict["user"].format(context=context)

    return prompt_dict


def get_max_tokens(text: str, model_name:str) -> int:
    # Return exact number of maximum tokens that the model can use for generation
    # total size of input prompt and the size of transcripts can vary

    model_max_tokens = get_model_max_tokens(model_name)

    enc = tiktoken.encoding_for_model(model_name)
    enc_text = enc.encode(text)

    return model_max_tokens - len(enc_text) - 20


def _create_retry_decorator(
    max_tries: int,
    run_manager: Any = None,
) -> Callable[[Any], Any]:
    import openai

    errors = [
        openai.error.Timeout,
        openai.error.APIError,
        openai.error.APIConnectionError,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=max_tries, run_manager=run_manager
    )


async def acompletion_with_retry(max_tries=6,
                                 run_manager=None,
                                 **kwargs: Any) -> str:

    retry_decorator = _create_retry_decorator(max_tries, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await aget_response_from_llm(**kwargs)

    return await _completion_with_retry(**kwargs)


async def aget_response_from_llm(model_name: str,
                                 prompt_dict: dict,
                                 context: str,
                                 stream: bool = True,
                                 summary_keywords: Any = None) -> str:

    prompt_dict = get_updated_prompts(prompt_dict, context, summary_keywords)

    max_tokens = get_max_tokens(prompt_dict["system"]+prompt_dict["user"], model_name)

    response = await openai.ChatCompletion.acreate(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt_dict["system"]},
            {'role': 'user', 'content': prompt_dict["user"]}
        ],
        temperature=0,
        max_tokens=max_tokens,
        stream=stream
    )

    if not stream:
        output = response['choices'][0]["message"]["content"]
    else:
        collected_response = []
        async for chunk in response:
            try:
                finish = chunk['choices'][0]["finish_reason"]
                if finish == "stop" or finish == "length":
                    break
                chunk_message = chunk['choices'][0]['delta']["content"]  # extract the message
                sys.stdout.write(chunk_message)
                sys.stdout.flush()
                collected_response.append(chunk_message)
            except:
                pdb.set_trace()
                continue

        output = "".join(collected_response)

    return output


def get_summary_with_keywords(documents: List[Document],
                              keywords: List[str],
                              per_document_template: PromptTemplate,
                              combine_document_template: PromptTemplate,
                              open_ai_model: str) -> str:
    llm = ChatOpenAI(model_name=open_ai_model,
                     temperature=0.0,
                     streaming=True,
                     callbacks=[StreamingStdOutCallbackHandler()]
                     )

    summary_keywords = ", ".join(keywords)

    per_document_llm_chain = LLMChain(llm=llm, prompt=per_document_template)

    smaller_summaries = []
    for i, d in enumerate(documents):
        logger.info(f'Summary {i}:\n')
        print('\n')

        if d.metadata["did_split_happen"]:
            print(f'Summary of video "{d.metadata["title"]}"'
                  f' from {d.metadata["start_min"]}:{d.metadata["start_sec"]} to '
                  f'{d.metadata["end_min"]}:{d.metadata["end_sec"]} \n')
            source_doc = d.metadata["source"] + f"&t={d.metadata['video_start']}s"
        else:
            print(f'Summary of video "{d.metadata["title"]}"\n')
            source_doc = d.metadata["source"]

        d_summary = per_document_llm_chain.run(context=d.page_content, summary_keywords=summary_keywords)
        smaller_summaries.append((source_doc, d_summary))

        if 'gpt-4' in open_ai_model:
            logger.info(f'\nWaiting\n')
            print('\n')
            print(f'Waiting to avoid token rate limits associated with GPT-4')
            time.sleep(47)

    logger.info("Creating combined summary")
    print('\n')
    print("Creating combined summary")
    print('\n')

    big_summary = ""
    for source, summary in smaller_summaries:
        big_summary += f"Source: {source}\n Summary: {summary}\n\n"

    llm_2 = ChatOpenAI(model_name=open_ai_model,
                       temperature=0.0,
                       streaming=True,
                       )

    combined_document_chain = LLMChain(llm=llm_2, prompt=combine_document_template)

    # return combined_document_chain.run(doc_summaries=big_summary, summary_keywords=summary_keywords)

    summaries = divide_big_summary_into_parts(big_summary, open_ai_model)

    while len(summaries) > 1:
        smaller_chunks = ""
        for smaller_summary in summaries:
            smaller_chunks += combined_document_chain.run(doc_summaries=smaller_summary,
                                                          summary_keywords=summary_keywords)

        summaries = divide_big_summary_into_parts(smaller_chunks, open_ai_model)

    return combined_document_chain.run(doc_summaries=summaries[0],
                                       summary_keywords=summary_keywords,
                                       callbacks=[StreamingStdOutCallbackHandler()])


async def aget_summary_with_keywords(documents: List[Document],
                                     keywords: List[str],
                                     per_document_template: dict,
                                     combine_document_template: dict,
                                     open_ai_model: str) -> str:
    summary_keywords = ", ".join(keywords)

    smaller_summaries = []
    for i, d in enumerate(documents):
        logger.info(f'Summary {i}:\n')
        print('\n')

        if d.metadata["did_split_happen"]:
            print(f'Summary of video "{d.metadata["title"]}"'
                  f' from {d.metadata["start_min"]}:{d.metadata["start_sec"]} to '
                  f'{d.metadata["end_min"]}:{d.metadata["end_sec"]} \n')
            source_doc = d.metadata["source"] + f"&t={d.metadata['video_start']}s"
        else:
            print(f'Summary of video "{d.metadata["title"]}"\n')
            source_doc = d.metadata["source"]

        d_summary = await acompletion_with_retry(model_name=open_ai_model,
                                                 prompt_dict=per_document_template,
                                                 context=d.page_content,
                                                 summary_keywords=summary_keywords)

        smaller_summaries.append((source_doc, d_summary))

        if 'gpt-4' in open_ai_model:
            logger.info(f'\nWaiting\n')
            print('\n')
            print(f'Waiting to avoid token rate limits associated with GPT-4')
            time.sleep(47)

    logger.info("Creating combined summary")
    print('\n')
    print("Creating combined summary")
    print('\n')

    big_summary = ""
    for source, summary in smaller_summaries:
        big_summary += f"Source: {source}\n Summary: {summary}\n\n"

    summaries = divide_big_summary_into_parts(big_summary, open_ai_model)

    while len(summaries) > 1:
        smaller_chunks = ""
        for smaller_summary in summaries:
            smaller_chunks += await acompletion_with_retry(model_name=open_ai_model,
                                                           prompt_dict=combine_document_template,
                                                           context=smaller_summary,
                                                           stream=False,
                                                           summary_keywords=summary_keywords)

        summaries = divide_big_summary_into_parts(smaller_chunks, open_ai_model)

    final_summary = await acompletion_with_retry(model_name=open_ai_model,
                                                 prompt_dict=combine_document_template,
                                                 context=summaries[0],
                                                 summary_keywords=summary_keywords)
    return final_summary


def get_summary_of_each_video(documents: List[Document],
                              per_document_template: PromptTemplate,
                              open_ai_model: str) -> str:
    llm = ChatOpenAI(model_name=open_ai_model,
                     temperature=0.0,
                     streaming=True,
                     callbacks=[StreamingStdOutCallbackHandler()]
                     )

    per_document_llm_chain = LLMChain(llm=llm, prompt=per_document_template)

    summary = ""
    for i, d in enumerate(documents):
        logger.info(f'Summary {i}:\n')
        print('\n')
        if d.metadata["did_split_happen"]:
            print(f'Summary of video "{d.metadata["title"]}" from '
                  f'{d.metadata["start_min"]}:{d.metadata["start_sec"]} to '
                  f'{d.metadata["end_min"]}:{d.metadata["end_sec"]} \n')
        else:
            print(f'Summary of video "{d.metadata["title"]}"\n')

        d_summary = per_document_llm_chain.run(context=d.page_content)
        if 'gpt-4' in open_ai_model:
            logger.info(f'\nWaiting\n')
            print('\n')
            print(f'Waiting to avoid token rate limits associated with GPT-4')
            time.sleep(47)
        summary += d_summary
        summary += f"\n\nSource: https://www.youtube.com/watch?v={d.metadata['source']}"
        if d.metadata["did_split_happen"]:
            summary += f"&t={d.metadata['video_start']}s"
        summary += "\n"

    return summary


async def aget_summary_of_each_video(documents: List[Document],
                                     per_document_template: dict,
                                     open_ai_model: str) -> str:

    summary = ""
    for i, d in enumerate(documents):
        logger.info(f'Summary {i}:\n')
        print('\n')
        if d.metadata["did_split_happen"]:
            print(f'Summary of video "{d.metadata["title"]}" from '
                  f'{d.metadata["start_min"]}:{d.metadata["start_sec"]} to '
                  f'{d.metadata["end_min"]}:{d.metadata["end_sec"]} \n')
        else:
            print(f'Summary of video "{d.metadata["title"]}"\n')

        d_summary = await acompletion_with_retry(model_name=open_ai_model,
                                                 prompt_dict=per_document_template,
                                                 context=d.page_content)

        if 'gpt-4' in open_ai_model:
            logger.info(f'\nWaiting\n')
            print('\n')
            print(f'Waiting to avoid token rate limits associated with GPT-4')
            time.sleep(47)
        summary += d_summary
        summary += f"\n\nSource: https://www.youtube.com/watch?v={d.metadata['source']}"
        if d.metadata["did_split_happen"]:
            summary += f"&t={d.metadata['video_start']}s"
        summary += "\n"

    return summary
