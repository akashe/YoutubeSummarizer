import pdb
import time
import sys

from langchain.chains import (
    StuffDocumentsChain,
    LLMChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain,
)
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from typing import List, Tuple

from utils import get_model_max_len

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

'''
def get_documents(transcripts: List[Tuple[str, str]],
                  open_ai_model: str) -> List[Document]:
    enc = tiktoken.encoding_for_model(open_ai_model)
    model_max_token_len = get_model_max_len(open_ai_model)

    documents = []
    for id_, t in transcripts:
        encoded_transcript = enc.encode(t)
        if len(encoded_transcript) > model_max_token_len:
            documents.extend([
                Document(
                    page_content=enc.decode(encoded_transcript[i:i + model_max_token_len]),
                    metadata={"source": "https://www.youtube.com/watch?v=" + id_}
                )
                for i in range(0, len(encoded_transcript), model_max_token_len)
            ]
            )
        else:
            documents.append(Document(page_content=t, metadata={"source": "https://www.youtube.com/watch?v=" + id_}))
    return documents
'''

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


'''
def get_chain_for_summary(documents, keywords):
    # This controls how each document will be formatted. Specifically,
    # it will be passed to `format_document` - see that function for more
    # details.

    summary_keywords = ", ".join(keywords)

    llm_1 = OpenAI(model_name=open_ai_model)

    per_document_prompt = PromptTemplate(
        template=per_document_prompt_template,
        input_variables=["context", "summary_keywords"]
    )

    map_chain = LLMChain(llm=llm_1, prompt=per_document_prompt)

    llm_2 = OpenAI(model_name=open_ai_model)
    reduce_summary_prompt = PromptTemplate(
        template=combine_document_prompt_template,
        input_variables=["doc_summaries", "summary_keywords"]
    )

    reduce_chain = LLMChain(llm=llm_2, prompt=reduce_summary_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=model_max_token_len,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="context",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    return map_reduce_chain.run(input_documents=documents, summary_keywords=summary_keywords)
'''
