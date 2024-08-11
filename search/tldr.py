import concurrent.futures
import glob
import json
import os
import re
import threading
import requests
import traceback
from typing import Annotated, List, Generator, Optional

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
import httpx
from loguru import logger

import leptonai
from leptonai import Client
from leptonai.kv import KV
from leptonai.photon import Photon, StaticFiles
from leptonai.photon.types import to_bool
from leptonai.util import tool
from datetime import datetime

################################################################################
# Constant values for the RAG model.
################################################################################

# Search engine related. You don't really need to change this.
TWITTER_API_SEARCH="https://api.twitter.com/2/tweets/search/recent"

# Specify the number of references from the search engine you want to use.
# 8 is usually a good number.
REFERENCE_COUNT = 8

# Specify the default timeout for the search engine. If the search engine
# does not respond within this time, we will return an error.
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5

# If the user did not provide a query, we will use this default query.
_default_query = "What's the best place to earn interest in crypto?"

# This is really the most important part of the rag model. It gives instructions
# to the model on how to generate the answer. Of course, different models may
# behave differently, and we haven't tuned the prompt to make it optimal - this
# is left to you, application creators, as an open problem.
_rag_query_text = """
You are an AI assistant called TLDR AI, designed to help onboard people into crypto for a better and decentralized world. You are given a user question, and please write a concise and accurate answer to the question. You will be given a set of related contexts to the question.

Your answer must be correct and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens.

Do not give any information that is not related to the question, and do not repeat content, especially in lists and instructions. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

# A set of stop words to use - this is not a complete set, and you may want to
# add more given your observation.
stop_words = [
    "<|im_end|>",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]

# This is the prompt that asks the model to generate related questions to the
# original question and the contexts.
# Ideally, one want to include both the original question and the answer from the
# model, but we are not doing that here: if we need to wait for the answer, then
# the generation of the related questions will usually have to start only after
# the whole answer is generated. This creates a noticeable delay in the response
# time. As a result, and as you will see in the code, we will be sending out two
# consecutive requests to the model: one for the answer, and one for the related
# questions. This is not ideal, but it is a good tradeoff between response time
# and quality.
_more_questions_prompt = """
You are a helpful assistant that helps the user to ask related questions about learning about using crypto protocols (not as much how they work), based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like crypto events, crypto company names, and crypto twitter celebrities are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Euler hack", in the follow up question, do not just say "the hack", but use the full name "the Euler hack". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

Remember, based on the original question and related contexts, suggest five such further questions. Do NOT repeat the original question, and NEVER include the answer as another question. Each related question should be no longer than 20 words. Here is the original question:
"""

def process_user_query(client: any, tools: any, data: any, message: str):
    """
    Search for the best protocol based on the user query.
    """

    # since we cannot use function calling, we will have to manually ask the model to identify arguments
    # and then use the arguments to call the function to get the best protocol
    messages = [
        {"role": "system", "content": f"""
         You are an assistant that generates JSON. You always return just the JSON with no additional description or context.

         Format:
         {{
            "function": {{
                "type": "string",
                "description": "The function from the list of tools available to be used to get an answer for the user's query"
            }},
            "args": {{
                "type": "object",
                "description": "The arguments to be used for the function, fetched from the schema of the function tool."
            }}
         }}

         Example:
         {{
            function: "get_best_protocol",
            args: {{
                "chain": "Ethereum",
                "category": "Bridge"
            }}
         }}
         
         Based on the user's query, find the relevant function to be used, as well as the arguments for the function.
         The functions follow OpenAI's function tool specification.
         
         Here are the available tools:

         {tools}
         """},
        {"role": "user", "content": message},
    ]

    res = client.chat.completions.create(
        model="mistralai/mixtral-8x7b-instruct-v0.1",
        messages=messages,
        max_tokens=256,
        stop=stop_words,
        stream=False,
        temperature=0.9,
    )

    res = json.loads(res.choices[0].message.content)

    logger.info(f"Querying... Function: {res['function']}, Args: {res['args']}")

    if 'function' in res:
        if res['function'] == "get_best_protocol":
            return search_with_llama(data['protocols'], res['args'])
        if res['function'] == "get_related_events":
            return search_with_events(data['events'], res['args'])
        if res['function'] == "get_best_yields":
            return search_with_pools(data['protocols'], data['pools'], res['args'])
        # if res['function'] == "get_twitter_posts":
        #     return search_with_twitter(res['args'])

    return []

def search_with_llama(data: List[any], filter: any):
    """
    Read from local DefiLlama data.
    """

    protocols = []

    if 'category' not in filter:
        return []
    
    filterChain = True

    if 'chain' not in filter:
        filterChain = False

    for protocol in data:
        if protocol is None:
            continue

        if 'chain' in protocol and 'category' in protocol:
            if filter['category'] in protocol["category"]:
                if(filterChain):
                    if (filter['chain'] in protocol["chain"] or filter['chain'] in protocol["chains"]):
                        protocols.append(protocol)
                else:
                    protocols.append(protocol)
        else:
            continue

    protocols = sorted(protocols, key=lambda x: x["tvl"], reverse=True)

    return protocols[:5]

def search_with_events(events: List[any], filter: any):
    """
    Search for events based on the user query.
    """

    results = []

    date_format = "%Y-%m-%d"
    
    startDate = None
    if('startDate' in filter):
        startDate = datetime.strptime(filter['startDate'])
    
    endDate = None
    if('endDate' in filter):
        endDate = datetime.strptime(filter['endDate'])

    tags = None
    if('tags' in filter):
        tags = filter['tags']

    topics = None
    if('topics' in filter):
        topics = filter['topics']

    location = None
    if('location' in filter):
        location = filter['location']

    for event in events:
        if event is None:
            continue

        if startDate is not None:
            if datetime.strptime(event["date"], date_format) < startDate:
                continue
            
        if endDate is not None:
            if datetime.strptime(event["date"], date_format) > endDate:
                continue
        
        if tags is not None:
            for tag in tags:
                if tag not in event["tags"]:
                    continue

        if topics is not None:
            for topic in topics:
                if topic not in event["topics"]:
                    continue
        
        if location is not None:
            if location['country'] not in event['country']:
                continue

            if location['city'] not in event['city']:
                continue

        results.append(event)

    return results[:5]

def search_with_pools(protocols: List[any], pools: List[any], filter: any):
    """
    Search for the best pools based on the user query.
    """

    results = []

    chain = "Mode"
    if 'chain' in filter:
        chain = filter['chain']

    project = "ionic-protocol"
    if 'project' in filter:
        project = filter['project']

    for pool in pools:
        if pool is None:
            continue

        if 'chain' not in pool or 'project' not in pool:
            continue

        if chain not in pool["chain"]:
            continue

        if project not in pool["project"]:
            continue
        
        for protocol in protocols: 
            if pool['project'] == protocol['slug']:
                pool['protocol'] = protocol
                break

        results.append(pool)

    results = sorted(results, key=lambda x: x["apy"], reverse=True)

    return results[:5]

class RAG(Photon):
    """
    Retrieval-Augmented Generation Demo from Lepton AI.

    This is a minimal example to show how to build a RAG engine with Lepton AI.
    It uses search engine to obtain results based on user queries, and then uses
    LLM models to generate the answer as well as related questions. The results
    are then stored in a KV so that it can be retrieved later.
    """

    requirement_dependency = [
        "openai",  # for openai client usage.
        "tweepy"
    ]

    extra_files = [
        *glob.glob("dist/**/*", recursive=True),
        "search/protocols.json"
    ]

    deployment_template = {
        # All actual computations are carried out via remote apis, so
        # we will use a cpu.small instance which is already enough for most of
        # the work.
        "resource_shape": "cpu.small",
        # You most likely don't need to change this.
        "env": {
            # Specify the LLM model you are going to use.
            "LLM_MODEL": "mixtral-8x7b",
            # For all the search queries and results, we will use the Lepton KV to
            # store them so that we can retrieve them later. Specify the name of the
            # KV here.
            "KV_NAME": "search-with-lepton",
            # If set to true, will generate related questions. Otherwise, will not.
            "RELATED_QUESTIONS": "true",
            # On the lepton platform, allow web access when you are logged in.
            "LEPTON_ENABLE_AUTH_BY_COOKIE": "true",
        },
        # Secrets you need to have: search api subscription key, and lepton
        # workspace token to query lepton's llama models.
        "secret": [
            # You need to specify the workspace token to query lepton's LLM models.
            "LEPTON_WORKSPACE_TOKEN",
        ],
    }

    data = {}

    # It's just a bunch of api calls, so our own deployment can be made massively
    # concurrent.
    handler_max_concurrency = 16

    # TODO: Heurist LLM API does not support function tools yet
    def local_function_client(self):
        """
        Gets a thread-local client, so in case openai clients are not thread safe,
        each thread will have its own client.
        """
        import openai

        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            thread_local.client = openai.OpenAI(
                base_url=f"https://{self.model}.lepton.run/api/v1/",
                api_key=os.environ.get("LEPTON_WORKSPACE_TOKEN"),
                # We will set the connect timeout to be 10 seconds, and read/write
                # timeout to be 120 seconds, in case the inference server is
                # overloaded.
                timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            )
            return thread_local.client
        
    def local_heurist_client(self):
        """
        Gets a thread-local client for the Heurist API.
        """
        import openai

        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            thread_local.client = openai.OpenAI(
                base_url=f"https://llm-gateway.heurist.xyz",
                api_key=os.environ.get("HEURIST_AUTH_KEY"),
                # We will set the connect timeout to be 10 seconds, and read/write
                # timeout to be 120 seconds, in case the inference server is
                # overloaded.
                timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            )
            return thread_local.client

    def init(self):
        """
        Initializes photon configs.
        """
        # First, log in to the workspace.
        self.model = os.environ["LLM_MODEL"]
        # An executor to carry out async tasks, such as uploading to KV.
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.handler_max_concurrency * 2
        )
        # Create the KV to store the search results.
        logger.info("Creating KV. May take a while for the first time.")
        self.kv = KV(
            os.environ["KV_NAME"], create_if_not_exists=True, error_if_exists=False
        )
        # whether we should generate related questions.
        self.should_do_related_questions = to_bool(os.environ["RELATED_QUESTIONS"])

        with open("search/protocols.json", "r") as f:
            file = f.read()
            self.data['protocols'] = json.loads(file)

        with open("search/tools.json", "r") as f:
            file = f.read()
            self.tools = file

        with open("search/events.json", "r") as f:
            file = f.read()
            self.data['events'] = json.loads(file)

        with open("search/pools.json", "r") as f:
            file = f.read()
            self.data['pools'] = json.loads(file)

    def get_related_questions(self, query, contexts):
        """
        Gets related questions based on the query and context.
        """

        def ask_related_questions(
            questions: Annotated[
                List[str],
                [(
                    "question",
                    Annotated[
                        str, "related question to the original question and context."
                    ],
                )],
            ]
        ):
            """
            ask further questions that are related to the input and output.
            """
            pass

        try:
            response = self.local_function_client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": _more_questions_prompt.format(
                            context="\n\n".join([json.dumps(c) for c in contexts])
                        ),
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                tools=[{
                    "type": "function",
                    "function": tool.get_tools_spec(ask_related_questions),
                }],
                max_tokens=512,
            )
            related = response.choices[0].message.tool_calls[0].function.arguments
            if isinstance(related, str):
                related = json.loads(related)
            logger.trace(f"Related questions: {related}")
            return related["questions"][:5]
        except Exception as e:
            # For any exceptions, we will just return an empty list.
            logger.error(
                "encountered error while generating related questions:"
                f" {e}\n{traceback.format_exc()}"
            )
            return []

    def _raw_stream_response(
        self, contexts, llm_response, related_questions_future
    ) -> Generator[str, None, None]:
        """
        A generator that yields the raw stream response. You do not need to call
        this directly. Instead, use the stream_and_upload_to_kv which will also
        upload the response to KV.
        """
        # First, yield the contexts.
        yield json.dumps(contexts)
        yield "\n\n__LLM_RESPONSE__\n\n"
        # Second, yield the llm response.
        if not contexts:
            # Prepend a warning to the user
            yield (
                "(The search engine returned nothing for this query. Please take the"
                " answer with a grain of salt.)\n\n"
            )
        for chunk in llm_response:
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""
        # Third, yield the related questions. If any error happens, we will just
        # return an empty list.
        if related_questions_future is not None:
            related_questions = related_questions_future.result()
            try:
                result = json.dumps(related_questions)
            except Exception as e:
                logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
                result = "[]"
            yield "\n\n__RELATED_QUESTIONS__\n\n"
            yield result

    def stream_and_upload_to_kv(
        self, contexts, llm_response, related_questions_future, search_uuid
    ) -> Generator[str, None, None]:
        """
        Streams the result and uploads to KV.
        """
        # First, stream and yield the results.
        all_yielded_results = []
        for result in self._raw_stream_response(
            contexts, llm_response, related_questions_future
        ):
            all_yielded_results.append(result)
            yield result
        # Second, upload to KV. Note that if uploading to KV fails, we will silently
        # ignore it, because we don't want to affect the user experience.
        _ = self.executor.submit(self.kv.put, search_uuid, "".join(all_yielded_results))

    @Photon.handler(method="POST", path="/query")
    def query_function(
        self,
        query: str,
        search_uuid: str,
        generate_related_questions: Optional[bool] = True,
    ) -> StreamingResponse:
        """
        Query the search engine and returns the response.

        The query can have the following fields:
            - query: the user query.
            - search_uuid: a uuid that is used to store or retrieve the search result. If
                the uuid does not exist, generate and write to the kv. If the kv
                fails, we generate regardless, in favor of availability. If the uuid
                exists, return the stored result.
            - generate_related_questions: if set to false, will not generate related
                questions. Otherwise, will depend on the environment variable
                RELATED_QUESTIONS. Default: true.
        """
        # Note that, if uuid exists, we don't check if the stored query is the same
        # as the current query, and simply return the stored result. This is to enable
        # the user to share a searched link to others and have others see the same result.
        if search_uuid:
            try:
                result = self.kv.get(search_uuid)

                def str_to_generator(result: str) -> Generator[str, None, None]:
                    yield result

                return StreamingResponse(str_to_generator(result))
            except KeyError:
                # do nothing
                pass
            except Exception as e:
                logger.error(
                    f"KV error: {e}\n{traceback.format_exc()}, will generate again."
                )
        else:
            raise HTTPException(status_code=400, detail="search_uuid must be provided.")

        # First, do a search query.
        query = query or _default_query
        # Basic attack protection: remove "[INST]" or "[/INST]" from the query
        query = re.sub(r"\[/?INST\]", "", query)
        
        client = self.local_heurist_client()

        self.search_function = lambda query: process_user_query(
            client,
            self.tools,
            self.data,
            query,
        )
        
        # Search function is just asking Heurist AI's OpenAI Client for function calls via tools
        contexts = self.search_function(query)

        print(contexts)

        system_prompt = _rag_query_text.format(
            context="\n\n".join(
                [f"{c}" for _, c in enumerate(contexts)]
            )
        )

        try:
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            llm_response = client.chat.completions.create(
                model="mistralai/mixtral-8x7b-instruct-v0.1",
                messages=messages,
                max_tokens=1024,
                stop=stop_words,
                stream=True,
                temperature=0.9,
            )


            if self.should_do_related_questions and generate_related_questions:
                # While the answer is being generated, we can start generating
                # related questions as a future.
                related_questions_future = self.executor.submit(
                    self.get_related_questions, query, contexts
                )
            else:
                related_questions_future = None
        except Exception as e:
            logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
            return HTMLResponse("Internal server error.", 503)

        return StreamingResponse(
            self.stream_and_upload_to_kv(
                contexts, llm_response, related_questions_future, search_uuid
            ),
            media_type="text/html",
        )

    @Photon.handler(mount=True)
    def ui(self):
        return StaticFiles(directory="dist/")

if __name__ == "__main__":
    rag = RAG()
    rag.launch()