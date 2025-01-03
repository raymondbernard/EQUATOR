import os
import re
import json
import requests
import chromadb
import time
from loguru import logger
from dotenv import load_dotenv
from charting import create_performance_chart
from utils import get_llm_stats, load_all_llm_answers_from_json
from openai import OpenAI
from groq import Groq
from bak.unloadOllama import unload_ollama_services, reload_ollama_services
# import csv
import sqlite3
from datetime import datetime  # Correct import
import pandas as pd
from IPython.display import display

load_dotenv()

logger.add("vectordb.log", format="{time} {level} {message}", filter="", level="INFO")

system_prompt = r"""SCORING CRITERIA
100%: The response must match the answer key given, even if the student used bad reasoning, logic to arrive at the final answer. 
0%: The does NOT match the answer key given. No partial credit allowed!
You must only return a JSON object with the score {'score': <0 or 100>}

TASK

Evaluate whether the STUDENT Answer matches the answer key given. If it does, assign a score of 100% Otherwise you must assign a score of 0%.  Provide a very short explanation on why.  
Just focus on the student's final answer!  Give full credit to the student if final answer matches answer key. Don't over think this.  Also do not evaluate based on the quality, logical reasoning , even if it is very persuasive!
Only consider the answer key as the source of true.  Your job is at risk of you do not follow our instructions.  If it the Answer Key match the students answer you must assign a score of 0%, no partial credit allowed.    
Return a JSON object with the explanation of why the student got his\her score == {"score": <0 or 100>}.  Keep it do less than two sentences {"evaluation": "<explanation>"}
You must only return a JSON object with the score {'score': <0 or 100>}
No partial credit allowed!"""

warning_prompt = r"""You think very carefully about the question asked. You make zero assumptions about classic problems. You are not to be tricked! 
Warning: THIS IS NOT THE CLASSIC PROBLEM! \n"""


def extract_model_parts(model_string):
    # Define the regex pattern to extract both parts
    pattern = r"^([^/]+)/([^/:]+)"
    # Use re.match to find the model parts
    match = re.match(pattern, model_string)
    if match:
        return match.group(1), match.group(2)
    return None, None


def sanitize_string(value):
    """
    Escapes curly braces in strings to prevent issues with format specifiers in logging.
    """
    if isinstance(value, str):
        return value.replace("{", "{{").replace("}", "}}")
    return value


def create_template_json(
    student_model,
    output_path,
    question_id,
    category,
    human_answer,
    question,
    student_answer,
    evaluator_response,
    score,
):
    # Ensure the directory for the output path exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(student_model)
    # Load existing data if the file exists
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as infile:
                template_data = json.load(infile)
        except (json.JSONDecodeError, FileNotFoundError):
            template_data = {}  # Start fresh if file is empty or corrupted
    else:
        template_data = {}

    # Define or update the structure of the template JSON
    template_data[question_id] = {
        "category": category,
        "question": question,
        "human_answer": human_answer,
        "model_answer": student_answer,
        "eval_response": evaluator_response,
        "score": score,
    }

    # Write the updated data back to the file
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(template_data, json_file, indent=4, ensure_ascii=False)

    print(f"Template JSON created/updated: {output_path}")
def unload_model(model):
    keep_alive = 0
    url = "http://localhost:11434/api/generate"
    payload = json.dumps({
        "model": model,
        "keep_alive": keep_alive
    })
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    

def stream_response(response):
    """
    Handle streaming response and capture key content.
    """
    content = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            print(decoded_line)  # Print each line of the streaming response
            content += decoded_line
    return content


class EQUATOR_Client(object):
    def __init__(self, execution_steps, student_model, ollama_EQUATOR_evaluator_model, groq_EQUATOR_evaluator_model):
        self.Groq_EQUATOR_evaluator_model = groq_EQUATOR_evaluator_model
        self.local_student = student_model
        self.Ollama_EQUATOR_evaluator = ollama_EQUATOR_evaluator_model
        self.base_url = "http://localhost:11434" # in container 
        self.student_base_url = "http://localhost:11434"
        self.execution_steps = execution_steps
        self.chroma_client = chromadb.PersistentClient(path=".")

    # def generate_completion(
    #     self, model, prompt, system=None, stream=False, suffix=None, options=None
    # ):
    #     url = f"{self.base_url}/api/generate"
    #     payload = {"model": model, "prompt": prompt, "stream": stream}
    #     if system:
    #         payload["system"] = system
    #     if suffix:
    #         payload["suffix"] = suffix
    #     if options:
    #         payload["options"] = options

    #     try:
    #         response = requests.post(url, json=payload)
    #         response.raise_for_status()  # Raise an error for bad responses
    #         if response.headers.get("Content-Type").startswith("application/json"):
    #             return response.json()
    #         else:
    #             logger.error(
    #                 f"Unexpected response content type: {response.headers.get('Content-Type')}"
    #             )
    #             logger.error(f"Response content: {response.text}")
    #             return None
    #     except requests.RequestException as e:
    #         logger.error(f"Failed to generate completion: {e}")
    #         return None

    def EQUATOR_Controller(
        self,
        model_path,
        lab,
        student_models,
        answer_save_path_round,
        count,
        prefix_replace,
    ):
        print("prefix ==", prefix_replace)

        chroma_client = chromadb.PersistentClient(path=".")  # ChromaDB client

        if self.local_student:
            pass

        ## Go through the vector db chroma.sqlite3 for all the question and id

        # Path to your chroma.sqlite3 file
        db_path = "chroma.sqlite3"

        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        batch_size = 1  # Define your batch size
        offset = 0  # Start offset
        stop_processing = False  # Flag to stop the outer loop

        while True:
            # Fetch a batch of results
            query = f"""
            SELECT 
                json_extract(string_value, '$.id'), 
                json_extract(string_value, '$.category'), 
                json_extract(string_value, '$.question'), 
                json_extract(string_value, '$.response') 
            FROM embedding_fulltext_search
            LIMIT {batch_size} OFFSET {offset}
            """
            print(f"Executing query with OFFSET={offset}, LIMIT={batch_size}")
            cursor.execute(query)
            results = cursor.fetchall()

            # Break the loop if no more records are fetched
            if not results:
                print("No more records found. Exiting.")
                break

            for row in results:
                question_id, category, question_text, response = row
                print(
                    f"Processing ID: {question_id}, Category: {category}, Question: {question_text}, Answer: {response}"
                )

                # Simulate stripping and processing text
                question = question_text.strip() if question_text else ""
                human_answer = response.strip() if response else ""

                # Extract score if evaluator_response is valid
                # if evaluator_response:
                # try:
                for student_model in student_models:
                    output_path = f"{answer_save_path_round}/round_{count + 1}/{prefix_replace}{lab+"-"}{student_model}.json"
                    print("line 203 Model Path  = ", model_path)

                    # Call your evaluator function
                    student_answer, evaluator_response = self.call_evaluator(
                        model_path=model_path,
                        prompt=question,
                    )
                    print("line 211, unpacking student answer = ", student_answer)
                    print("line 212, unpacking evaluator  = ", evaluator_response)

                    score = self.extract_score_from_string(evaluator_response)

                    create_template_json(
                        student_model,
                        output_path,
                        question_id,
                        category,
                        human_answer,
                        question,
                        student_answer,
                        evaluator_response,
                        score,
                    )

                # except:
                #     print(f"Evaluation failed for Question ID {question_id}.")
                #     evaluator_response = "Evaluation failed"
                #     score = "N/A"

                # Stop processing if a condition is met
                if question_id == "1":  # Replace "1" with the desired stop condition
                    print("Stop condition met. Exiting.")
                    stop_processing = True  # Set the flag to stop outer loop
                    break

            # Increment offset to fetch the next batch
            if stop_processing:
                print("Breaking the outer loop.")
                break
            offset += batch_size  # Move to the next batch

        # Close the database connection after processing
        conn.close()
        print("Database connection closed.")

    # def generate_chat(self, model, messages, system=None, stream=False):
    #     print("line 249, self.evaluation_steps =", self.execution_steps)

    #     if "ollama_to_openrouter_evaluate" in self.execution_steps:
    #         url = f"{self.base_url}/api/chat"
    #         payload = {"model": model, "messages": messages, "stream": stream}

    #     if "local_llm_evaluate" in self.execution_steps:
    #         url = f"{self.student_base_url}/api/chat"
    #         model = self.local_student
    #         payload = {"model": model, "messages": messages, "stream": stream}
    #         print("line 257, payload and model", payload, model)
    #     if system:
    #         payload["system"] = system

    #     try:
    #         response = requests.post(url, json=payload)
    #         response.raise_for_status()  # Raise an error for bad responses
    #         if response.headers.get("Content-Type").startswith("application/json"):
    #             return response.json()
    #         else:
    #             logger.error(
    #                 f"Unexpected response content type: {response.headers.get('Content-Type')}"
    #             )
    #             logger.error(f"Response content: {response.text}")
    #             return None
    #     except requests.RequestException as e:
    #         logger.error(f"Failed to generate chat: {e}")
    #         return None

    def generate_embeddings(self, model, input_text, truncate=True):
        # print("line 277 model ", model)
        # print(
        #     "line 279 local_llm_evaluation in self.execution_steps =",
        #     self.execution_steps,
        # )
        if "ollama_to_openrouter_evaluate" or "ollam_to_groq_evaluate" in self.execution_steps:
            url = f"{self.base_url}/api/embed"
            payload = {"model": model, "input": input_text, "truncate": truncate}
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()  # Raise an error for bad responses
                if response.headers.get("Content-Type").startswith("application/json"):
                    response_json = response.json()
                    if "embeddings" in response_json:
                        return response_json
                    else:
                        logger.error(
                            f"No embeddings found in response: {response_json}"
                        )
                        return None
                else:
                    logger.error(
                        f"Unexpected response content type: {response.headers.get('Content-Type')}"
                    )
                    logger.error(f"Response content: {response.text}")
                    return None
            except requests.RequestException as e:
                logger.error(f"Failed to generate embeddings: {e}")
                return None

        if "local_llm_evaluate" in self.execution_steps:
            url = f"{self.student_base_url}/api/embed"
            payload = {"model": model, "input": input_text, "truncate": truncate}
            # print("line 311, payload = ", payload)
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()  # Raise an error for bad responses
                if response.headers.get("Content-Type").startswith("application/json"):
                    response_json = response.json()
                    if "embeddings" in response_json:
                        return response_json
                    else:
                        logger.error(
                            f"No embeddings found in response: {response_json}"
                        )
                        return None
                else:
                    logger.error(
                        f"Unexpected response content type: {response.headers.get('Content-Type')}"
                    )
                    logger.error(f"Response content: {response.text}")
                    return None
            except requests.RequestException as e:
                logger.error(f"Failed to generate embeddings: {e}")
                return None

    def add_to_vector_db(self, vector_db, entry_id, serialized_conversations, metadata):
        response = self.generate_embeddings(
            model="all-minilm", input_text=serialized_conversations
        )

        if not response or "embeddings" not in response:
            logger.error(
                f"Failed to retrieve embeddings for entry {entry_id}. Response: {response}"
            )
            return
        # Flatten the embedding if it is nested
        embedding = self.flatten_embedding(response["embeddings"])

        converted_metadata = {
            k: self.convert_metadata_value(v) for k, v in metadata.items()
        }

        try:
            vector_db.add(
                ids=[str(entry_id)],
                embeddings=[embedding],
                documents=[serialized_conversations],
                metadatas=[converted_metadata],
            )
        except Exception as e:
            logger.error(f"Error adding entry {entry_id} to the vector DB: {e}")

    def create_vector_db(self, conversations):
        vector_db_name = "conversations"
        try:
            self.chroma_client.delete_collection(name=vector_db_name)
        except ValueError:
            pass  # Handle collection not existing
        vector_db = self.chroma_client.create_collection(name=vector_db_name)
        for c in conversations:
            serialized_conversations = json.dumps(c)
            self.add_to_vector_db(vector_db, c["id"], serialized_conversations, c)

    def convert_metadata_value(self, value):
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return value

    def flatten_embedding(self, embedding):
        # Flatten nested embeddings if necessary
        if isinstance(embedding[0], list):
            return [item for sublist in embedding for item in sublist]
        return embedding

    def add_to_vector_db(self, vector_db, entry_id, serialized_conversations, metadata):
        response = self.generate_embeddings(
            model="all-minilm", input_text=serialized_conversations
        )
        if not response or "embeddings" not in response:
            logger.error(
                f"Failed to retrieve embeddings for entry {entry_id}. Response: {response}"
            )
            return
        # Flatten the embedding if it is nested
        embedding = self.flatten_embedding(response["embeddings"])

        converted_metadata = {
            k: self.convert_metadata_value(v) for k, v in metadata.items()
        }

        try:
            vector_db.add(
                ids=[str(entry_id)],
                embeddings=[embedding],
                documents=[serialized_conversations],
                metadatas=[converted_metadata],
            )
        except Exception as e:
            logger.error(f"Error adding entry {entry_id} to the vector DB: {e}")

    def retrieve_embedding(self, prompt, n_results=1):
        # print("line 411 retrieve embedding prompt  ==", prompt)
        response = self.generate_embeddings(model="all-minilm", input_text=prompt)
        # print("line 413 Generate Embeddings ==", response)
        if not response or "embeddings" not in response:
            logger.error("Failed to retrieve embeddings from the model.")
            return None
        prompt_embedding = self.flatten_embedding(response["embeddings"])
        vector_db = self.chroma_client.get_collection(name="conversations")
        try:
            results = vector_db.query(
                query_embeddings=[prompt_embedding], n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector DB: {e}")
            return None

    
    # Generate student answer
    def student(self, model_path, full_prompt_student, retries=3, delay=2):
        """
        Call the student API, retrying if the response is None.

        Args:
            model_path (str): The model path to use for the completion.
            full_prompt (str): The prompt to send to the API.
            retries (int): The number of times to retry if the API returns None.
            delay (int): The delay in seconds between retries.
        Returns:
            str: The response content from the API.
        """
        model_path = str(model_path)
        # print("line 442 - Model Path = ", model_path)
        print("line 468 , in Execution steps =", self.execution_steps)

        if "ollama_to_openrouter_evaluate" or "groq_to_openrouter_evaluate" in self.execution_steps:
            api_key = os.environ.get("OPENROUTER_KEY")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            model_path = str(model_path)
            # print("line 248 - Model Path = ", model_path)
            for attempt in range(retries):
                try:
                    completion = client.chat.completions.create(
                        model=model_path,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a student who is being tested, please follow the directions given exactly. You are welcomed to reason through the question"
                                + "You must return only your final answer in a JSON Object example  {'student_answer':'<My final Answer here>'}",
                            },
                            {"role": "user", "content": warning_prompt + full_prompt_student},
                        ],
                    )

                    response = completion.choices[0].message.content
                    if response:
                        return response
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: Received None response"
                        )

                except Exception as e:
                    logger.error(
                        f"Attempt {attempt + 1} failed: Failed to get response from Openrouter.ai API: {e}"
                    )

                if attempt < retries - 1:
                    time.sleep(delay)

            logger.error("All retry attempts failed")
            return None
        
       
        elif "ollama_to_groq_evaluate" in self.execution_steps:
            api_key = os.environ.get("GROQ_API_KEY")
            print("groq api key: " + api_key)

            client_student = Groq(api_key=os.environ.get("GROQ_API_KEY"),)

            for attempt in range(retries):
                try:
                    completion = client_student.chat.completions.create(
                        temperature=0,
                        model=self.Groq_EQUATOR_evaluator_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a student who is being tested, please follow the directions given exactly. You are welcomed to reason through the question"
                                + "You must return only your final answer in a JSON Object example  {'student_answer':'<My final Answer here>'}",
                            },
                            {"role": "user", "content": warning_prompt + full_prompt_student},
                        ],
                    )

                    response = completion.choices[0].message.content
                    if response:
                        return response
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: Received None response"
                        )

                except Exception as e:
                    logger.error(
                        f"Attempt {attempt + 1} failed: Failed to get response from Groq  API: {e}"
                    )
                if attempt < retries - 1:
                    time.sleep(delay)

            logger.error("All retry attempts failed")
            return None
        
        elif "groq_to_ollama_evaluate" in self.execution_steps:
            url = "http://localhost:11434/api/chat"
            payload = {
                "model": "llama3.2",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a student who is being tested, please follow the directions given exactly. You are welcomed to reason through the question "
                                   + "You must return only your final answer in a JSON Object example {'student_answer':'<My final Answer here>'}",
                    },
                    {"role": "user", "content": warning_prompt + full_prompt_student},
                ]
            }
            for attempt in range(retries):
                try:
                    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, stream=True)
                    response.raise_for_status()
                    
                    complete_message = ""
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            complete_message += chunk["message"]["content"]
                            if chunk.get("done"):
                                break

                    if complete_message:
                        return {"student_answer": complete_message}
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed: Missing 'student_answer' in response")

                except requests.RequestException as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")

                if attempt < retries - 1:
                    time.sleep(delay)

            logger.error("All retry attempts failed")
            return None

        # else:
        #     logger.warning("Execution step 'ollama_to_groq_evaluate' not found in execution_steps.")
        #     return None
        

                    # models = ["llama3.2:latest", "all-minilm:latest"]

                    # for model in models:
                    #     result = unload_model(model)
                    #     print(json.dumps(result, indent=4))
                    # time.sleep(5)

                    # unload_ollama_services()

                    # time.sleep(5)
                    # reload_ollama_services()
                    # time.sleep(5)

        

    def extract_score_from_string(self, response_string):
        # Regular expressions to match different patterns that indicate a score
        patterns = [
            r"\"score\"\s*:\s*(\d+)",  # JSON-like: "score": 0 or "score":0
            r"'score':\s*(\d+)",  # Python dict-like: {'score': 0}
            r"'grade':\s*(\d+)",  # Python dict-like: {'grade': 0}
            r"Grade:\s*(\d+)",  # Grade without ratio, e.g., Grade: 0
            r"Grade:\s*{'score':\s*(\d+)}",  # Grade followed by Python dict, e.g., Grade: {'score': 0}
            r"Score:\s*{'score':\s*(\d+)}",  # Score followed by Python dict, e.g., Score: {'score': 0}
            r"\*\*Score:\*\*\s*{'score':\s*(\d+)}",  # Markdown Score followed by Python dict, e.g., **Score:** {'score': 20}
            r"\*\*Grade:\*\*\s*{'score':\s*(\d+)}",  # Markdown Grade followed by Python dict, e.g., **Grade:** {'score': 0}
            r"score\s*is\s*(\d+)%",  # Plain text: score is 0%
            r"score\s*of\s*\*\*(\d+)%\*\*",  # Markdown: score of **0%**
            r"the\s*score\s*assigned\s*is\s*(\d+)%",  # Assigned score: the score assigned is 0%
            r"Grade:\s*A\s*\(\s*(\d+)%\)",  # Grade with percentage, e.g., Grade: A (100%)
            r"Grade:\s*[F]\s*\(\s*(\d+)/\d+\)",  # Grade F with ratio, e.g., Grade: F (0/10)
            r"Grade:\s*(\d+)/\d+",  # Ratio format, e.g., Grade: 0/10
            r"\*\*Grade:\*\*\s*(\d+)/\d+",  # Markdown style: **Grade:** 0/10
            r"\*\*Grade:\*\*\s*F\s*\(\s*(\d+)/\d+\)",  # Markdown style with grade F: **Grade:** F (0/100)
            r"Grade:\s*\*\*(\d+)/\d+\*\*",  # Markdown format, e.g., **Grade:** 0/10
            r"Grade:\s*F\s*\(\s*(\d+)\s*out\s*of\s*\d+\)",  # Grade F with "out of", e.g., Grade: F (0 out of 10)
            r"You\s*received\s*a\s*score\s*of\s*(\d+)\s*out\s*of\s*\d+",  # Plain text: You received a score of 0 out of 10
            r"\*\*(\d+)/100\s*score\*\*",  # Markdown style, e.g., **100/100 score**
            r"would\s*earn\s*a\s*score\s*of\s*(\d+)",  # Plain text: would earn a score of 100
            r"return\s*a\s*score\s*of\s*(\d+)",  # Plain text: return a score of 0
        ]

        # Iterate over each pattern to find a match
        for pattern in patterns:
            match = re.search(pattern, response_string, re.IGNORECASE)
            if match:
                return int(match.group(1))

        # If no matching score pattern is found, return None
        return None

    def call_evaluator(self, model_path, prompt):
        # print("line 558 question for student look up in vector db == ", prompt)
        results = self.retrieve_embedding(prompt)
        # print("line 560 retrieve embeddings =", results)
        if results is None:
            logger.error("Failed to retrieve similar documents.")
            return None
        # question_id = ""
        context = ""
        if "documents" in results and results["documents"]:
            metadatas = results.get("metadatas", [])[0]
            for metadata in metadatas:
                context += f"Question: {metadata.get('question', '')}\n"
                context += f"Answer: {metadata.get('response', '')}\n\n"
            logger.info(context)
        else:
            logger.warning("No similar documents found.")

        full_prompt_student = (
            f"Instructions: Submit your final answer in a JSON object {{'student_answer': '<My final Answer>'}}. "
            f"Question for Student: {prompt}\n"
            f"Final Answer = {{'student_answer': '<My final Answer>'}}"
        )
        # print("line 580 full prompt for student to answer  =", full_prompt_student)
        # print("line 581  = model_path : ", model_path)

        student_answer = self.student(model_path, full_prompt_student)
        if not student_answer:
            logger.error("Failed to get Student Answer.")
            return None
        logger.info(f"Student Answer: {student_answer}")

        evaluator_system_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a strict high school teacher. "
                    f"Provide the score in the form of a JSON object: {{'score': <0 or 100>}}"
                    "Your role is to grade strictly based on the provided answer key. "
                    "Only examine the final answer given by the student in a JSON object: {'student_answer': <My final Answer>}. "
                    "If the student's answer matches the answer key, assign a score of 100; otherwise, assign a score of 0. "
                    "No partial credit allowed! Do not consider logical arguments or reasoning; they are irrelevant in this case. "
                    f"Provide the score in the form of a JSON object: {{'score': <0 or 100>}}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Answer key: {sanitize_string(context)}\n"
                    f"Student answer: {sanitize_string(student_answer)}"
                ),
            },
        ]

        # response = self.generate_chat(
        #     model="llama3.3", messages=evaluator_system_prompt, stream=False
        # )

        if "ollama_to_openrouter_evaluate" or "ollama_to_groq_evaluate" in execution_steps:
            EQUATOR_client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
            completion_eval = EQUATOR_client.chat.completions.create(
                temperature=0,
                model=self.Ollama_EQUATOR_evaluator,
                messages=evaluator_system_prompt,
            )

            response_eval = completion_eval.choices[0].message.content
            # print("line 622 response_eval =", response_eval)
            # print("line 623 completion_eval =", completion_eval)

            
            if response_eval:
                logger.info(f"Evaluator Full Response Line 624 {response_eval}")
                # if "choices" in response_eval and len(response_eval["choices"]) > 0:
                #     evaluator_response = response_eval["choices"][0]["content"]
                # elif "message" in response_eval and "content" in response_eval["message"]:
                #     evaluator_response = response_eval["message"]["content"]
                # else:
                #     logger.error("Failed to get evaluator response.")
                #     return None
                # logger.info(f"Evaluator Response: {evaluator_response}")

                return student_answer, response_eval
            else:
                logger.error("Failed to get evaluator response.")
                return None
            
        if "groq_to_openrouter" or "gorq_to_ollama_evaluate" in execution_steps:

            api_key = os.environ.get("GROQ_API_KEY")
            EQUATOR_client = OpenAI(base_url="https://api.groq.com/openai/v1/models", api_key=api_key)
            completion_eval = EQUATOR_client.chat.completions.create(
                temperature=0,
                model=self.Groq_EQUATOR_evaluator_model,
                messages=evaluator_system_prompt,
            )

            response_eval = completion_eval.choices[0].message.content
            # print("line 622 response_eval =", response_eval)
            # print("line 623 completion_eval =", completion_eval)

            
            if response_eval:
                logger.info(f"Evaluator Full Response Line 624 {response_eval}")
                # if "choices" in response_eval and len(response_eval["choices"]) > 0:
                #     evaluator_response = response_eval["choices"][0]["content"]
                # elif "message" in response_eval and "content" in response_eval["message"]:
                #     evaluator_response = response_eval["message"]["content"]
                # else:
                #     logger.error("Failed to get evaluator response.")
                #     return None
                # logger.info(f"Evaluator Response: {evaluator_response}")

                return student_answer, response_eval
            else:
                logger.error("Failed to get evaluator response.")
                return None

    def VectorDB_Controller(self, keepVectorDB):
        if not keepVectorDB:
            # open file for vector score
            # Open and load the JSON file
            with open("linguistic_benchmark.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            # Initialize a list to store the parsed conversations
            conversations = []

            # Iterate over the list of dictionaries
            for entry in data:
                parsed_entry = {
                    "id": entry.get(
                        "index", ""
                    ),  # Use "index" as the unique identifier
                    "category": entry.get("category", ""),  # Extract category
                    "question": entry.get("question", ""),  # Extract the question
                    "response": entry.get(
                        "human_answer", ""
                    ),  # Extract the human answer
                }
                conversations.append(parsed_entry)
            logger.info(conversations)
            EQUATOR_Client.create_vector_db(self,conversations)

if __name__ == "__main__":

    execution_steps = [
        # "ollama_to_groq_evaluate", # working
        # "ollama_to_openroute",
        "groq_to_ollama_evaluate"
        # "groq_to_openrouter_evaluate",
        # "generate_statistics",

    ]

    # openrouter_models = [
    #     "google/learnlm-1.5-pro-experimental:free",
    #     "liquid/lfm-40b:free",
    #     "meta-llama/llama-3.2-11b-vision-instruct:free",
    #     "nousresearch/hermes-3-llama-3.1-405b:free",
    #     "qwen/qwen-2-7b-instruct:free",
    #     "microsoft/phi-3-medium-128k-instruct:free",
    # ]

    openrouter_models = [
        "google/learnlm-1.5-pro-experimental:free"
        # "liquid/lfm-40b:free",
        # "meta-llama/llama-3.2-11b-vision-instruct:free",
        # "nousresearch/hermes-3-llama-3.1-405b:free",
        # "qwen/qwen-2-7b-instruct:free",
        # "microsoft/phi-3-medium-128k-instruct:free",
    ]

    
    local_student = ["RayBernard4/llama3.2:latest"]  # local Ollama clients ]

    groq_EQUATOR_evaluator_model = "llama-3.3-70b-versatile"
    ollama_EQUATOR_evaluator_model = "llama3.2:latest"
    student_model = "llama3.2:latest"

    keepVectorDB = True  # Keep vector database
    
    answer_rounds = 2  # Number of rounds of questions to ask each model
    benchmark_name = "Bernard"
    # Change to false if you want a new vector db
    # date_now = "2024-11-26"  # datetime.now().strftime('%Y-%m-%d')
    date_now = datetime.now().strftime("%Y-%m-%d")

    if "groq_to_ollama_evaluate" in execution_steps:
        client = EQUATOR_Client(execution_steps, student_model, ollama_EQUATOR_evaluator_model, groq_EQUATOR_evaluator_model)
        client.VectorDB_Controller(keepVectorDB)
        for model in local_student:
            model_path = model
            lab, student_models = extract_model_parts(model)
            if student_models:
                print(f"Extracted Lab name: {lab}")

                print(f"Extracted model name: {student_models}")
            else:
                print("Model name not found.")
            student_models = [student_models]
            print("1. GETTING EQUATOR Evaluator ANSWERS -Local Student")
            # Change to false if you want a new vector db
            # date_now = "2024-11-26"  # datetime.now().strftime('%Y-%m-%d')
            folder_name = f"{date_now}-{benchmark_name}"
            answers_save_path = f"./{folder_name}/llm_outputs"
            auto_eval_save_path = f"./{folder_name}/auto_eval_outputs"
            stats_save_path = f"./{folder_name}/tables_and_charts"
            for n in range(answer_rounds):
                print(f"\n----- Round: {n+1} of {answer_rounds} -----")
                answer_save_path_round = f"{auto_eval_save_path}"
                client.EQUATOR_Controller(
                    model_path,
                    lab,
                    student_models,
                    answer_save_path_round=answer_save_path_round,
                    count=n,
                    prefix_replace="auto_eval-",
                )

    if "ollama_to_groq_evaluate" in execution_steps:
        client = EQUATOR_Client(execution_steps, student_model, ollama_EQUATOR_evaluator_model, groq_EQUATOR_evaluator_model)
        client.VectorDB_Controller(keepVectorDB)
        for model in local_student:
            model_path = model
            lab, student_model = extract_model_parts(model)
            if student_model:
                print(f"Extracted Lab name: {lab}")
                print(f"Extracted model name: {student_model}")
            else:
                print("Model name not found.")
            student_models = [student_model]
            print("1. GETTING EQUATOR Evaluator ANSWERS -Local Student")
            # Change to false if you want a new vector db
            # date_now = "2024-11-26"  # datetime.now().strftime('%Y-%m-%d')
            folder_name = f"{date_now}-{benchmark_name}"
            answers_save_path = f"./{folder_name}/llm_outputs"
            auto_eval_save_path = f"./{folder_name}/auto_eval_outputs"
            stats_save_path = f"./{folder_name}/tables_and_charts"
            for n in range(answer_rounds):
                print(f"\n----- Round: {n+1} of {answer_rounds} -----")
                answer_save_path_round = f"{auto_eval_save_path}"
                client.EQUATOR_Controller(
                    model_path,
                    lab,
                    student_models,
                    answer_save_path_round=answer_save_path_round,
                    count=n,
                    prefix_replace="auto_eval-",
                )

    if "ollama_to_openrouter_evaluate" in execution_steps:
        for model in openrouter_models:
            model_path = model
            lab, student_model = extract_model_parts(model)
            client = EQUATOR_Client(execution_steps, student_model, ollama_EQUATOR_evaluator_model, groq_EQUATOR_evaluator_model)
            if student_model:
                print(f"Extracted Lab name: {lab}")
                print(f"Extracted model name: {student_model}")
            else:
                print("Model name not found.")
            student_models = [student_model]
            folder_name = f"{date_now}-{benchmark_name}"
            answers_save_path = f"./{folder_name}/llm_outputs"
            auto_eval_save_path = f"./{folder_name}/auto_eval_outputs"
            stats_save_path = f"./{folder_name}/tables_and_charts"
            print("1. GETTING BERNARD LLM Evaluator ANSWERS")
            for n in range(answer_rounds):
                print(f"\n----- Round: {n+1} of {answer_rounds} -----")
                answer_save_path_round = f"{auto_eval_save_path}"
                client.EQUATOR_Controller(
                    model_path,
                    lab,
                    student_models,
                    answer_save_path_round=answer_save_path_round,
                    count=n,
                    prefix_replace="auto_eval-",
                )

    if "generate_statistics" in execution_steps:
        folder_name = f"{date_now}-{benchmark_name}"
        auto_eval_save_path = f"./{folder_name}/auto_eval_outputs"
        stats_save_path = f"./{folder_name}/tables_and_charts"
        sub_eval_folders = [f"/round_{r+1}" for r in range(answer_rounds)]
        print("2. GENERATING STATISTICS")
        all_stats_dfs = {}
        save_info = [
            {
                "path": auto_eval_save_path,
                "chart_title": "LLM Linguistic Benchmark Performance",
                "type": "",
            }
        ]
        for info in save_info:
            save_path = info["path"]
            chart_title = info["chart_title"]
            info_type = info["type"]
            print("Eval for path:", save_path)
            all_llm_evals = load_all_llm_answers_from_json(
                save_path,
                prefix_replace="auto_eval-",
                sub_folders=sub_eval_folders,
            )
            stats_df = get_llm_stats(
                all_llm_evals, stats_save_path, file_suffix=info_type, bootstrap_n=10000
            )
            display(stats_df)
            barplot, plt = create_performance_chart(
                stats_df.reset_index(),
                chart_title,
                highlight_models=["o1-preview"],
            )
            barplot.figure.savefig(
                f"{stats_save_path}/performance_chart{info_type}.png"
            )
            plt.show()
            all_stats_dfs[chart_title] = stats_df
        print("-- DONE STATS --\n")

