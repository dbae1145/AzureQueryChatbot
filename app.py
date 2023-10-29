import openai
import warnings
import pandas as pd
import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import streamlit as st
from azure.storage.blob import BlobServiceClient, generate_container_sas, BlobSasPermissions
import tiktoken
import pyodbc
from datetime import datetime, timedelta
from pdf_processor import PDFProcessor
import secrets
import string
import os
warnings.filterwarnings("ignore")
tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def azure_search(search_client, q, query_vector, filter, filter_by, name, top_results, semantic_search):
    filter_string = f"school eq '{name}'" if filter_by == "school" else f"sourcefile eq '{name}'"
    common_params = {
        'search_text': q,
        'top': top_results,
        'vector': query_vector,
        'top_k': 50 if query_vector else None,
        'vector_fields': 'embedding' if query_vector else None,
    }

    if semantic_search:
        common_params.update({
            'query_language': 'en-us',
            'query_speller': 'lexicon',
            'semantic_configuration_name': 'default',
        })

    if filter:
        common_params['filter'] = filter_string

    r = search_client.search(**common_params)

    # Collect results along with their search.score
    results = [{'sourcefile': doc['sourcefile'],
                'sourcepage': doc['sourcepage'],
                'content': doc['content'],
                'school': doc['school'],
                'score': doc['@search.score']} for doc in r]

    # Sort results by search.score in descending order
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    results_df = pd.DataFrame(results)

    return results_df

def gpt_response(gpt_engine, custom_instruction, content, citation_lst, prompt_cost, completion_cost):
    response = openai.ChatCompletion.create(
        model=gpt_engine,
        messages=[
            {'role': 'system', 'content': custom_instruction},
            {'role': 'user', 'content': content}
        ],
        temperature=0.7,
        max_tokens=2000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    output = response['choices'][0]['message']['content']

    functions_citations = [
        {
            "name": "identify_citations_used",
            "description": "Identify citations that are used in a given prompt",
            "parameters": {
                "type": "object",
                "properties": {

                    "citations_used": {
                        "type": "array",
                        "description": "An array of objects representing citations mentioned in the given prompt",

                        "items": {
                            "type": "object",
                            "properties": {
                                "citation": {
                                    "type": "string",
                                    "description": "Retrieve the citations used in the given prompt",
                                    "enum": citation_lst
                                },
                            }, "required": ["citation"],
                        },
                    },
                },
            },
        }
    ]

    chat_completion_citations = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'system',
                   'content': 'Your task is to identify citations that are mentioned in a given prompt'}] + [
                     {'role': 'user', 'content': output}],
        temperature=0.0,
        max_tokens=3000,
        n=1,
        functions=functions_citations,
        function_call="auto",
    )

    arguments_str = chat_completion_citations['choices'][0]['message']['function_call']['arguments']
    arguments_dict = json.loads(arguments_str)
    citations_list = [citation['citation'] for citation in arguments_dict.get('citations_used', [])]

    prompt_token = response['usage']['prompt_tokens']
    completion_token = response['usage']['completion_tokens']
    actual_cost = prompt_token * prompt_cost + completion_token * completion_cost
    return output, citations_list, actual_cost

def school_identifier(prompt, school_lst):
    system_prompt = "Identify schools in a given prompt. There can be multiple schools. In that case, separate companies by comma."

    functions = [
        {
            "name": "identify_schools",
            "description": "Identify schools in a given prompt",
            "parameters": {
                "type": "object",
                "properties": {
                    "schools": {
                        "type": "string",
                        "description": "Identify schools in a given prompt that are matching the school_lst",
                        "enum": school_lst
                    }
                },
                "required": ["schools"],
            },
        }
    ]

    chat_completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'system', 'content': system_prompt}] + [{'role': 'user', 'content': prompt}],
        temperature=0.0,
        max_tokens=200,
        n=1,
        functions=functions,
        function_call="auto",
    )
    try:
        schools = json.loads(chat_completion.choices[0].message.function_call.arguments).get("schools")
        if type(schools) != list:
            items = schools.split(',')
            items = [item.strip() for item in items]
        else:
            items = schools
    except:
        items = None

    school_identifier_cost = ((chat_completion['usage']['prompt_tokens'] / 1000) * 0.0015) + ((chat_completion['usage']['completion_tokens'] / 1000) * 0.002)

    return items, school_identifier_cost

def search_query_generator(prompt):
    search_query_prompt = """
    Below is a question asked by the user that needs to be answered by searching in a knowledge base about universities.
    You have access to Azure Cognitive Search index with 100's of documents.
    Generate a search query from a given user prompt. Make sure to exclude any school name (e.g., "Find me tuition fees for Harvard and MIT" -> "Tuition fees")
    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    Do not include any special characters like '+'.
    If the question is not in English, translate the question to English before generating the search query.
    If you cannot generate a search query, return just the number 0.

   """

    functions = [
        {
            "name": "search_query_generator",
            "description": "Create a search query to search the Azure Cognitive Search index given user's prompt",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Query string to retrieve documents from azure search eg: 'Tuition fees'",
                    }
                },
                "required": ["search_query"],
            },
        }
    ]

    chat_completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'system', 'content': search_query_prompt}] + [{'role': 'user', 'content': prompt}],
        temperature=0.0,
        max_tokens=32,
        n=1,
        functions=functions,
        function_call="auto",
    )
    try:
        query_text = json.loads(chat_completion.choices[0].message.function_call.arguments).get("search_query")
    except:
        query_text = prompt

    search_query_generator_cost = ((chat_completion['usage']['prompt_tokens'] / 1000) * 0.0015) + ((chat_completion['usage']['completion_tokens'] / 1000) * 0.002)

    return query_text, search_query_generator_cost

# Secret code generator
def generate_code(length=49):
    characters = string.ascii_letters + string.digits
    code = ''.join(secrets.choice(characters) for i in range(length))
    return code

def school_review(df):
    if 'selected_schools' not in st.session_state:
        st.session_state.selected_schools = []
    if 'citations_used' not in st.session_state:
        st.session_state.citations_used = []
    if 'estimated_cost' not in st.session_state:
        st.session_state.estimated_cost = 0
    if 'min_page_numbers' not in st.session_state:
        st.session_state.min_page_numbers = {}
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    openai.api_key = os.environ.get("OPENAI_KEY")

    blob_storage_endpoint = os.environ.get("AZURE_BLOB_ENDPOINT")
    account_key_blob = os.environ.get("AZURE_BLOB_KEY")
    container_name = 'school'
    index_name = 'school'

    blob_service_client = BlobServiceClient(account_url=blob_storage_endpoint)

    sas_token = generate_container_sas(
        blob_service_client.account_name,
        container_name,
        account_key=account_key_blob,
        permission=BlobSasPermissions(read=True, write=True, delete=True, list=True),
        expiry=datetime.utcnow() + timedelta(days=30),
        protocol='https',
        version='2023-07-01-preview'
    )
    search_index_key = os.environ.get("AZURE_SEARCH_INDEX_KEY")

    search_client = SearchClient(
        endpoint=os.environ.get("AZURE_SEARCH_INDEX_ENDPOINT"),
        index_name=index_name,
        credential=AzureKeyCredential(search_index_key))

    st.title("Documents Review")

    st.subheader("Upload a PDF into a Search Index")
    uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)
    st.session_state.uploaded_file = uploaded_file
    submit = st.button("Submit")
    if st.session_state.uploaded_file is not None and submit:
        azure_storage_acct_url = os.environ.get("AZURE_BLOB_STORAGE_SAS_URL")
        storage_container_name = container_name
        azure_search_endpoint = os.environ.get("AZURE_SEARCH_INDEX_ENDPOINT"),
        search_index_key = os.environ.get("AZURE_SEARCH_INDEX_KEY")
        search_index = index_name
        openai_key = os.environ.get("OPENAI_KEY")

        pdf_processor = PDFProcessor(1,
                                     st.session_state.uploaded_file,
                                     azure_storage_acct_url,
                                     storage_container_name,
                                     azure_search_endpoint,
                                     search_index_key,
                                     search_index,
                                     openai_key,
                                     "School")
        pdf_processor.run()

    st.subheader("Citation")
    selectbox_container = st.container()
    with selectbox_container:
        citation_viewer = st.selectbox(
            "Choose citation to view",
            st.session_state.citations_used
        )

    st.subheader("Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

        greeting_message = f"Welcome! I am your dedicated digital assistant, specialized to provide assistance with your credit documents.\nMy purpose is to extract and present informative data from your documents.\nPlease don't hesitate to pose your queries whenever you are prepared to proceed."

        st.session_state.messages.append({"role": "assistant", "content": greeting_message})

    if prompt := st.chat_input(""):
        st.session_state.messages.append({"role": "user", "content": f"""{prompt}"""})

        st.session_state.selected_schools, school_identifier_cost = school_identifier(prompt, df['School'].unique().tolist())
        search_term, search_query_cost = search_query_generator(prompt)
        st.session_state.query_text = search_term
        num_results = 3

        st.markdown(f"Searching for **{st.session_state.query_text}**")

        query_vector = openai.Embedding.create(model="text-embedding-ada-002", input=search_term)["data"][0]["embedding"]

        df_all = pd.DataFrame([])
        if type(st.session_state.selected_schools) == list:
            for school in st.session_state.selected_schools:
                df_comp = azure_search(search_client, st.session_state.query_text, query_vector, 1, "school", school, num_results, 0)
                df_all = pd.concat([df_all, df_comp])
        elif type(st.session_state.selected_schools) == type(None):
            df_all = azure_search(search_client, st.session_state.query_text, query_vector, 0, None, None, num_results, 0)

        unique_citations_list = df_all['sourcepage'].unique().tolist()
        st.session_state.unique_citations_list = unique_citations_list

        concatenated_values = df_all.apply(
            lambda row: f"{row['school']} (Citations: {row['sourcepage']}): {row['content']}", axis=1)
        final_string = '\n\n'.join(concatenated_values)

        st.session_state.final_string = final_string

        system_prompt = f"""
            Look at the user's question and check the provided data carefully. Your job is to find differences and similarities in the data from different sources. Keep your answer short, clear, and to the point, focusing on what the user asked.

            User's question: {prompt}

            Only use the facts from the sources listed below to answer. If the sources don’t have enough information, just say you don’t know. Don’t make up answers that aren’t in the sources. If asking the user for more details will help, go ahead and ask.

            Each source is listed with a company name, followed by citations and the related information (Example: Company A (Citations: [citation1, citation2, citation3]): content).

            When you answer:
            Return your response as a table separated by companies.
            Clearly list the citations for any specific facts, numbers, or statements you "actually use in your response" from the sources.
            If the sources don’t have specific information or clear answers to the user's question, don’t list any citations and make it clear that there isn’t enough information.

            """
        st.session_state.system_prompt = system_prompt

        estimated_prompt_tokens = tiktoken_len(st.session_state.final_string) + tiktoken_len(st.session_state.system_prompt)
        estimated_completion_tokens = 1024

        prompt_cost = 0.06 / 1000
        completion_cost = 0.12 / 1000
        gpt_model = 'gpt-4'

        st.session_state.prompt_cost = prompt_cost
        st.session_state.completion_cost = completion_cost
        st.session_state.gpt_model = gpt_model

        estimated_cost = (estimated_prompt_tokens * prompt_cost) + (estimated_completion_tokens * completion_cost)
        st.session_state.estimated_cost = estimated_cost + school_identifier_cost + search_query_cost

        st.session_state.messages.append({"role": "assistant", "content": f"""The estimated cost to run your query is {'{:.3f}'.format(st.session_state.estimated_cost)}."""})

        gpt_output, confirmed_citation, cost_spent = gpt_response(st.session_state.gpt_model,
                                                                  st.session_state.system_prompt,
                                                                  st.session_state.final_string,
                                                                  st.session_state.unique_citations_list,
                                                                  st.session_state.prompt_cost,
                                                                  st.session_state.completion_cost)

        st.session_state.actual_cost = cost_spent
        if len(confirmed_citation) > 0:
            min_page_numbers = {}

            for citation in confirmed_citation:
                try:
                    base_file_name, page_number = citation.rsplit("-", 1)
                    base_file_name = f"{base_file_name}.pdf"
                    page_number = int(page_number.split(".")[0])

                    if base_file_name in min_page_numbers:
                        min_page_numbers[base_file_name] = min(min_page_numbers[base_file_name], page_number)
                    else:
                        min_page_numbers[base_file_name] = page_number
                except:
                    continue

        st.session_state.min_page_numbers = min_page_numbers
        st.session_state.citations_used = list(min_page_numbers.keys())

        st.session_state.cost_spent = cost_spent + school_identifier_cost + search_query_cost

        st.session_state.messages.append({"role": "assistant", "content": f"""Searching for {st.session_state.query_text}\n\n{gpt_output}\n\nActual Cost Spent ($): {'{:.3f}'.format(st.session_state.cost_spent)} ({st.session_state.gpt_model})"""})

    st.session_state.selected_citation = citation_viewer

    start_page = 0
    if st.session_state.selected_citation:
        start_page = st.session_state.min_page_numbers[st.session_state.selected_citation]

    layout_container = st.empty()

    # primaryColor = "#231f1c"
    # backgroundColor = "#e3dfdc"
    # secondaryBackgroundColor = "#f5f8f2"
    # textColor = "#1b0608"


    chat_html = ""
    for message in st.session_state.messages:
        icon = "👤" if message["role"] == "user" else "🤖"
        background_color = "#f7f8fa" if message["role"] == "user" else "#ffffff"

        content = message["content"].replace("$", "\$")
        chat_html += f"<div class='message {message['role']}' style='background-color: {background_color}; margin: 5px; padding: 10px; border-radius: 5px; '>{icon} {content}</div>"

    custom_html = f"""
        <div style="display: flex; flex-direction: row; height: 100vh;">
            <div style="flex: 1; padding: 10px; overflow-y: auto;">
                {chat_html}  <!-- This is where the chat messages will be displayed -->
            </div>
            <div style="flex: 1; padding: 10px;">
                <div style="overflow: auto; position: relative; height: 100%;">
                    <embed src="{blob_storage_endpoint}/{container_name}/{st.session_state.selected_citation}?{sas_token}#page={start_page + 1}"
                            style="border: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
                            type="application/pdf">
                    </embed>
                </div>
            </div>
        </div>
    """

    layout_container.markdown(custom_html, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide")

    conn = pyodbc.connect(f'DSN=MySqlServerDSN;UID={os.environ.get("SQL_USERNAME")};PWD={os.environ.get("SQL_PASSWORD")}')

    table_name = 'School'
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)

    school_review(df)

if __name__ == "__main__":
    main()
