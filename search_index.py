'''
October 5, 2023
Go to https://portal.azure.com/#home, log in with your credentials. The Azure Search Service we're going to use is "srch-openai-poc"

This script dictates the following:
1. extract texts from pdf (digital vs. non-digital doc)
- For digital docs, use open-source python package called fitz
- For non-digital docs, use azure form recognizer
2. #1 will create a list of tuples called "page_map". Each tuple in "page_map" consistst of page_number, start_index, and content (text)
3. Once the page map is created, it concatenates all contents into one giant string.
4. Chunk the giant string of content into pieces. The chunking size is manually determined (I've set 500 tokens to be the chunk size of each chunk)
5. Once chunks are created, for each chunk, the page numbers in the original pdf file are tracked and matched. Then it creates a list of tuples consisting of a text chunk and a list of identified pages such as (chunk (=text), [page1, page2, page3]).
6. Then it takes the list from #5,

'''
#import necessary python packages
import argparse
import base64
import glob
import html
import io
import os
import re
import time
import fitz
import json
import openai
import requests
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.storage.blob import BlobServiceClient
from pypdf import PdfReader, PdfWriter
from tenacity import retry, stop_after_attempt, wait_random_exponential
import pandas as pd
import warnings
import tiktoken
import pyodbc
import urllib
import sqlalchemy
from datetime import datetime, timedelta, date
import sys
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.pdfpage import PDFPage
tokenizer = tiktoken.get_encoding('cl100k_base')
warnings.filterwarnings("ignore")

def get_pdf_searchable_pages(fn):
    searchable_pages = []
    non_searchable_pages = []
    page_num = 0
    with open(fn, 'rb') as infile:
        for page in PDFPage.get_pages(infile):
            page_num += 1
            if 'Font' in page.resources.keys():
                searchable_pages.append(page_num)
            else:
                non_searchable_pages.append(page_num)
    if page_num > 0:
        return searchable_pages, non_searchable_pages

def get_file_extension(fn):
    return os.path.splitext(fn)[1][1:]

def blob_name_from_file_page(filename, page=0):
    if os.path.splitext(filename)[1].lower() == ".pdf":
        return os.path.splitext(os.path.basename(filename))[0] + f'-{page}' +".pdf"
    else:
        return os.path.basename(filename)

def upload_blobs(filename, container_name):
    blob_service = BlobServiceClient(account_url=f"https://stonexsearchdatapoc.blob.core.windows.net",
                                     credential=storage_creds)
    blob_container = blob_service.get_container_client(container_name)
    if not blob_container.exists():
        blob_container.create_container()

    blob_name = os.path.basename(filename)
    with open(filename, "rb") as data:
        blob_container.upload_blob(blob_name, data, overwrite=True)

def filename_to_id(filename):
    filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename)
    filename_hash = base64.b16encode(filename.encode('utf-8')).decode('ascii')
    return f"file-{filename_ascii}-{filename_hash}"

def before_retry_sleep(retry_state):
    print(f"Rate limited on the OpenAI API, sleeping before retrying...")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), before_sleep=before_retry_sleep)
def compute_embedding(text):
    return openai.Embedding.create(engine="text-embedding-ada", input=text)["data"][0]["embedding"]

def create_search_index(index_name):
    print(f"Ensuring search index {index_name} exists")
    index_client = SearchIndexClient(endpoint=f"https://srch-openai-poc.search.windows.net/",
                                     credential=search_creds)
    if index_name not in index_client.list_index_names():
        index = SearchIndex(
            name=index_name,
            fields=[
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String", analyzer_name="en.microsoft", searchable=True, filterable=True, sortable=True, facetable=True),
                SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            hidden=False, searchable=True, filterable=False, sortable=False, facetable=False,
                            vector_search_dimensions=1536, vector_search_configuration="default"),
                SimpleField(name="category", type="Edm.String", analyzer_name="en.microsoft", searchable=True, filterable=True, sortable=True, facetable=True),
                SearchableField(name="sourcepage", type="Edm.String", analyzer_name="en.microsoft",searchable=True, filterable=True, sortable=True, facetable=True),
                # SearchField(name="sourcepage", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True, filterable=False, sortable=False, facetable=False),
                SearchableField(name="sourcefile", type="Edm.String", analyzer_name="en.microsoft",searchable=True, filterable=True, sortable=True, facetable=True),
                SearchableField(name="company", type="Edm.String", analyzer_name="en.microsoft", searchable=True, filterable=True, sortable=True, facetable=True),
                SearchableField(name="sector", type="Edm.String", analyzer_name="en.microsoft", searchable=True,filterable=True, sortable=True, facetable=True),
                SearchableField(name="industry", type="Edm.String", analyzer_name="en.microsoft", searchable=True,filterable=True, sortable=True, facetable=True),
                SearchableField(name="date", type="Edm.String", analyzer_name="en.microsoft", searchable=True,filterable=True, sortable=True, facetable=True)

            ],
            semantic_settings=SemanticSettings(
                configurations=[SemanticConfiguration(
                    name='default',
                    prioritized_fields=PrioritizedFields(
                        title_field=None, prioritized_content_fields=[SemanticField(field_name='content')]))]),
            vector_search=VectorSearch(
                algorithm_configurations=[
                    VectorSearchAlgorithmConfiguration(
                        name="default",
                        kind="hnsw",
                        hnsw_parameters=HnswParameters(metric="cosine")
                    )
                ]
            )
        )
        print(f"Creating {index_name} search index")
        index_client.create_index(index)
    else:
        print(f"Search index {index_name} already exists")

def index_sections(filename, sections, index_name):
    print(f"Indexing sections from '{filename}' into search index {index_name}")
    search_client = SearchClient(endpoint=f"https://srch-openai-poc.search.windows.net/",
                                 index_name=index_name,
                                 credential=search_creds)
    i = 0
    batch = []
    for s in sections:
        batch.append(s)
        i += 1
        if i % 200 == 0:
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            print(f"Suceeded: {succeeded}")
            print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            batch = []

    if len(batch) > 0:
        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        print(f"Suceeded: {succeeded}")
        print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")

def get_file_extension(fn):
    return os.path.splitext(fn)[1][1:]

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=15,
    length_function=tiktoken_len,
    separators=['\n','\n\n']
)

def get_ticker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code

def get_sector_and_industry(company_name):
    # Attempt to extract the first ticker symbol from the search results
    ticker_symbol = get_ticker(company_name)
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    sector = info.get('sector', None)
    industry = info.get('industry', None)
    return sector, industry

def bing_search(query):
    mkt = 'en-US'
    params = {'q': query, 'mkt': mkt}
    headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

    try:
        response = requests.get(bing_search_endpoint,headers=headers, params=params)
        response.raise_for_status()
        json = response.json()
        return json["webPages"]["value"]

    except Exception as ex:
        raise ex

def extract_name_date(prompt):
    """Extract the company name, the most recent date, and the sector from the first section of a credit agreement using OpenAI Chat API."""

    func_desc_name_date = [
        {
            "name": "company_name_doc_date",
            "description": "Call multiple functions in one call",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Function to extract a company name that is a credit facilities from the first section of a loan covenant review.",

                    },
                    "most_recent_date": {
                        "type": "string",
                        "description": "Function to extract the research date from the first section of a loan covenant review.",

                    },
                },
                "required": ["company_name", "most_recent_date"],
            }
        }
    ]

    completion_name_date = openai.ChatCompletion.create(
        engine='gpt-35-turbo',
        messages=[
            {'role': 'system', 'content': "Identify the company name that is a borrower and the most recent date stated in the following credit agreement"},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.1,
        max_tokens=100,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        functions=func_desc_name_date,
        function_call="auto"
    )

    results = {}
    try:
        name = json.loads(completion_name_date.choices[0].message.function_call.arguments).get("company_name")
        results['name'] = name
    except AttributeError:
        results['name'] = None

    print(f"Name: {name}")

    try:
        date_str = json.loads(completion_name_date.choices[0].message.function_call.arguments).get("most_recent_date")
        results['date'] = date_str
    except (AttributeError, ValueError):
        results['date'] = None

    print(f"Date: {date_str}")

    return results

def extract_sector_industry(name):
    question = f"Which sector and industry does {name} belong to?"
    bing_results = bing_search(question)
    bing_results = [f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in bing_results]
    bing_prompt = "\n\n".join(bing_results) + "\n\nQuestion: " + question

    func_desc_industry = [
        {
            "name": "industry",
            "description": "identify the industry of a given company",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Function to identify the specific sector that the identified company belongs to.",
                        "enum": sectors
                    },
                    "industry": {
                        "type": "string",
                        "description": "Function to identify the specific industry that the identified company belongs to.",
                        "enum": industries
                    },

                },
                "required": ["sector","industry"],
            }
        }
    ]

    completion_industry = openai.ChatCompletion.create(
        engine='gpt-35-turbo',
        messages=[
            {'role': 'system', 'content': "Use the following sources to identify the sector and the industry of a given company"},
            {'role': 'user', 'content': bing_prompt}
        ],
        temperature=0.1,
        max_tokens=100,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        functions=func_desc_industry,
        function_call="auto"
    )

    industry = json.loads(completion_industry.choices[0].message.function_call.arguments).get("industry")
    sector = json.loads(completion_industry.choices[0].message.function_call.arguments).get("sector")

    return sector, industry

def locate_page_numbers(chunks, page_map):
    chunk_page_numbers = []
    for chunk in chunks:
        start_offset = texts.find(chunk)
        end_offset = start_offset + len(chunk)
        associated_pages = set()

        for page, offset, content in page_map:
            content_end_offset = offset + len(content)
            if start_offset < content_end_offset and end_offset > offset:
                associated_pages.add(page)

        chunk_page_numbers.append((chunk, sorted(list(associated_pages))))

    return chunk_page_numbers

def create_sections(filename, chunk_page_numbers, use_vectors):
    file_id = filename_to_id(filename)
    chunk0 = chunk_page_numbers[0][0]

    output = extract_name_date(chunk0)
    name = output['name']
    date = output['date']

    # if name:
    #     try:
    #         sector, industry = get_sector_and_industry(name)
    #     except:
    #         sector, industry = extract_sector_industry(name)

    if name:
        try:
            sector, industry = None, None
        except:
            sector, industry = None, None


    # Create a DataFrame with a single row containing the name, date, industry, and filename
    df = pd.DataFrame({
        "name": [name],
        "date": [date],
        "sector": [sector],
        "industry": [industry],
        "filename": [filename]
    })

    sections = []
    for i, (content, pagenum) in enumerate(chunk_page_numbers):
        print(i)
        pagenum_blob_name = []
        for page in pagenum:
            pagenum_blob_name.append(blob_name_from_file_page(filename, page))

        #remove unnecessary fields
        section = {
            "id": f"{file_id}-chunk-{i}",
            "content": content,
            "category": None,
            "sourcepage": pagenum_blob_name[0] if len(pagenum_blob_name)>0 else '',
            "sourcefile": filename,
            "company": name,
            "sector": sector,
            "industry": industry,
            "date": date
        }

        if use_vectors:
            section["embedding"] = compute_embedding(content)

        sections.append(section)

    # Return the list of sections and the DataFrame
    return sections, df

def get_document_text(filename, run_form_recognizer):
    offset = 0
    page_map = []

    if run_form_recognizer:
        form_recognizer_client = DocumentAnalysisClient(endpoint=f"https://cog-fr-openai-poc.cognitiveservices.azure.com/", credential=formrecognizer_creds, headers={"x-ms-useragent": "azure-search-chat-demo/1.0.0"})
        with open(filename, "rb") as f:
            poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", document=f)
        form_recognizer_results = poller.result()

        for page_num, page in enumerate(form_recognizer_results.pages):
            tables_on_page = [table for table in form_recognizer_results.tables if
                              table.bounding_regions[0].page_number == page_num + 1]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1] * page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >= 0 and idx < page_length:
                            table_chars[idx] = table_id

            # build page text by replacing charcters in table spans with table html
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += form_recognizer_results.content[page_offset + idx]
                elif not table_id in added_tables:
                    page_text += table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)

            page_text += " "
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)

    else:
        doc = fitz.open(filename)
        for i in range(len(doc)):
            page = doc.load_page(i)  # zero-based index
            text = page.get_text("text")
            page_map.append((i, offset, text))
            offset += len(text)

    return page_map

azd_credential = AzureDeveloperCliCredential(tenant_id="73e0ca39-fa24-487a-8ab7-7a070d5c9f84", process_timeout=60)
default_creds = azd_credential
search_creds = default_creds
storage_creds = default_creds
use_vectors = 1

openai.api_type = "azure"
openai.api_base = "https://oai-data-poc2.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "72552b98f5b446d2af344c00bb009247"

bing_search_api_key = "6fbab84f76b442d7a6bad4326fd3dacf"
bing_search_endpoint = "https://api.bing.microsoft.com/" + "v7.0/search"

sectors = [
    "Technology",
    "Healthcare",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Industrials",
    "Materials",
    "Telecommunication Services",
    "Utilities",
    "Real Estate",
    "Communication Services",
    "Cryptocurrency",
    "Renewable Energy",
    "Agriculture",
    "Transportation",
    "Aerospace & Defense",
    "Hospitality",
    "Construction",
    "Specialty Retail",
    "E-Commerce",
    "Biotechnology",
    "Pharmaceuticals",
    "Mining",
    "Chemicals",
    "Automotive",
    "Textiles, Apparel & Luxury Goods",
    "Leisure Products",
    "Professional Services",
    "Environmental Services",
    "Food Products",
    "Beverages",
    "Tobacco",
    "Household Products",
    "Personal Products",
    "Diversified Consumer Services",
    "Commercial Services & Supplies",
    "Office Electronics"
]

industries = [
    "Software & Services",
    "Technology Hardware & Equipment",
    "Semiconductors & Semiconductor Equipment",
    "Pharmaceuticals, Biotechnology & Life Sciences",
    "Health Care Equipment & Services",
    "Automobiles & Components",
    "Consumer Durables & Apparel",
    "Consumer Services",
    "Retailing",
    "Food & Staples Retailing",
    "Food, Beverage & Tobacco",
    "Household & Personal Products",
    "Oil, Gas & Consumable Fuels",
    "Energy Equipment & Services",
    "Banks",
    "Diversified Financials",
    "Insurance",
    "Real Estate",
    "Capital Goods",
    "Commercial & Professional Services",
    "Transportation",
    "Chemicals",
    "Metals & Mining",
    "Paper & Forest Products",
    "Diversified Telecommunication Services",
    "Wireless Telecommunication Services",
    "Electric Utilities",
    "Gas Utilities",
    "Water Utilities",
    "Multi-Utilities",
    "Real Estate Investment Trusts (REITs)",
    "Real Estate Management & Development",
    "Media & Entertainment",
    "Interactive Media & Services",
    "Telecommunication Services",
    "Bitcoin",
    "Altcoins",
    "Blockchain Technology",
    "Solar",
    "Wind",
    "Hydroelectric",
    "Biomass & Biofuels",
    "Crop Production",
    "Livestock",
    "Agricultural Products & Services",
    "Airlines",
    "Railroads",
    "Trucking",
    "Shipping",
    "Aerospace",
    "Defense",
    "Hotels, Resorts & Cruise Lines",
    "Restaurants & Leisure",
    "Building Products",
    "Construction & Engineering",
    "Homebuilding",
    "Apparel Retail",
    "Computer & Electronics Retail",
    "Home Improvement Retail",
    "Online Retail",
    "Internet & Direct Marketing Retail",
    "Biotechnology",
    "Pharmaceuticals",
    "Gold",
    "Precious Metals & Minerals",
    "Diversified Metals & Mining",
    "Specialty Chemicals",
    "Diversified Chemicals",
    "Commodity Chemicals",
    "Auto Components",
    "Automobile Manufacturers",
    "Textiles",
    "Apparel, Accessories & Luxury Goods",
    "Leisure Products",
    "Consulting Services",
    "Employment Services",
    "Waste Management",
    "Environmental Services",
    "Packaged Foods & Meats",
    "Agricultural Products",
    "Soft Drinks",
    "Brewers",
    "Distillers & Vintners",
    "Tobacco",
    "Household Products",
    "Personal Products",
    "Education Services",
    "Diversified Consumer Services",
    "Commercial Services & Supplies",
    "Office Electronics"
]

#Create a cost estimate calculator for form recognizer (= # of pages * $0.01) print cost after analyzing each document. For a mix of searchable and non-searchable pages, treat the entire file as non-searchable and sum up the pages all together and then multiply $0.01
create_search_index('covenant')

directory = r'C:\Users\sbae\PycharmProjects\Projects\Credit_Loan_Covenant_Review'
valid_extensions = set(['pdf'])
filenames = []
for dirname, _, files in os.walk(directory):
    for filename in files:
        if not filename.startswith("~$"):
            if get_file_extension(filename) in valid_extensions:
                filenames.append(os.path.join(dirname, filename))

file_dict = {
    'Filename': [],
    'Searchable Pages': [],
    'Non-Searchable Pages':[]
             }

for file in filenames:
    searchable_pages, non_searchable_pages = get_pdf_searchable_pages(file)
    file_dict['Filename'].append(file)
    file_dict['Searchable Pages'].append(searchable_pages)
    file_dict['Non-Searchable Pages'].append(non_searchable_pages)

file_df = pd.DataFrame(file_dict)
file_df['Run Form Recognizer'] = file_df['Non-Searchable Pages'].apply(lambda x: 1 if len(x)>0 else 0)

df_push2sql = pd.DataFrame([])

for index, row in file_df.iterrows():

    file = row['Filename']
    run_form_recognizer = row['Run Form Recognizer']

    upload_blobs(file, 'covenant')

    page_map = get_document_text(file, run_form_recognizer)

    texts = ''
    for i, (page, offset, content) in enumerate(page_map):
        texts += content + '\n\n'

    chunks = text_splitter.split_text(texts)

    chunk_page_numbers = locate_page_numbers(chunks, page_map)

    file_base_name = os.path.basename(file)
    use_vectors = 1
    sections, df = create_sections(file_base_name, chunk_page_numbers, use_vectors)

    df_push2sql = pd.concat([df_push2sql, df])

    index_sections(file_base_name, sections, 'covenant')

df_push2sql["sector"] = df_push2sql["sector"].apply(lambda x: None)
df_push2sql["industry"] = df_push2sql["industry"].apply(lambda x: None)

driver = '{ODBC Driver 17 for SQL Server}'
server = 'sql-data-dev-canadacentral-001.database.windows.net'
port = '1433'
database = 'TextAnalytics'
username = 'sbae@onex.com'
authentication = 'ActiveDirectoryPassword'
pwd = "Fogonthewater21*"

conn = pyodbc.connect('DRIVER=' + driver +
                      ';SERVER=' + server +
                      ';PORT=' + port +
                      ';DATABASE=' + database +
                      ';UID=' + username +
                      ';AUTHENTICATION=' + authentication +
                      ';PWD=' + pwd
                      )

table_name = 'Covenant'

tuples_list = list(df_push2sql.itertuples(index=False, name=None))

query = f'''INSERT INTO {table_name} (Company, Date, Sector, Industry, Filename) VALUES (?, ?, ?, ?, ?)'''

cursor = conn.cursor()
cursor.executemany(query, tuples_list)
conn.commit()
conn.close()


from pdf_processor import PDFProcessor
import os
azure_tenant_id = "c3ee72f8-ed8b-42b9-85d5-8f44c470e8be"
azure_storage_acct_url = "https://sbaepoc.blob.core.windows.net/?sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2023-10-29T07:03:23Z&st=2023-10-28T23:03:23Z&spr=https&sig=zl1qUIHtqO%2FVPYCR4SIUUXrWv4xpsBtWZdb6U4nIw14%3D"

storage_container_name = "school"
azure_search_endpoint = "https://cog-srch-poc.search.windows.net"
search_index_key = "3hrp5ixLJauQhce3iqlAlgltgM2Qb1K93gB6kF04yNAzSeDgVM6u"
search_index = "school"
openai_key = "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"

path = "/home/pc/doc_qa/sample"

def get_file_extension(filename):
    return os.path.splitext(filename)[1][1:]

valid_extensions = set(['pdf'])
filenames = []
for dirname, _, files in os.walk(path):
    for filename in files:
        if not filename.startswith("~$"):
            if get_file_extension(filename) in valid_extensions:
                filenames.append(os.path.join(dirname, filename))

for file in filenames:
    pdf_processor = PDFProcessor(0,
                                 file,
                                 azure_storage_acct_url,
                                 storage_container_name,
                                 azure_search_endpoint,
                                 search_index_key,
                                 search_index,
                                 openai_key,
                                 "School")
    pdf_processor.run()

