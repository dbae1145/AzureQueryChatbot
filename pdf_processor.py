import base64
import os
import re
import fitz
import json
import openai
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from tenacity import retry, stop_after_attempt, wait_random_exponential
import pandas as pd
import warnings
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pyodbc
tokenizer = tiktoken.get_encoding('cl100k_base')
warnings.filterwarnings("ignore")

class PDFProcessor:
    def __init__(self,
                 user_file_uploader,
                 file,
                 azure_storage_acct_url,
                 storage_container_name,
                 azure_search_endpoint,
                 search_index_key,
                 search_index,
                 openai_key,
                 sql_table):

        self.use_file_uploader = user_file_uploader
        self.file = file
        self.azure_storage_acct_url = azure_storage_acct_url
        self.storage_container_name = storage_container_name
        self.azure_search_endpoint = azure_search_endpoint
        self.search_index_key = search_index_key
        self.search_index = search_index
        self.openai_key = openai_key
        self.sql_table = sql_table

    def blob_name_from_file_page(self, page):
        file_name, file_extension = os.path.splitext(self.file.name if self.use_file_uploader else os.path.basename(self.file))
        return f"{file_name}-{page}.pdf" if file_extension.lower() == ".pdf" else self.file.name



    def upload_blobs(self):
        blob_service = BlobServiceClient(account_url=self.azure_storage_acct_url)
        blob_container = blob_service.get_container_client(self.storage_container_name)

        if not blob_container.exists():
            blob_container.create_container()

        blob_name = self.file.name if self.use_file_uploader else os.path.basename(self.file)

        with (self.file if self.use_file_uploader else open(self.file, "rb")) as data:
            blob_container.upload_blob(blob_name, data, overwrite=True)


    def filename_to_id(self):
        filename = self.file.name if self.use_file_uploader else self.file
        filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename)
        filename_hash = base64.b16encode(filename.encode('utf-8')).decode('ascii')
        return f"file-{filename_ascii}-{filename_hash}"

    def extract_name(self, prompt):

        func_desc_name_date = [
            {
                "name": "school_name",
                "description": "Call multiple functions in one call",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "school_name": {
                            "type": "string",
                            "description": "Function to extract a school (university) name.",

                        }
                    },
                    "required": ["school_name"],
                }
            }
        ]

        completion_name_date = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'system', 'content': "Identify a school (university) name."},
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
            name = json.loads(completion_name_date.choices[0].message.function_call.arguments).get("school_name")
            results['name'] = name
        except AttributeError:
            results['name'] = 'not found'

        return results


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
    def compute_embedding(self, text):
        return openai.Embedding.create(model="text-embedding-ada-002", input=text)["data"][0]["embedding"]

    def create_search_index(self):
        print(f"Ensuring search index {self.search_index} exists")
        search_creds = AzureKeyCredential(self.search_index_key)
        index_client = SearchIndexClient(endpoint=self.azure_search_endpoint,credential=search_creds)

        if self.search_index not in index_client.list_index_names():
            index = SearchIndex(
                name=self.search_index,
                fields=[
                    SimpleField(name="id", type="Edm.String", key=True),
                    SearchableField(name="content", type="Edm.String", analyzer_name="en.microsoft", searchable=True, filterable=True, sortable=True, facetable=True),
                    SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                                hidden=False, searchable=True, filterable=False, sortable=False, facetable=False,
                                vector_search_dimensions=1536, vector_search_configuration="default"),
                    SimpleField(name="category", type="Edm.String", analyzer_name="en.microsoft", searchable=True, filterable=True, sortable=True, facetable=True),
                    SearchableField(name="sourcepage", type="Edm.String", analyzer_name="en.microsoft",searchable=True, filterable=True, sortable=True, facetable=True),
                    SearchableField(name="sourcefile", type="Edm.String", analyzer_name="en.microsoft",searchable=True, filterable=True, sortable=True, facetable=True),
                    SearchableField(name="school", type="Edm.String", analyzer_name="en.microsoft", searchable=True, filterable=True, sortable=True, facetable=True)

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
            print(f"Creating {self.search_index} search index")
            index_client.create_index(index)
        else:
            print(f"Search index {self.search_index} already exists")

    def index_sections(self, sections):
        file_name = self.file.name if self.use_file_uploader else self.file
        print(f"Indexing sections from '{file_name}' into search index {self.search_index}")
        search_creds = AzureKeyCredential(self.search_index_key)
        search_client = SearchClient(endpoint=self.azure_search_endpoint,
                                     index_name=self.search_index,
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

    def get_file_extension(self):
        return self.file.name.split('.')[-1] if self.use_file_uploader else s.path.splitext(self.file)[1][1:]

    def tiktoken_len(self, text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def locate_page_numbers(self, texts, chunks, page_map):
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

    def create_sections(self, file_base_name, chunk_page_numbers, use_vectors):
        file_id = self.filename_to_id()
        chunk0 = chunk_page_numbers[0][0]

        output = self.extract_name(chunk0)
        name = output['name']
        print(name)

        df = pd.DataFrame({
            "name": [name],
            "filename": [file_base_name]
        })

        sections = []
        for i, (content, pagenum) in enumerate(chunk_page_numbers):
            pagenum_blob_name = []
            for page in pagenum:
                pagenum_blob_name.append(self.blob_name_from_file_page(page))

            section = {
                "id": f"{file_id}-chunk-{i}",
                "content": content,
                "category": None,
                "sourcepage": pagenum_blob_name[0] if len(pagenum_blob_name)>0 else '',
                "sourcefile": file_base_name,
                "school": name
            }

            if use_vectors:
                section["embedding"] = self.compute_embedding(content)

            sections.append(section)

        return sections, df

    def get_document_text(self):
        offset = 0
        page_map = []

        if self.use_file_uploader:
            self.file = open(self.file.name, 'rb')
            pdf_bytes = self.file.read()
            pdf = fitz.open("pdf", pdf_bytes)
        else:
            pdf_bytes = self.file
            pdf = fitz.open(pdf_bytes)

        for i in range(len(pdf)):
            page = pdf.load_page(i)  # zero-based index
            text = page.get_text("text")
            page_map.append((i, offset, text))
            offset += len(text)

        return page_map

    def run(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=15,
            length_function=self.tiktoken_len,
            separators=['\n','\n\n']
        )

        openai.api_key = self.openai_key

        self.upload_blobs()

        self.create_search_index()

        page_map = self.get_document_text()

        texts = ''
        for i, (page, offset, content) in enumerate(page_map):
            texts += content + '\n\n'

        chunks = text_splitter.split_text(texts)

        chunk_page_numbers = self.locate_page_numbers(texts, chunks, page_map)

        file_base_name = self.file.name if self.use_file_uploader else os.path.basename(self.file)
        use_vectors = 1
        sections, df = self.create_sections(file_base_name, chunk_page_numbers, use_vectors)

        self.index_sections(sections)

        conn = pyodbc.connect(f'DSN=MySqlServerDSN;UID={os.environ.get("SQL_USERNAME")};PWD={os.environ.get("SQL_PASSWORD")}')

        table_name = self.sql_table

        tuples_list = list(df.itertuples(index=False, name=None))

        query = f'''INSERT INTO {table_name} (School, Filename) VALUES (?, ?)'''

        cursor = conn.cursor()
        cursor.executemany(query, tuples_list)
        conn.commit()
        conn.close()




