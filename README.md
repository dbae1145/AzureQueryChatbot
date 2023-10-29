# AzureQueryChatbot

Welcome to the **AzureQueryChatbot: Interactive PDF Content and Citation Retrieval Using Azure Services**! This app provides an easy and intuitive interface where users can query information about the contents stored in an Azure Search Index. The application is built to handle queries about various entities and display the relevant information in a tabular format.

![Example Image](https://github.com/dbae1145/images/blob/main/AzureQueryChatBotExample.png)

## Features

### 1. Upload PDF Files
- The application allows users to upload PDF files.
- Upon uploading, the contents of the PDF are pushed to an Azure Search Index for further querying.
- Citations within the PDF are stored separately in Azure Storage Blob.

### 2. Query Contents
- Users can input queries through the chat interface.
- The application processes the query and creates a search query.
- The search query then retrieves relevant documents from the Azure Search Index.
- Two types of searches are performed:
  - Traditional keyword search.
  - Vector search.
- After retrieving the documents, GPT is applied to formulate a user-friendly response.
- The response is then displayed on the front interface of the application.

### 3. Show Citations
- For each query, the application shows relevant PDF files as citations with related pages displayed.
- The citations are retrieved from the Azure Storage Blob.

### 4. Tabular Format for Multiple Entities
- The application has the capability to query about multiple entities (e.g., universities)
- The information about the entities is displayed in a tabular format.

## How to Use

1. Clone the repository to your local machine.
2. Install the required dependencies (`requirements.txt`)
3. Run `app.py`.
4. Access the chat interface through your browser.
5. Start querying and explore the contents from the Azure Search Index.

## Dependencies

- Azure Search Service SDK
- Azure Storage Blob SDK

## Configuration

Please refer to the Azure documentation for setting up the Azure Search Index and Azure Storage Blob. You will need to update the application with your Azure credentials and configuration details.

## Contribution

Your contributions are welcome! Please follow the following steps to contribute to the project:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

Please make sure to update the README.md file with the details of your changes.

