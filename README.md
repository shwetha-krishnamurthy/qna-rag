# QnA-RAG

QnA-RAG (Question and Answering using Retrieval-Augmented Generation) is a project that combines retrieval-based and generative models to provide accurate answers to user queries. It uses a retrieval-augmented approach to extract relevant information from a given set of documents and then generates precise responses based on the retrieved context.

## Features

- **Hybrid Approach**: Utilizes both retrieval and generation to produce context-aware answers.
- **Custom Document Set**: Allows you to upload your own documents to create a knowledge base for answering questions.
- **Scalable Backend**: Built with modular components that can be extended or modified for custom use-cases.

## Architecture

The system is composed of the following key components:

1. **Document Retriever**: Extracts relevant content from a corpus of documents based on the user query.
2. **Generative Model**: Uses the retrieved content to generate a well-informed answer.
3. **User Interface**: Provides a simple API or UI for interacting with the QnA system.

## Getting Started

### Prerequisites

- Python 3.8 or later
- pip
- GPU (Optional, but recommended for faster performance)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shwetha-krishnamurthy/qna-rag.git
   cd qna-rag
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Start the Application**:
   ```bash
   python app.py
   ```
   This command will launch the application and expose an endpoint or UI for interacting with the QnA system.

2. **Ask Questions**:
   - Use the API endpoint `/ask` to send queries along with a document set.
   - Alternatively, interact via the provided UI.

### Example

Here is an example of how to use the API:

```python
import requests

url = "http://localhost:8000/ask"
query = {
    "question": "What is Retrieval-Augmented Generation?",
    "documents": ["doc1.txt", "doc2.txt"]
}
response = requests.post(url, json=query)
print(response.json())
```

## Models Used

- **Retriever**: Typically a dense retriever like DPR (Dense Passage Retriever) or BM25 is used to fetch relevant passages.
- **Generator**: A transformer-based language model such as GPT-3 or T5 is employed to generate the answer.

## Contributing

Contributions are welcome! Please open an issue first to discuss what you would like to change. You can also fork the repository, make your changes, and submit a pull request.

## Contact

For questions or support, please reach out to [Shwetha Krishnamurthy](https://github.com/shwetha-krishnamurthy).


