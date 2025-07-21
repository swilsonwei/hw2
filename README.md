📅 Understanding RAG
    Loading data into RAG --- file to submit is load_data.py
    With Python, create a script that loads data into a sparse vector index (Milvus) from GitHub
    This should experiment with chunking in different ways
        By file, by function, etc

Reranking data with FastAPI --- file to submit is main.py
    With FastAPI, create a server that does the following:
    Queries data in a sparse vector index (Milvus)
    Queries data in a dense vector index (Milvus)
    Combines the results of the two queries and reranks the results with OpenAI
    The re-ranked results should be only the top 5 results

Create some integration tests for your RAG --- file to submit is test_rag.py
    With Python, create a script that creates integration tests for your RAG
    You should have at least five tests that guarantee results are returned
    You should have at least five tests that prevent abuse of your RAG system

Submission
    Zip up to the three files

    load_data.py
    test_rag.py
    main.py
And then upload to DataExpert.io