# FinFeed RAG ChatBot
This project aims to develop an AI assistant that efficiently aggregates current news related to finance, economy, and politics from YouTube news channels within a specified timeframe. The assistant allows users to inquire about recent economic news and receive responses through a Retrieval-Augmented Generation (RAG) system. The system simultaneously presents a dynamic  sentiment graph for each context relevant to the query and examines public opinions derived from YouTube comments.
# Authors: 
Aryama Singh (as3844@cornell.edu), Diliya Yalikun, Korel Gundem, Nazanin Komeilizadeh, Roberto Nunez.
# Modeling Approach
![Example Image](dataflow2.png)
In our modelling approach, chaining and context transmission to the LLM model is crucial for generating precise responses. After preprocessing and vectorizing text chunks, embeddings are stored in a Pinecone vector database. Upon receiving a user query, we use cosine similarity to identify and rank the most relevant text chunks efficiently.
These top-matching chunks are then chained together to form a cohesive context. LangChain, a framework for building applications with language models, facilitates this process by seamlessly integrating different components and ensuring efficient data flow.
The curated context is then sent to our LLM model, GPT-3.5 Turbo, chosen for its advanced natural language capabilities. By providing the model with rich and relevant input, we ensure accurate and contextually appropriate responses. LangChain and the vector database work together to maintain a dynamic and responsive system, meeting high standards of information accuracy and relevance.

## Sentiment Analysis on the comments
We wanted to analyze people’s reaction to the current news. We achieved this by passing the comments as context to our LLM model (chat gpt 3.5 turbo) and prompted it to return the sentiments of the people based on the following schema:

General Sentiment: It gives the general sentiment of the comment either Positive, Neutral or Negative.

Aggressiveness Score:  It defines the tone of the language used between 0 to 5, ranging from least aggressive to most aggressive.

General Political Tendency: It defines a general political leaning of the comment and justifies the reasoning.

## Sentiment Analysis on the context 
We evaluated pre-trained LLM models for financial sentiment analysis, including FinancialBERT, ProsusAI/finbert, and GPT-3.5 Turbo, using a pre-labeled Kaggle dataset. Accuracy scores were used to rank their performance.
This helped us identify the best model for sentiment analysis of our text chunks. The attached table shows the accuracy scores of FinancialBERT, ProsusAI/finbert, and GPT-3.5 Turbo on the Kaggle dataset
| Model    | FinancialBERT | ProsusAI/finbert | GPT-3.5 |
|----------|---------------|------------------|---------|
| Accuracy |      79%      |        89%       |    71%  |




## FinFeed conda environment

If you want the most streamlined expereince possible this semester, you should set up a finfeed conda environment and run all of the notebooks with this environment.

Check to make sure you have conda by running the following in your command line interface:

    conda --version

If you don't have conda, google how to install it!

Once you have conda run:

    conda env create --name finfeed_env --file=finfeed_env.yml

Press [y] to all of the prompts.  You will be downloading a lot of packages.

Once this is done:

    conda activate finfeed_env

To check everything is there:

    conda list

Should show all of the packages!