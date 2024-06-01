import os
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker ##this is new
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone,  PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field
import ast
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
load_dotenv("API_KEYS") #### don't forget to change!!!
import matplotlib
matplotlib.use('agg')

class Classification(BaseModel):
    answer: str = Field(description="""Do Answer the question based on the context below but pretend like you are a news reporter who just received the context as the latest news. 
                            If you can't answer the question, reply 'I do not have enough information to answer this question' """)
    sentiment: str = Field(description="The sentiment of the comments. Rate each comment.")
    aggressiveness: str = Field(
        description="How aggressive the comments is on a scale from 1 to 10. 1 being polite and 10 means aggressive."
    )
    language: str = Field(description="The language the text is written in")
    political_tendency : str = Field(description= """The political leaning of the comment. Emphasize political leaning on both economic andsocial policies. Justify your answer.""")


class FinFeedRAG:
    def __init__(self, pine_cone_api_key, openai_api_key, pinecone_index, embeddings_model= OpenAIEmbeddings(),model='gpt-3.5-turbo'):
        self.openai_api_key=openai_api_key
        self.api_key_pinecone = pine_cone_api_key
        self.pinecone_index = pinecone_index
        # Initialize Pinecone connection
        self.vector_db = None
        self.embeddings=embeddings_model
        self.model=model
        self.template_classification = """
                            Answer the question based on the context below but pretend like you are a news reporter who just received the context as the latest news. 
                            If you can't answer the question, reply "I do not have enough information to answer this question".
                
                            Extract the properties mentioned in the 'Classification' function and also justify your analysis.

                            Context: {context}

                            Public Comments: {comments}
                            
                            Question: {question}
                            """
        self.template = """
                        Answer the question based on the context provided below, which is structured in a dictionary format. Assume the role of a news reporter. Each time you use information from the context, you must cite it explicitly. Cite the source accompanying each context entry by including it directly in your response. Additionally, for each context, comments from public is provided. At the end of the answer, please provide some funny examples from people's opinion.
                        
                        Use at least 5 context to provide a comprehensive and verbose answer. If you lack sufficient information to formulate a response, please state: "I do not have enough information to answer this question."
                        
                        Contexts:
                        {context}
                        
                        Question:
                        {question}
                        
                        Citing the context:
                        When referencing a specific context in your answer, use the format:
                        'According to [source], ...'. For example, if drawing from the first context, you would write:
                        'According to Yahoo Finance, ...'.
                        
                        Providing people's opinion:
                        At the end of your answer, include people's opinion using the format (this should be at the end, after each context mentioned):
                        "People's Comments on the video:\n
                        [people's comments in quotes]."
                        
                        """





        self.template_prompt_engineer = """
Transform the following user query into a concise and optimized prompt suitable for retrieving relevant chunks from vector data base which consists of news on finance, economics, and politics. Ensure the rephrased prompt clearly reflects key terms and concepts from these fields to improve accuracy in data querying.
Original Query: '{question}'
"""



#####################################################
    def extract_comments(self, query):
        retriever = PineconeVectorStore.from_existing_index(
            index_name=self.pinecone_index,
            embedding=self.embeddings
        ).as_retriever(
            search_type='similarity',
            search_kwargs={'k': 10}
        )

        documents = retriever.invoke(query)
        comments = []
        for doc in documents:
            comments.extend(doc.metadata.get('youtube_comments', []))
        comments_str = "\n".join(comments)
        return comments_str
    def prompt_tagging(self,template=None):
        if template is None:
            template = self.template_classification
        return ChatPromptTemplate.from_template(template)

    def tagging_chain(self, query):
        retriever = PineconeVectorStore.from_existing_index(
            index_name=self.pinecone_index,
            embedding=self.embeddings
        ).as_retriever(
            search_type='similarity',
            search_kwargs={'k': 10}
        )

        comments = self.extract_comments(query)
        comments_runnable = RunnableLambda(lambda _: comments)

        chaining = (
            {
                "context": retriever,
                "comments": comments_runnable,
                "question": RunnablePassthrough()
            }
            | self.prompt_tagging() #uses self.template_classification
            | self.llm_tagging() #for structured output
        )

        return chaining.invoke(query).dict()

    
    def get_all_vector_ids(self,  input_vector, top_k = 10000, index= None):
        if index is None:
            index = self.pinecone_index
        results = index.query(vector=input_vector, top_k= top_k, include_values=False)
        # ids = set()
        ids=[]
        for result in results['matches']:
            # ids.add(result['id'])
            ids.append(result['id'])
        return ids


    def insert_youtube_metadata(self, top_k=10000, index= None):
        """Set top_k >= total_vector_count of vector database
        """
        if index is None:
            index= self.pinecone_index
        assert top_k > self.vector_db.describe_index_stats().total_vector_count, "top_k is less that total_vector_count of vector database"

        my_index = Pinecone(self.api_key_pinecone).Index(index)
        all_ids = self.get_all_vector_ids([0.0] * 1536, top_k= top_k , index=my_index)  

        for id in all_ids:
        
            old_metadata = index.fetch([id])['vectors'][id]['metadata']
            index.update(
            id= id, 
            set_metadata={'youtube_response_metadata': str(self.df_youtube_data[self.df_youtube_data['txt_file_path'].apply(lambda x : x.replace('\\','/')) == 
                                                                old_metadata['source']]['response_metadata'].values[0]),
                        'youtube_comments': self.df_youtube_data[self.df_youtube_data['txt_file_path'].apply(lambda x : x.replace('\\','/')) 
                                                                 == old_metadata['source']]['list_of_comments'].values[0]
        }, 
            namespace=""
        )
  

    def llm_tagging(self, model = 'gpt-3.5-turbo'):
       
        llm = ChatOpenAI(openai_api_key=self.openai_api_key, model=model).with_structured_output(Classification)
        return llm

    
    def initialize_pinecone(self):
        if self.vector_db is None:  # Check if it's already initialized
            pc = Pinecone(api_key=self.api_key_pinecone)
            self.vector_db = pc.Index(self.pinecone_index)  # Connect to the index and store the connection
        return self.vector_db
        
    
    def preprocess_youtube_text(self, text_file, chunksize,chunkoverlap, preprocess_yt=True):
        if preprocess_yt:
            self.preprocess_input(text_file,save_back_to_file=True)
        else:
            None
        loader = TextLoader(text_file) #text instance of langchain
        text_documents = loader.load() 
        # Assuming RecursiveCharacterTextSplitter is a class you have access to or have created
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
        processed_text = splitter.split_documents(text_documents)
        # Further processing can be done here if necessary
        return processed_text

    def upload_to_vb(self,text,embeddings,chunksize, chunkoverlap,index=None,preprocess_yt=True):
        if index is None:
            index = self.pinecone_index
        return PineconeVectorStore.from_documents(self.preprocess_youtube_text(text,chunksize,chunkoverlap,preprocess_yt), self.embeddings, index_name=index)


    def preprocess_input(self, text_file,save_back_to_file=True):
        # Simple text preprocessing: lowercasing, removing punctuation need to add more preprocessing steps do research on it
        # Read and process the content and rewrite it
        if save_back_to_file==True:
            with open(text_file, 'r') as file:
                # Read the contents of the file
                text = file.read()
            processed_text = text.lower()
            processed_text = re.sub(r'[^\w\s]', '', processed_text)
            tokens = word_tokenize(processed_text)
            filtered_words = [word for word in tokens if word.lower() not in stopwords.words('english')]
            # Join words back into a single string
            final_text = ' '.join(filtered_words)
            # Write the processed content back, replacing the original
            with open(text_file, 'w') as file:
                file.write(final_text)
        else:
            with open(text_file, 'r') as file:
                # Read the contents of the file
                text = file.read()
            processed_text = text.lower()
            processed_text = re.sub(r'[^\w\s]', '', processed_text)
            tokens = word_tokenize(processed_text)
            filtered_words = [word for word in tokens if word.lower() not in stopwords.words('english')]
            # Join words back into a single string
            final_text = ' '.join(filtered_words)
            return final_text
        
    def most_common(self, input_text_file,most_common=10):
        # Preprocess the text
        processed_text = self.preprocess_input(input_text_file,save_back_to_file=False)    
        # Extract keywords based on frequency, assuming more frequent terms are more relevant
        words = processed_text.split()
        word_freq = Counter(words)
        common_words = word_freq.most_common(most_common)  # Get the top 5 words       
        # Form a query by joining the most common words
        query = ' '.join(word for word, _ in common_words)
        return query

    def retrieve_embeddings(self, query, most_similar=2):
        assert self.vector_db is not_none, "Initialize Pinecone first"
        query_result = self.vector_db.query(vector=self.embeddings.embed_query(query), top_k=most_similar)
        ids = [item['id'] for item in query_result['matches']]
        return [self.vector_db.fetch(ids)['vectors'][id]['values'] for id in ids]

    def provide_context(self, query,index=None,most_similar=10):
        if index is None:
            index = self.pinecone_index
        # Provide context to LLM
        return PineconeVectorStore.from_existing_index(index_name=index,embedding=self.embeddings).as_retriever(search_type='similarity',
                search_kwargs={
                'k':most_similar}).invoke(query)
        
    def prompt(self,template=None):
        if template is None:
            template = self.template
        return ChatPromptTemplate.from_template(template)

    def prompt_eng(self,template=None):
        if template is None:
            template = self.template_prompt_engineer
        return ChatPromptTemplate.from_template(template)
        
    def llm(self,model=None):
        if model is None:
            model = self.model
        return ChatOpenAI(openai_api_key=self.openai_api_key, model=model)
        
    def parser(self):
        return StrOutputParser()
    def chain_prompt_eng(self,query):
        chaining_eng =  (
        {
         "question": RunnablePassthrough()}
        | self.prompt_eng()
        | self.llm()
        | self.parser())
        return chaining_eng.invoke(query)

    def chain(self,query):
        #complete_query = self.prompt().format(context=self.provide_context(query),question=query)
        #response = self.llm().invoke(complete_query)
        #return self.parser().invoke(response)
        chaining = (
        {"context": PineconeVectorStore.from_existing_index(index_name=self.pinecone_index,embedding=self.embeddings).as_retriever(search_type='similarity',
                search_kwargs={
                'k': 10}), 
         "question": RunnablePassthrough()}
        | self.prompt()
        | self.llm()
        | self.parser())
        #query=str(self.prompt_eng(query))
        return chaining.invoke(query)
    
    def pipe(self,chunk):
        pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
        return pipe(chunk)

    def get_sentiment(self,chunks,neutrality_threshdold=0.3):
        """Gets the compound sentiment of the chunks based on their individual sentiment
        Parameters
        ----------
        chunks : list
            List of text chunks
        neutrality_threshdold : float, optional
            A hyperparameter neutrality_threshdold tunes how certain we need to be of a sentiment to classify it as positive or negative
            (If neutrality_threshdold=1, any list of chunks will result in a neutral sentiment
            If neutrality_threshdold=0, any list of chunks will be classified as positive or negative)
        Returns
        -------
        int
            1 for positive, 0 for neutral, and -1 for negative
        """
        #Assing a numerical value to each sentiment to simplify calculations
        sentiment_values = {'positive':1, 'neutral':0, 'negative':-1}
        #Run each chunk through sentiment model
        sentiments = [self.pipe(chunk.page_content)[0] for chunk in chunks]
        #Print out model output
        #print(sentiments)
        #For each chunk, we compute a sentiment score by multiplying the score times the sentiment value corresponding to its label
        sentiment_scores = [(sentiment['score'])*sentiment_values[sentiment['label']] for sentiment in sentiments]
        #Average sentiment_scores
        avg_sentiment_score = sum(sentiment_scores)/len(sentiment_scores)
        if avg_sentiment_score >= neutrality_threshdold:
            return ('positive',sentiments)
        elif avg_sentiment_score <= -neutrality_threshdold:
            return ('negative',sentiments)
        else:
            return ('neutral',sentiments)




   
    def chain1(self, query):
        # Initialize the retriever using an existing Pinecone index with specified embeddings
     
        retriever = PineconeVectorStore.from_existing_index(
        index_name=self.pinecone_index,
        embedding=self.embeddings
    ).as_retriever(
        search_type='similarity',
        search_kwargs={'k': 10}  # Retrieve top 10 similar results
    )

    # Invoke the retriever with the query and process metadata
        retrieved_items = retriever.invoke(query)
        metadata = []
        for item in retrieved_items:                # Extract and evaluate the nested metadata string if it exists
            meta_string = item.metadata['youtube_reponse_metadata']
            metadata.append(meta_string) 
        
        # Combine retrieved metadata into a dictionary
        dic = {i: meta for i, meta in enumerate(metadata)}
    
        # Create a chaining operation where metadata is included as context
       
        chaining = (
            {"context": retriever, 
             "metadata": RunnablePassthrough(dic),
             "question": RunnablePassthrough()}
            | self.prompt()
            | self.llm()
            | self.parser()
        )
    
        # Invoke the complete chain with the initial query
        return chaining.invoke(query)

    

    def chain2(self, query):
        # Initialize the retriever using an existing Pinecone index with specified embeddings
        retriever = PineconeVectorStore.from_existing_index(
            index_name=self.pinecone_index,
            embedding=self.embeddings
        ).as_retriever(
            search_type='similarity',
            search_kwargs={'k': 10}  # Retrieve top 10 similar results
        )
      
        # Invoke the retriever with the query and process metadata
        retrieved_items = retriever.invoke(query)
        metadata = []
        for item in retrieved_items:
            content=item.page_content
            meta = ast.literal_eval(item.metadata['youtube_response_metadata'])['snippet']['channelTitle']
            context_entry = {
                "text": content,
                "source": meta
            }
            metadata.append(context_entry)
    
        # Combine retrieved metadata into a dictionary
        dic = {f'Context {i}': context for i, context in enumerate(metadata)}


    
        # Use a lambda function for passing context and question to prompt
        context_and_question = RunnablePassthrough(lambda: {'context': dic ,'question': query})
    
        # Create a sequence of operations
        # Assuming your self.prompt(), self.llm(), and self.parser() are methods that handle their respective parts
        result = {'context': RunnablePassthrough(lambda x : dic) ,'question': RunnablePassthrough()} | self.prompt() | self.llm() | self.parser()
    
        # Invoke the complete chain with the initial query
        return result.invoke(query)


    def chain3(self, query):
        # Initialize the retriever using an existing Pinecone index with specified embeddings
        retriever = PineconeVectorStore.from_existing_index(
            index_name=self.pinecone_index,
            embedding=self.embeddings
        ).as_retriever(
            search_type='similarity',
            search_kwargs={'k': 10}  # Retrieve top 10 similar results
        )
      
        # Invoke the retriever with the query and process metadata
        retrieved_items = retriever.invoke(query)
        metadata = []
        comments = []
        youtube_urls =[]
        contexts=[]
       
        
        for item in retrieved_items:
            yt_url = item.metadata['youtube_video_url']
            content=item.page_content
            comments=item.metadata["youtube_comments"]
            meta = ast.literal_eval(item.metadata['youtube_response_metadata'])['snippet']['channelTitle']
            context_entry = {
                "text": content,
                "source": meta,
                "public_opinion":comments
            }
            metadata.append(context_entry)
            comments.extend(item.metadata.get('youtube_comments', []))
            youtube_urls.append(yt_url)
            contexts.append(content)

        comments_str = "\n".join(comments)
        dic = {f'Context {i}': context for i, context in enumerate(metadata)}
        # Use a lambda function for passing context and question to prompt
        context_and_question = RunnablePassthrough(lambda: {'context': dic ,'question': query})
        
        # Create a sequence of operations
        # Assuming your self.prompt(), self.llm(), and self.parser() are methods that handle their respective parts
        result = {'context': RunnablePassthrough(lambda x : dic) ,'question': RunnablePassthrough()} | self.prompt() | self.llm() | self.parser()
        
        comments_runnable = RunnableLambda(lambda _: comments)
        sentiment_chain = (
            {
                "context": retriever,
                "comments": comments_runnable,
                "question": RunnablePassthrough()
            }
            | self.prompt_tagging() #uses self.template_classification
            | self.llm_tagging() #for structured output
        )

        dic_sentiment = sentiment_chain.invoke(query).dict()

        sentiment = dic_sentiment['sentiment']
        aggresiveness = dic_sentiment['aggressiveness']
        political_tendency=dic_sentiment['political_tendency']
        # Invoke the complete chain with the initial query

        fig,ax=self.analyze_and_plot_sentiment(contexts)
        return result.invoke(query),sentiment,aggresiveness,political_tendency, youtube_urls,ax,fig


    def analyze_and_plot_sentiment(self,contexts):
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        
        list_sent = []
        data_sent = pd.DataFrame({'labels': []})
        
        # Analyze sentiment for each context
        for item in contexts:
            list_sent.append(sentiment_analyzer(item)[0]["label"])
        
        data_sent['labels'] = list_sent
        
        plt.style.use('seaborn-v0_8-dark')
        
        # Count the sentiment labels
        sentiment_counts = data_sent['labels'].value_counts()
        
        # Create a figure and a set of subplots
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Define colors consistent with the seaborn-dark theme
        colors = ['green' if label == 'positive' else 'red' if label == 'negative' else 'blue' for label in sentiment_counts.index]
        
        # Plot the sentiment counts as a bar plot
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        
        # Customize the plot
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Number of Chunks')
        #ax.set_title('Sentiment Analysis of Context Chunks')
        ax.set_xticks(range(len(sentiment_counts)))
        ax.set_xticklabels(sentiment_counts.index, rotation=0)
        ax.grid(True)
        
        # Set the face color of the figure and axes to white
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Close the figure to prevent it from displaying automatically
        plt.close(fig)
        
        return fig, ax
            

    
     