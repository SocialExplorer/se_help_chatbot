from groq import Groq
from pinecone import Pinecone
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_relevant_excerpts(user_question, docsearch):
    """
    This function retrieves the most relevant excerpts from ACS data based on the user's question.
    Parameters:
    user_question (str): The question asked by the user.
    docsearch (PineconeVectorStore): The Pinecone vector store containing the ACS data.
    Returns:
    str: A string containing the most relevant excerpts from ACS data.
    """

    # Perform a similarity search on the Pinecone vector store using the user's question
    relevant_docs = docsearch.similarity_search(user_question)

    # Extract the page content from the top 3 most relevant documents and join them into a single string
    #relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevant_docs[:3] if 'page_content' in doc])

    return relevant_docs


def acs_data_chat_completion(client, model, user_question, relevant_excerpts):
    """
    Generates a response to the user's question using the pre-trained Llama model.

    Parameters:
    client (Groq): The Groq client used to interact with the pre-trained model.
    model (str): The name of the pre-trained model.
    user_question (str): The question asked by the user.
    relevant_excerpts (str): A string containing the most relevant excerpts from ACS data.

    Returns:
    str: A string containing the response to the user's question.
    """
    # Define the system prompt
    system_prompt = '''
    You are a data scientist specializing exclusively in Census American Community Survey (ACS) data and Social Explorer public data.
    You work as a chatbot at Social Explorer, a company dedicated to providing access to and analysis of census data.
    Your task is to assist users with questions related solely to census ACS data.

    Use the provided data excerpts to answer the user's question as accurately as possible.
    If the information is not available in the excerpts, admit that you don't have that information.
    '''

    user_message = f'''
    User Question: {user_question}

    Relevant Data Excerpt(s):

    {relevant_excerpts}
    '''

    # Generate a response to the user's question using the pre-trained model
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_message.strip()
            }
        ],
        model=model
    )

    # Extract the response from the chat completion
    response = chat_completion.choices[0].message.content

    return response

def main():
    """
    This is the main function that runs the application. It initializes the Groq client and the HuggingFaceEmbeddings model,
    gets user input from the Streamlit interface, retrieves relevant excerpts from ACS data based on the user's question,
    generates a response to the user's question using a pre-trained model, and displays the response.
    """

    model = 'llama3-70b-8192'

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize the Groq client
    groq_api_key = os.getenv('GROQ_API_KEY')
    pinecone_api_key=os.getenv('PINECONE_API_KEY')
    pinecone_index_name = "acs-tables"
    client = Groq(
        api_key=groq_api_key
    )

    pc = Pinecone(api_key = pinecone_api_key)
    docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)

    # Display the title and introduction of the application
    print("ACS Data Chatbot")
    multiline_text = """
    Welcome! Ask questions about Census American Community Survey (ACS) data. The app matches your question to relevant excerpts from ACS data and generates a response using a pre-trained model.
    """

    print(multiline_text)


    while True:
        # Get the user's question
        user_question = input("Ask a question about ACS data: ")

        if user_question:
            # Retrieve relevant excerpts
            relevant_excerpts = get_relevant_excerpts(user_question, docsearch)

            # Generate response using the Llama model via Groq client
            response = acs_data_chat_completion(client, model, user_question, relevant_excerpts)
            print("\nResponse:")
            print(response)
            print("\n" + "=" * 60 + "\n")
        else:
            print("Please enter a question.")


if __name__ == "__main__":
    main()