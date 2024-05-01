from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Replace with your actual OpenAI API key


# Initialize the OpenAI LLM with your API key
llm = OpenAI(openai_api_key=openai_api_key)

# Define a prompt template for classification
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Determine if the following text is a tweet or a question: \n\n\"{text}\"\n\nAnswer: "
)

# Example input text
input_text = "What is the weather like today?"

# Format the prompt with the actual text
prompt = prompt_template.format(text=input_text)

# Use the LLM to generate text based on the prompt
response = llm(prompt)
print(response)