import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from root directory
from exception import CustomException
from logger import logger
class LLMService:
    def __init__(self):
        load_dotenv()
        self.model  =ChatGroq(api_key=os.getenv("GROQ_API_KEY"),model = "llama-3.3-70b-versatile", max_tokens=1000)

    def get_ticket_summary(self, ticket_text: str):
        try:
            prompt_template = '''you are a customer support assistant. Summarize the following customer support ticket in one sentence:
            Ticket: {ticket_text}
            summary:'''
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.model
            response = chain.invoke({"ticket_text": ticket_text})
            return response
        except Exception as e:
            logger.error(f"Error occurred while getting ticket summary: {e}")
            raise CustomException(e, sys)
        
    def extract_ticket_info(self, ticket_text:str)->dict:
        try:
            prompt_template = '''you are a customer support assistant. Extract the following information from the customer support ticket:
            1. Customer Name
            2. Issue Type (e.g., billing, technical, account, feedback)
            3. Urgency Level (low, medium, high)
            4. Preferred Contact Method (email, phone, chat)
            Ticket: {ticket_text}
            Provide the information in JSON format with keys: 'customer_name', 'product_name', 'order_id'.
            JSON:{format_instructions} '''
            output_parser = JsonOutputParser()
            prompt = ChatPromptTemplate.from_template(
                template  =prompt_template,partial_variables={"format_instructions":output_parser.get_format_instructions()}
            )
            chain = prompt|self.model|output_parser
            response = chain.invoke({
                "ticket_text": ticket_text
            })
            return response
        except Exception as e:
            logger.error(f"Error occurred while extracting ticket info: {e}")
            raise CustomException(e, sys)
        