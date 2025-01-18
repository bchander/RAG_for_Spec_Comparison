'''
Document Comparison using basic RAG and just Prompting
without langgraph and graphRAG

Created by: Bhanu
Modified on: 27th Dec 2024

'''

#........Importing Required Libraries........#
import json
from io import BytesIO
import numpy as np
import pandas as pd
import fitz
import streamlit as st
from streamlit_chat import message
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
# load_dotenv()

#......Get environment variables......#
# openai.api_key = os.getenv("OPENAI_KEY")
# OPENAI_KEY = os.getenv("OPENAI_KEY")
# gpt_model = os.getenv("GPT_MODEL")
# embedding_model = os.getenv("EMBEDDING_MODEL")
# client = OpenAI(api_key = OPENAI_KEY)

#......Setting the environment variables manually......#
gpt_model = 'GPT_MODEL' #'gpt-3.5-turbo-0613' 
embedding_model = "EMBEDDING_MODEL"
client = OpenAI(api_key = "YOUR_KEY")

# Function to extract text from PDF file
def extract_text_from_pdf(file):
    #....using fitz to extract text from PDF file....#
    pdf_reader = fitz.open(stream=file.getvalue(), filetype="pdf")
    full_text = ""
    page_texts = []
    # for page_num in range(pdf_reader.getNumPages()):
    for page_num in range(pdf_reader.page_count):
        page = pdf_reader[page_num]
        text = page.get_text("text") 
        blocks = page.get_text("blocks")
        processed_blocks = []
        for b in blocks:
            block_text = b[4].strip()
            if block_text:
                if ":" in block_text:
                    processed_blocks.append(block_text)
                else:
                    processed_blocks.append(block_text.replace("\n", " "))
        processed_text = "\n".join(processed_blocks)
        full_text += processed_text + "\n\n"
        page_texts.append(processed_text)
        
    return full_text.strip(), page_texts


#......Function to extract specs from the input datasheet text......#
def extract_specs(text1, text2):
    response = client.chat.completions.create(
        model=gpt_model, 
        messages = [ {"role": "assistant", "content": """Identify the specifications from both the provided contexts and extract the values for the identified specifications from both the contexts. 
                      Compare the specifications from the provided two contexts and provide the comparison in a structured JSON format. Each specification should have an value entry for both contexts. 
                      The values should also include the associated units if available. If a specification is missing in one of the contexts, include 'N/A' for that entry.
                      The first two entires of the specification should be 'company' and 'product/Model Number'. Extract as many specifications as possible from both the provided contexts, even if they are not common to both.
                      Format your response as: 
                      {
                        "company": ["Value from context1", "Value from context2"],
                        "product/Model Number": ["Value from context1", "Value from context2"],
                        "Specification 3": ["Value from context1", "Value from context2"], 
                        "Specification 4": ["Value from context1", "Value from context2"], 
                        ... 
                      }
                      
                      """ 
                      },
                      {"role": "system", "content": f"context1: {text1}, context2: {text2}"},
                    #   {"role": "user", "content": query_text}
                      ], #prompt=prompt_template,
        max_tokens=1200,
        temperature = 0.1,
    )
    return response.choices[0].message.content


###............Main Function.............###
def main():
    st.header("Specification Comparator with Basic RAG ")
    
    #......Upload two PDF files to be compared......# 
    uploaded_file_1 = st.file_uploader("Upload First PDF", type="pdf")
    uploaded_file_2 = st.file_uploader("Upload Second PDF", type="pdf")
    
    submit=st.button("Submit")

    #If both files are uploaded and submit button is clicked, generate response......#
    if submit and uploaded_file_1 and uploaded_file_2:
        with st.spinner("Comparing Documents.....Please Wait"):
            #......Extracting text from uploaded PDF documents......#
            full_text_1, page_texts_1 = extract_text_from_pdf(uploaded_file_1)
            full_text_2, page_texts_2 = extract_text_from_pdf(uploaded_file_2)
            # text_list = [page_texts_1, page_texts_2]

            #......Extracting specifications from the extracted text using LLM......#
            spec_list_1 = extract_specs(page_texts_1, page_texts_2)

            # Parse the JSON response
            if spec_list_1:
                try:
                    data = json.loads(spec_list_1)
                    # st.write(data)
                    # Convert the JSON to a DataFrame
                    df = pd.DataFrame(data)
                    df2 = df.transpose().reset_index()
                    df2.columns = ['Specification', 'File 1', 'File 2']
                    print("JSON data loaded successfully.")

                    # Convert DataFrame to Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df2.to_excel(writer, index=False, sheet_name='comparison_results')
                        writer.close()
                    excel_data = output.getvalue()
                    
                    # Provide download button
                    st.download_button(
                        label="Download data as Excel",
                        data=excel_data,
                        file_name='comparison_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
            else:
                print("spec_list_1 is empty or None.")

            #......Display the comparison results......#
            st.write(" ") 
            st.write("Comparison Results as dataframe")
            st.write(data)
    else:
        st.write("Please upload the PDF files and click on Submit button to compare the documents")

if __name__=="__main__":
    main()
