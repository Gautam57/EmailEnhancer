import streamlit as st

from llama_cpp import Llama

import re
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

# st.set_option('server.enableCORS', False)
# st.set_option('server.enableXsrfProtection', False)
# st.set_option('server.fileWatcherType', none)  # 👈 Disables the problematic watcher

# from langchain.prompts import PromptTemplate
# from langchain_community.llms import CTransformers


@st.cache_resource
def load_llama_model():
    return Llama.from_pretrained(
        repo_id="tensorblock/Llama-3-OffsetBias-8B-GGUF",
        filename="Llama-3-OffsetBias-8B-Q5_K_M.gguf",
    )

@st.cache_resource
def load_hf_model():
    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_id = "microsoft/Phi-4-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,load_in_4bit=True,device_map="auto"  )
    return tokenizer, model

def generate_prompt(Email_tone, CreatedBody):
     ## Prompt Template

    # template="""Enhance {CreatedBody} by fixing grammar and spelling, and updating the tone to {Email_tone} based on the received email: {Received_emailBody}"""
    # template="""Update the email tone to {Email_tone}, by correcting the grammar and spelling of email body : {CreatedBody}"""

    template = f"""Please rewrite the following email in a {Email_tone} tone, correcting grammar and spelling:
                    "{CreatedBody}" """
    single_line_template = re.sub(r'\s+', ' ', template).strip()
    print(single_line_template)

    return single_line_template

def getLLamaresponse(Email_tone, CreatedBody):
    try:
        llm = load_llama_model()
        Updated_prompt = generate_prompt(Email_tone, CreatedBody)
        
        num_tokens = len(llm.tokenize(Updated_prompt.encode("utf-8")))
        print(f"Token count: {num_tokens}")

        start_time = time.time()

        # Using the prompt template to generate the response
        response = llm(Updated_prompt, max_tokens=num_tokens+50, temperature=0.1,stop=["Return only the improved email."])

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for generation: {(elapsed_time/60):.2f} seconds")

        return response["choices"][0]["text"]
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def getHuggingFaceResponse(Email_tone, CreatedBody):
    try:
        tokenizer, model = load_hf_model()
        Updated_prompt = generate_prompt(Email_tone, CreatedBody)
        
        start_time = time.time()
        # Using the tokenizer and model to generate the response
        inputs = tokenizer(Updated_prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for generation: {(elapsed_time/60):.2f} seconds")

            # Decode the output and remove the prompt
        return tokenizer.decode(output[0], skip_special_tokens=True).replace(Updated_prompt, "").strip()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None


st.set_page_config(page_title="Email Enhancer",
                    page_icon='✉️',
                    layout='centered',
                    initial_sidebar_state='collapsed',
                    menu_items={'About': "An Webpage which helps to enhance your email with corrected grammer and spellings"}
                    )

st.header("Email Enhancer 🤖")

CreatedBody=st.text_area("Enter the Email Body you Created")

## creating to more columns for additonal 2 fields

Email_tone=st.selectbox('Select the tone of the email',
                            ('Formal','Neutral','Friendly','Apologetic','Assertive','Gentle',
                            'Sarcastic','Empathetic','Encouraging','Humorous','Grateful','Critical',
                            'Optimistic','Pessimistic','Inquisitive','Cautious','Urgent','Dismissive',
                            'Defensive','Diplomatic','Sympathetic','Excited','Irritated','Inspirational'),index=0)

submit=st.button("Enhance")

## Final response
if submit:
    with st.spinner('Enhancing your email...'):
        st.write(getLLamaresponse(Email_tone,CreatedBody))
        # st.write(getHuggingFaceResponse(Email_tone,CreatedBody)) 

