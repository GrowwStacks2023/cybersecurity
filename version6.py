from flask import Flask, request, render_template_string
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import openai
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

openai_api_key = "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"


csv_files = [
    {"name": "SCF", "pkl": "/home/vyavasthapak/Daniel/SCF/", "faiss": "/home/vyavasthapak/Daniel/SCF/"},
    {"name": "NIST1", "pkl": "/home/vyavasthapak/Daniel/NIST1/", "faiss": "/home/vyavasthapak/Daniel/NIST1/"},
    {"name": "NISTSMB", "pkl": "/home/vyavasthapak/Daniel/NISTSMB/", "faiss": "/home/vyavasthapak/Daniel/NISTSMB/"},
    {"name": "SMB", "pkl": "/home/vyavasthapak/Daniel/SMB/", "faiss": "/home/vyavasthapak/Daniel/SMB/"}
]

def load_vector_store(pkl_path, faiss_path):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    try:
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"Error loading FAISS vector store: {e}")
        raise
    return vectorstore

vectorstores = [load_vector_store(f['pkl'], f['faiss']) for f in csv_files]

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer in the format:
SCF Control:
Methods:
Description:\n

SCF Control:
Methods:
Description:\n

If there are multiple SCF controls, provide the list of them in the above format and start the next one from new line by leaving 3 line space between.
SCF Control:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chat_history = []

context_mapping = [
    {"first_csv_file_header": "NIST CSF", "second_csv_file_header": "NIST ID"},
    {"first_csv_file_header": "Function Grouping", "second_csv_file_header": "Group"},
    {"second_csv_file_header": "Group", "third_csv_file_header": "Group"},
    {"second_csv_file_header": "Sub-Group", "third_csv_file_header": "Sub Group"},
    {"second_csv_file_header": "Sub-Group 2", "third_csv_file_header": "Sub Group 2"},
    {"third_csv_file_header": "Group", "fourth_csv_file_header": "Group"},
    {"third_csv_file_header": "Sub Group", "fourth_csv_file_header": "Subgroup 1"},
    {"third_csv_file_header": "Sub Group 2", "fourth_csv_file_header": "Subgroup 2"}
]

def get_next_header(current_header, direction):
    for mapping in context_mapping:
        if current_header in mapping:
            return mapping.get(direction)
    return None

def reformat_response(response):
    openai.api_key = openai_api_key
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Reformat the following text into JSON format with each SCF control as a separate JSON object containing 'SCF', 'Methods', and 'Description'. Ensure each SCF control is correctly parsed and formatted as JSON. Also, elaborate the description using NLP and AI."
                },
                {
                    "role": "user",
                    "content": response
                }
            ]
        )
        formatted_text = completion.choices[0].message['content']

        try:
            formatted_json = json.loads(formatted_text)
        except json.JSONDecodeError:
            parts = formatted_text.split("SCF Control:")
            formatted_parts = []
            for part in parts:
                if part.strip():
                    scf_control = part.strip()
                    method_index = scf_control.find("Methods:")
                    description_index = scf_control.find("Description:")

                    scf = scf_control[:method_index].strip() if method_index != -1 else ""
                    methods = scf_control[method_index:description_index].strip() if method_index != -1 and description_index != -1 else ""
                    description = scf_control[description_index:].strip() if description_index != -1 else ""

                    formatted_part = {
                        "SCF": scf.replace("SCF Control:", "").strip(),
                        "Methods": methods.replace("Methods:", "").strip(),
                        "Description": description.replace("Description:", "").strip()
                    }
                    formatted_parts.append(formatted_part)
            formatted_json = formatted_parts

        return json.dumps(formatted_json, indent=4)

    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return json.dumps({"error": "An error occurred while formatting the response."}, indent=4)

def calculate_similarity(query, text):
    vectorizer = TfidfVectorizer().fit_transform([query, text])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def rank_controls(user_input, controls):
    rankings = []

    for control in controls:
        scf = control['SCF']
        methods = control['Methods']
        description = control['Description']
        
        
        input_relevance = calculate_similarity(user_input, description + " " + methods)
        domain_relevance = calculate_similarity(user_input, scf)
        control_description_relevance = calculate_similarity(user_input, scf + " " + description)
        
        
        final_score = (input_relevance * 0.4) + (domain_relevance * 0.3) + (control_description_relevance * 0.3)
        
        rankings.append({
            "SCF": scf,
            "Methods": methods,
            "Description": description,
            "Input Relevance": input_relevance,
            "Domain Relevance": domain_relevance,
            "Control + Description Relevance": control_description_relevance,
            "Final Score": final_score
        })
    
    # Sort controls by final score
    rankings.sort(key=lambda x: x['Final Score'], reverse=True)
    
    return rankings

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_question = request.form["question"]
        selected_vectors = request.form.getlist("vector_spaces")
        selected_vectors = [int(i) for i in selected_vectors]  # Convert to integers

        output_texts = []

        # Process through selected vector stores only
        context = user_question
        for vectorstore_index in selected_vectors:
            vectorstore = vectorstores[vectorstore_index]
            docs = vectorstore.similarity_search(context)
            response = chain.run(input_documents=docs, question=user_question, prompt=PROMPT)
            formatted_response = reformat_response(response)
            output_texts.append(formatted_response)

            # Determine the next header
            next_header = get_next_header(
                list(csv_files[vectorstore_index].values())[0],  # current header in the current file
                "second_csv_file_header" if vectorstore_index < 3 else "fourth_csv_file_header"
            )
            context = formatted_response  # Update context for next CSV

        # Save to a text file
        with open("output.txt", "w") as f:
            for text in output_texts:
                f.write(text + "\n\n")

        # Rank the controls based on relevance
        controls = []
        for output in output_texts:
            controls.extend(json.loads(output))

        ranked_controls = rank_controls(user_question, controls)

        # Save the ranked results to final.txt
        with open("final.txt", "w") as f:
            for rank, control in enumerate(ranked_controls, start=1):
                f.write(f"Rank {rank}:\n")
                f.write(f"SCF: {control['SCF']}\n")
                f.write(f"Methods: {control['Methods']}\n")
                f.write(f"Description: {control['Description']}\n")
                f.write(f"Input Relevance: {control['Input Relevance']:.4f}\n")
                f.write(f"Domain Relevance: {control['Domain Relevance']:.4f}\n")
                f.write(f"Control + Description Relevance: {control['Control + Description Relevance']:.4f}\n")
                f.write(f"Final Score: {control['Final Score']:.4f}\n\n")
        
        chat_history.append(("User", user_question))
        chat_history.append(("Bot", "\n".join(final.txt)))
    
    chat_html = ""
    for speaker, message in chat_history:
        chat_html += f"<p><strong>{speaker}:</strong> <pre>{message}</pre></p>"
    
    return render_template_string(f'''
        <!doctype html>
        <html>
        <head>
            <title>Cyber Security AI</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .chat-container {{
                    width: 80%;
                    max-width: 1500px;
                    background-color: white;
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                .chat-box {{
                    max-height: 900px;
                    overflow-y: auto;
                    padding: 15px;
                    height: 600px;
                    width: 96%;
                    background: #EEEDED;
                    border: 1px solid #ccc;
                    margin-bottom: 25px;
                    border-radius: 7px;
                    padding: 13px;
                    overflow-y: scroll;
                }}
                .input-box {{
                    width: 100%;
                    display: flex;
                    justify-content: space-between;
                }}
                .input-box input {{
                    width: 85%;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }}
                .input-box button {{
                    width: 12%;
                    padding: 10px;
                    border: none;
                    border-radius: 5px;
                    background-color: #EF7810;
                    color: white;
                    font-size: 16px;
                }}
                .dropdown-container {{
                    margin-bottom: 20px;
                    width: 100%;
                }}
                .dropdown-container label {{
                    font-weight: bold;
                }}
                .dropdown-container select {{
                    width: 100%;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                }}
            </style>
        </head>
        <body>
            <div class="chat-container">
                <h1>Cyber Security AI</h1>
                <div class="chat-box">
                    {chat_html}
                </div>
                <form method="post" class="input-box">
                    <div class="dropdown-container">
                        <label for="vector_spaces">Select CSV Files to Search:</label>
                        <select name="vector_spaces" id="vector_spaces" multiple required>
                            {''.join([f'<option value="{i}">{csv["name"]}</option>' for i, csv in enumerate(csv_files)])}
                        </select>
                    </div>
                    <input type="text" id="question" name="question" placeholder="Type your message..." required>
                    <button type="submit">Send</button>
                </form>
            </div>
        </body>
        </html>
    ''')

if __name__ == "__main__":
    app.run(debug=True)