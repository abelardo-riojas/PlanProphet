from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import os
from utils import generate_response, chunk_text, list_of_questions
from sklearn.metrics.pairwise import cosine_similarity
from numpy import argsort
from sentence_transformers import SentenceTransformer
import concurrent.futures

#TODO: checkout asym searches and hyde procedure
#TODO: add github (follow ticket) make a post 

list_of_questions = list_of_questions[:3]
#TODO: parallelizing requests, handling retries better, inspect api response when lagged the fuck out
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

#THROTTLE THIS HOW U WANT
NUM_PARALLEL_API_CALLS = 3

class GloVars:

    def __init__(self):

        self.all_chunks = []

        self.number_of_pages = 0

        self.chunk_embeddings = []        


session = GloVars()

def answer_question(question):

    print('embedding q')
    #embed the question
    question_embedding = model.encode([question])[0]

    print('finding simliarity')
    #get the scores
    similarity_scores = cosine_similarity([question_embedding], session.chunk_embeddings)[0]

    print('making context')
    #find the closest chunk and concatenate it's neighbors
    inds = argsort(similarity_scores)[-3:][::-1]
    contexts = []

    print(f'length of chunks: {len(session.all_chunks)}')
    for ind in inds:
        if ind == 0:
            context = session.all_chunks[ind] + session.all_chunks[ind+1]
        elif ind == len(session.all_chunks)-1:
            context = session.all_chunks[ind-1] + session.all_chunks[ind]
        else:
            context = session.all_chunks[ind-1] + session.all_chunks[ind] + session.all_chunks[ind+1]
        contexts.append(context)

    #TODO: add the top 3 chunks and neighbors
    #put the context in the prompt
    prompt = f'''
    The following are 2 sections of a healthcare plan design document. These sections were embedded and were determined to have the highest similarity to the question posed by the user. Answer the question to the best of your ability. The text was extracted from a PDF file therefore formatting issues or errant linebreaks may appear. If you are incapable of answering the question directly, simply state that you do not know.

    Second most relevant section:
    {contexts[1]}

    Most relevant section:
    {contexts[0]}
    '''


    print(prompt)

    print('sending to openai')
    #send the prompt and question to openai
    answer = generate_response(prompt, question)

    #huzzah! we have an answer
    #answer = response['choices'][0]['message']["content"]
    print('done')
    return answer, contexts


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['pdf']
        
        # Generate a secure filename to avoid any malicious input
        filename = secure_filename(file.filename)
        
        # Save the uploaded file to the UPLOAD_FOLDER
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read the PDF file using PdfReader
        reader = PdfReader(file_path)


        number_of_pages = len(reader.pages)
        session.number_of_pages = number_of_pages


        session.all_chunks = []
        session.chunk_embeddings = []

        for page_num in range(number_of_pages):
            # Extract the text from the page
            page = reader.pages[page_num]
            text = page.extract_text()

            # Split the page text into chunks
            n = 2  # Specify the number of chunks
            chunks = chunk_text(text, n)
            if chunks:
                session.all_chunks.extend(chunks)
        for chunk in session.all_chunks:
            chunk_embedding = model.encode([chunk])[0]
            session.chunk_embeddings.append(chunk_embedding)

        return redirect('/questions')

    return render_template('upload.html')


@app.route('/questions', methods=['GET', 'POST'])
def answer_questions():
    if request.method == 'POST':
        if 'question' in request.form:  # User submitted their own question
            question = request.form['question']
            answer, context = answer_question(question)
            return render_template('answer.html', question=question, answer=answer, context=context)
        else:  # User clicked "Answer Default Questions"
            # Process default questions and generate answers
            # Replace the following lines with your logic to answer default questions
            default_answers = []
            default_contexts = []
            num_threads = NUM_PARALLEL_API_CALLS  # Adjust this value based on the desired level of parallelism
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit each question to the executor and store the resulting Future objects
                question_futures = [executor.submit(answer_question, question) for question in list_of_questions]
                # Iterate over the question_futures list to retrieve the answers and contexts
                for future in concurrent.futures.as_completed(question_futures):
                    answer, context = future.result()
                    default_answers.append(answer)
                    default_contexts.append(context)

        return render_template('default_answers.html', questions=list_of_questions, answers=default_answers, contexts=default_contexts)
    
    return render_template('questions.html')


if __name__ == '__main__':
    app.run()
