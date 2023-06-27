from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import os
from utils import generate_response, chunk_text
from sklearn.metrics.pairwise import cosine_similarity
from numpy import argsort
from sentence_transformers import SentenceTransformer


#jenni's list of questions for an spd
list_of_questions = [
    "Is Applied Behavior Analysis (Autism) therapy covered?",
    "Is weight loss (Bariatric/obesity) surgery covered?",
    "Is Orthognathic (jaw) surgery covered?",
    "Is Temporomandibular joint dysfunction (TMJ) surgery covered and what are the limits?",
    "Is Diabetes self-management training covered and what are the limits?",
    "What coverage is there for breast pumps and what are the limits?",
    "What coverage is there for diabetes supplies, is insulin counted, and what are the limits?",
    "What coverage is there for foot orthotics and what are the limits?",
    "What coverage is there for hearing aids and what are the limits?",
    "What coverage is there for wigs and what are the limits?",
    "Are non-therapeutic (elective) abortions covered?",
    "Are contraceptive procedures covered?",
    "Are fertility donor services covered and do they require a diagnosis of infertility?",
    "Are surrogacy services covered and do they require a diagnosis of infertility?",
    "Are out of network (OON) options covered for fertility treatments?",
    "What are the vendor options and limits for fertility treatment?",
    "Is Iatrogenic Infertility Treatment (Fertility Preservation) covered and what are the limits?",
    "For those who suffer from gender dysphoria, is gender reassignment surgery covered?",
    "For those who suffer from gender dysphoria, are gender reconstruction services covered? These include hormone replacement therapy and non-surgical options like laser hair removal, body contouring, and cosmetic procedures.",
    "For those who suffer from gender dysphoria, is vocal therapy covered?",
    "For those who require assisted living or medical care at the home, is Home Health Care covered and what are the visitation limits?",
    "For those who require assisted living or medical care at the home, is Private Duty Nursing covered and what are the visitation limits?",
    "Is preventative nutritional counseling covered?",
    "For those who suffer from mental disorders related to eating, is nutritional counseling covered?",
    "Are other forms of nutritional counseling covered beyond preventative and mental health related counseling?",
    "Is BRCA testing covered and what are the limits?",
    "Is Acupuncture covered and what are the limits?",
    "Is Cardiac Rehabilitative Therapy covered and what are the limits?",
    "Are Chiropractic services covered and what are the limits?",
    "Is Occupational Therapy covered and what are the limits?",
    "Is Pulmonary Rehabilitative Therapy covered and what are the limits?",
    "Is Speech therapy covered and what are the limits?",
    "Are services from a Skilled Nursing Facility covered and what is the day limit?",
    "What are the limits for travel and lodging expenses for medical procedures? This includes minimum travel distance, transportation, lodging, and meals as well as any other limits imposed.",
    "Are routine eye exams covered and what are the limits?",
    "Is Vision Therapy covered and what are the limits?"
]
#TODO: checkout asym searches and hyde manuver


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
number_of_pages = 0
all_chunks = []
chunk_embeddings = []
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)


def answer_question(question):

    print('embedding q')
    #embed the question
    question_embedding = model.encode([question])[0]

    print('finding simliarity')
    #get the scores
    similarity_scores = cosine_similarity([question_embedding], chunk_embeddings)[0]

    print('making context')
    #find the closest chunk and concatenate it's neighbors
    inds = argsort(similarity_scores)[-3:][::-1]
    contexts = []
    for ind in inds:
        if ind == 0:
            context = all_chunks[ind] + all_chunks[ind+1]
        elif ind == len(all_chunks)-1:
            context = all_chunks[ind-1] + all_chunks[ind]
        else:
            context = all_chunks[ind-1] + all_chunks[ind] + all_chunks[ind+1]
        contexts.append(context)

    #TODO: add the top 3 chunks and neighbors
    #put the context in the prompt
    prompt = f'''
    The following are 3 sections of a healthcare plan design document. These sections were embedded and were determined to have the highest similarity to the question posed by the user. Answer the question to the best of your ability. The text was extracted from a PDF file therefore formatting issues or errant linebreaks may appear. If you are incapable of answering the question directly, simply state that you do not know.

    Third most relevant section:
    {contexts[2]}

    Second most relevant section:
    {contexts[1]}

    Most relevant section:
    {contexts[0]}
    '''

    print('sending to openai')
    #send the prompt and question to openai
    response = generate_response(prompt, question)

    #huzzah! we have an answer
    answer = response['choices'][0]['message']["content"]
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

        for page_num in range(number_of_pages):
            # Extract the text from the page
            page = reader.pages[page_num]
            text = page.extract_text()

            # Split the page text into chunks
            n = 2  # Specify the number of chunks
            chunks = chunk_text(text, n)
            if chunks:
                all_chunks.extend(chunks)


        for chunk in all_chunks:
            chunk_embedding = model.encode([chunk])[0]
            chunk_embeddings.append(chunk_embedding)

        return redirect('/questions')
    
    return render_template('upload.html')


@app.route('/questions', methods=['GET', 'POST'])
def answer_questions():
    if request.method == 'POST':
        question = request.form['question']
        answer, context = answer_question(question)
        return render_template('answer.html', question=question, answer=answer, context=context)
    else:
        batch_size = 5  # Number of default questions to load in each batch
        start_index = int(request.args.get('start_index', 0))
        end_index = start_index + batch_size
        default_questions = list_of_questions  # Replace this with your own function to retrieve default questions
        
        if start_index >= len(default_questions):
            # No more questions left
            return render_template('no_questions.html')
        
        batch_questions = default_questions[start_index:end_index]
        num_questions = len(default_questions)  # New line
        
        return render_template('questions.html', questions=batch_questions, start_index=start_index, end_index=end_index, num_questions=num_questions)  # Updated line



if __name__ == '__main__':
    app.run()
