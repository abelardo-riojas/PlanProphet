from flask import Flask, render_template, request, url_for, send_from_directory, redirect, Response
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from utils import generate_response, chunk_text
from sklearn.metrics.pairwise import cosine_similarity
from numpy import argsort
from sentence_transformers import SentenceTransformer

#init the model
model_name = 'multi-qa-mpnet-base-dot-v1'
model = SentenceTransformer(model_name)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
number_of_pages = 0
all_chunks = []
chunk_embeddings = []


def answer_question(question):
    #embed the question
    question_embedding = model.encode([question])[0]

    #get the scores
    similarity_scores = cosine_similarity([question_embedding], chunk_embeddings)[0]

    #find the closest chunk and concatenate it's neighbors
    ind = argsort(similarity_scores)[-1]
    if ind == 0:
        context = all_chunks[ind] + all_chunks[ind+1]
    elif ind == len(all_chunks)-1:
        context = all_chunks[ind-1] + all_chunks[ind]
    else:
        context = all_chunks[ind-1]+ all_chunks[ind] + all_chunks[ind+1]

    #put the context in the prompt
    prompt = f'''
    The following is a section of a healthcare plan design document. This section was embedded and was determined to have the highest similarity to the question posed by the user. Answer the question to the best of your ability. The text was extracted from a PDF file therefore formatting issues or errant linebreaks may appear. If you are incapable of answering the question directly, simply state that you do not know.

    {context}
    '''

    #send the prompt and question to openai
    response = generate_response(prompt, question)

    #huzzah! we have an answer
    answer = response['choices'][0]['message']["content"]

    return answer


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
            n = 3  # Specify the number of chunks
            chunks = chunk_text(text, n)
            if chunks:
                all_chunks.extend(chunks)


        for chunk in all_chunks:
            chunk_embedding = model.encode([chunk])[0]
            chunk_embeddings.append(chunk_embedding)

        return render_template('qa.html')
    
    return render_template('upload.html')




if __name__ == '__main__':
    app.run()
