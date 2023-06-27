from retry import retry
import openai
from langchain.chains.openai_functions import create_citation_fuzzy_match_chain
from langchain.prompts.chat import SystemMessage
from langchain.chat_models import ChatOpenAI
import openai

openai.api_key_path = "surest_openai.key"



def chunk_text(text, n):
    # text into n equal-sized chunks
    chunk_size = len(text) // n
    if chunk_size == 0:
        # don't return a chunk!
        return None
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


def generate_response(prompt, question):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    chain = create_citation_fuzzy_match_chain(llm)
    chain.prompt.messages[0] = SystemMessage(content='You are a health insurance expert who answers questions with correct and exact citations. Your job is to read parsed PDF files of health insurance design documents (known as Standard Plan Documents or SPDs). These documents might include spelling mistakes or errant line breaks, so be careful.' )

    result = chain.run(question=question, context=prompt)

    full_answer = []
    for idx,fact in enumerate(result.answer):
        full_answer.append(f"Fact #{idx+1}\t\t"+fact.fact + '\t\t'+ str(fact.substring_quote))

    return full_answer


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