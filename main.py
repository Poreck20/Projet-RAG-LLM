import os
import getpass
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate



llm = ChatCohere(model="command-r-plus", temperature=0.1)  
documents = []
for file_path in ["C:/Users/bouba/Nouvelle_vie/Projet_RAG_LLM/Data/old_cv.pdf"]: 
    loader =PyPDFLoader(file_path)
    documents.extend(loader.load())
    
       
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("COHERE_API_KEY")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

embeddings = CohereEmbeddings(cohere_api_key=key, model="embed-english-v2.0")
vectorstore = Chroma.from_documents(splits, embeddings)

retriever = vectorstore.as_retriever()


cv_prompt_template = """
Vous allez générer un CV orienté Data Science à partir des informations suivantes. Organisez les informations de manière claire et professionnelle.

Voici les sections à inclure :
      DIALLO Alpha 
1. **Expérience professionnelle**
    - Inclure toutes les experiences 
    - Résumer chaque expérience en 2-3 points.
2. **Compétences techniques**
    - Liste des outils, langages et frameworks spécifiques à la Data Science.
    - Inclure des compétences générales comme "résolution de problèmes", "analyse statistique", etc.
3. **Formation**
    - Indiquer tous les diplômes et certificats obtenus. Je suis actuellement en Master 1.
    - Préciser les cours ou spécialisations liés à la Data Science.
4. **Projets**
    - Décrire brièvement 2-3 projets pertinents (par exemple, "Prédiction des ventes avec Machine Learning").
5. **Autres informations**
    - Langues parlées, hobbies, ou toute autre information pertinente.

Sections de texte récupérées : {context}

Générez un CV structuré et concis avec des sous-sections claires et des listes à puces. Utilisez un style professionnel et soigné.

"""
prompt = PromptTemplate(input_variables=["context"], template=cv_prompt_template)



from IPython.display import Markdown, display

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    retriever | format_docs 
    | prompt  
    | llm 
)

cv = rag_chain.invoke("Générer un nouveau CV orienté Data Science")
cv_text = cv.content  

display(Markdown(f"### CV généré\n{cv_text}"))
# print("CV généré\n",cv_text)