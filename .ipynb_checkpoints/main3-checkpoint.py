import os
import getpass
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate

# Entrer la clé API Cohere
os.environ["COHERE_API_KEY"] = getpass.getpass("4Mr58t0OfgVdvXK3sKtNrujHRNoo8Iax3VvULbn0")

# Charger le modèle Cohere avec température (ajoutez la température ici)
llm = ChatCohere(model="command-r-plus", temperature=0.7)  # température ajustable entre 0 et 1

# Charger les anciens documents (par exemple, des fichiers PDF ou d'autres formats)
documents = []
for file_path in [
    "C:/Users/bouba/Nouvelle_vie/Projet_RAG_LLM/Data/old_cv1.pdf", 
    "C:/Users/bouba/Nouvelle_vie/Projet_RAG_LLM/Data/old_cv2.pdf"
]:  # Vos chemins de fichiers
    loader =PyPDFLoader(file_path)
    documents.extend(loader.load())

# Fractionner les documents en morceaux plus petits
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Indexer les morceaux de texte avec Cohere
embeddings = CohereEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding_function=embeddings)

# Créer un récupérateur pour récupérer les informations pertinentes
retriever = vectorstore.as_retriever()

# Créer un prompt pour générer un CV
cv_prompt_template = """
Vous allez générer un CV à partir des informations suivantes. Organisez les informations de manière claire et professionnelle.

Voici les sections à inclure :
1. Expérience professionnelle
2. Compétences
3. Formation
4. Autres informations (projets, certifications, etc.)

Sections de texte récupérées : {context}

Générez un CV structuré et concis. Si une section manque d'informations, indiquez-le.

"""
prompt = PromptTemplate(input_variables=["context"], template=cv_prompt_template)

# Créer la chaîne RAG pour générer le CV
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Utilisation correcte de l'opérateur `|`
rag_chain = (
    retriever | format_docs  # Récupérer et formater les documents
    | prompt  # Appliquer le modèle de prompt
    | llm  # Utiliser le modèle de génération
)

# Générer le CV
cv = rag_chain.invoke("Générer mon CV à partir des documents ci-dessus")
print("CV généré : ", cv)
