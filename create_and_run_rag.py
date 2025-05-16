import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List
from langchain_core.documents import Document
from huggingface_hub import InferenceClient  
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configurer_les_cles_api():
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Clé API Pinecone non trouvée dans les variables d'environnement.")
    if not huggingface_api_key:
        raise ValueError("Clé API Hugging Face non trouvée dans les variables d'environnement.")
    return pinecone_api_key, huggingface_api_key

def initialiser_pinecone(api_key: str):
    try:
        pc = Pinecone(api_key=api_key)
        return pc
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de Pinecone : {e}")
        raise

def creer_index_pinecone(pc: Pinecone, index_name: str = "documents-index", dimension: int = 384, metric: str = 'cosine', cloud: str = 'aws', region: str = 'us-east-1'):
    try:
        if index_name not in pc.list_indexes(): 
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            logging.info(f"Index Pinecone '{index_name}' créé avec succès.")
        else:
            logging.info(f"L'index Pinecone '{index_name}' existe déjà.")
    except Exception as e:
        logging.error(f"Erreur lors de la création de l'index Pinecone : {e}")
        raise

def charger_documents(chemin_dossier: str) -> List[Document]:
    try:
        loader = DirectoryLoader(chemin_dossier, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = chunk.metadata.get('source', 'Inconnu')
        logging.info(f"Documents chargés et divisés en {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logging.error(f"Erreur lors du chargement des documents : {e}")
        raise

def creer_embeddings_et_stocker(chunks: List[Document], index_name: str, api_key: str) -> PineconeVectorStore:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        vectorstore = PineconeVectorStore.from_documents( # Utilisez from_documents
            chunks,
            embeddings,
            index_name=index_name,  # Ajoutez index_name ici
        )
        logging.info("Embeddings créés et stockés dans Pinecone.")
        return vectorstore
    except Exception as e:
        logging.error(f"Erreur lors de la création et du stockage des embeddings : {e}")
        raise

def configurer_rag_mcp(vectorstore: PineconeVectorStore, huggingface_api_key: str):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        client = InferenceClient(token=huggingface_api_key)

        def llm(prompt):
            response = client.text_generation(prompt=prompt, model="mistralai/Mistral-7B-v0.3", max_new_tokens=1024, temperature=0.7, top_p=0.95)
            return response

        template = """<context>
{context}
</context>

Réponds à la question suivante en utilisant uniquement le contexte fourni ci-dessus.
Si tu ne trouves pas l'information dans le contexte, indique-le clairement.
Tu dois obligatoirement citer les sources des documents utilisés pour formuler ta réponse.
Inclus les noms des fichiers sources à la fin de ta réponse sous forme de liste de références.

Question: {question}

Réponse:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs: List[Document]) -> str:
            formatted_docs = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Source inconnue')
                formatted_docs.append(f"[Document {i + 1}] Source: {source}\nContenu: {doc.page_content}")
            return "\n\n".join(formatted_docs)

        rag_chain = (
            {"context": (lambda q: format_docs(retriever.get_relevant_documents(q))), "question": RunnablePassthrough()} # Utilisez retriever.get_relevant_documents
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        logging.error(f"Erreur lors de la configuration du pipeline RAG : {e}")
        raise

def main():
    chemin_documents = "extracted_text/"

    try: 
        pinecone_api_key, huggingface_api_key = configurer_les_cles_api()
        pc = initialiser_pinecone(pinecone_api_key)
        creer_index_pinecone(pc)
        chunks = charger_documents(chemin_documents)
        vectorstore = creer_embeddings_et_stocker(chunks, "documents-index", pinecone_api_key)
        rag_chain = configurer_rag_mcp(vectorstore, huggingface_api_key)

        questions = ['Quel musée est le plus beau à Lyon ?','Combien il y a de métro à Lyon ?']
        for question in questions:
            réponse = rag_chain.invoke(question)
            print(f"\nRéponse: {réponse}")
    except Exception as e:
        logging.error(f"Une erreur s'est produite lors de l'exécution du programme : {e}")

if __name__ == "__main__":
    main()