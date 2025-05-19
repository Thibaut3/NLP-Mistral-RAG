# NLP-Mistral-RAG

Ce projet met en œuvre une application de Chatbot utilisant le modèle de langage Mistral-7B-Instruct-v0.3 et l'architecture RAG (Retrieval Augmented Generation). Il exploite Pinecone pour le stockage et la récupération efficaces des embeddings de documents.

## Fonctionnalités

* **RAG avec Mistral 7B Instruct v0.3 :** Utilise un modèle de langage puissant pour générer des réponses informatives aux questions.
* **Vectorisation avec Hugging Face :** Convertit les documents en embeddings vectoriels à l'aide des modèles Sentence Transformers.
* **Stockage vectoriel avec Pinecone :** Stocke et indexe les embeddings dans Pinecone pour une récupération rapide et évolutive.
* **Chargement de documents :** Charge les documents à partir d'un répertoire local, les traite par chunks et les prépare pour l'indexation.
* **Invite personnalisée :** Utilise une invite personnalisée pour fournir un contexte au modèle de langage et améliorer la qualité des réponses.
* **Gestion des erreurs :** Inclut une gestion robuste des erreurs pour divers scénarios, tels que les échecs d'initialisation de l'API, les problèmes de chargement de documents et les erreurs de création d'index.
* **Logging :** Utilise le logging pour suivre les étapes importantes du processus et faciliter le débogage.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les éléments suivants :

* Python 3.7 ou supérieur
* Pip
* Un compte Pinecone et une clé API
* Un compte Hugging Face et une clé API
* Les variables d'environnement configurées pour les clés API :
    * `PINECONE_API_KEY`
    * `HUGGINGFACE_API_KEY`

## Installation

1.  Clonez ce référentiel :

    ```bash
    git clone https://github.com/Thibaut3/NLP-Mistral-RAG.git
    cd votre-repo
    ```

2.  Créez un environnement virtuel (recommandé) :

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Pour Linux et macOS
    venv\Scripts\activate  # Pour Windows
    ```

## Configuration

1.  **Configurez les clés API :**
    * Définissez les variables d'environnement `PINECONE_API_KEY` et `HUGGINGFACE_API_KEY` avec vos clés API Pinecone et Hugging Face respectivement. Vous pouvez le faire en utilisant la ligne de commande ou en définissant les variables dans votre fichier `.bashrc`, `.zshrc` ou un outil similaire.

2.  **Préparez vos documents :**
    * Placez vos fichiers texte dans le répertoire `extracted_text/`. Vous pouvez modifier le chemin d'accès dans le script si nécessaire.

## Utilisation

1.  Exécutez le script :

    ```bash
    python create_and_run_rag.py
    ```

Le script effectuera les étapes suivantes :

* Charge les documents à partir du répertoire spécifié.
* Crée des embeddings pour les documents à l'aide de Hugging Face Sentence Transformers.
* Stocke les embeddings dans un index Pinecone.
* Configure un pipeline RAG à l'aide de Mistral 7B Instruct v0.3.
* Pose une série de questions et affiche les réponses générées.

## Structure du code

* `create_and_run_rag.py` : Le script Python principal qui met en œuvre l'application RAG.
* `requirements.txt` : Un fichier contenant la liste des dépendances Python requises.
* `extracted_text/` : Un répertoire où vous devez placer vos fichiers texte de documents.

### Fonctions clés

* `configurer_les_cles_api()` : Récupère les clés API Pinecone et Hugging Face à partir des variables d'environnement.
* `initialiser_pinecone(api_key: str)` : Initialise le client Pinecone.
* `creer_index_pinecone(pc: Pinecone, index_name: str, dimension: int, metric: str, cloud: str, region: str)` : Crée un index Pinecone pour stocker les embeddings.
* `charger_documents(chemin_dossier: str)` : Charge les documents à partir d'un répertoire, les divise en chunks et extrait le texte.
* `creer_embeddings_et_stocker(chunks: List[Document], index_name: str, api_key: str)` : Génère des embeddings pour les chunks de texte et les stocke dans Pinecone.
* `configurer_rag_mcp(vectorstore: PineconeVectorStore, huggingface_api_key: str)` : Configure le pipeline RAG avec le modèle Mistral 7B Instruct v0.3, l'invite et le retriever.
* `main()` : La fonction principale qui coordonne les différentes étapes du processus.

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres suivants :

* `chemin_documents` : Modifiez le chemin d'accès au répertoire contenant vos documents.
* `index_name` : Modifiez le nom de l'index Pinecone.
* `chunk_size`, `chunk_overlap`: Ajustez les paramètres de division de texte pour optimiser les performances RAG.
* `model_name`: Changez le modèle de génération de texte.
* `template`: Modifiez l'invite pour adapter le comportement du modèle de langage.
* `search_kwargs`: Ajustez les paramètres de recherche de similarité dans Pinecone.

* Mistral AI pour le modèle de langage Mistral 7B Instruct v0.3.
* Pinecone pour la base de données vectorielle.
* Hugging Face pour les modèles et les outils de transformation.
* LangChain pour faciliter la création d'applications RAG.
