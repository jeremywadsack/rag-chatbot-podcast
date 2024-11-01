# RAG-based Chatbot project
A RAG chatbot with Llamaindex, FAISS, OpenAI
Tutorial: https://learnbybuilding.ai/tutorials/rag-chatbot-on-podcast-llamaindex-faiss-openai

# Dev Setup
You will need python and pip installed.

[Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Make sure the following dependencies are installed

```bash
 brew install libxml2 libxslt
```

Create and start the conda virtual environment and install the dependencies:

```bash
conda env create --file environment.yaml
conda activate rag-chatbot-podcast
```

You need to install a couple packages directly with `pip` because they are not available from Conda. :'(

```bash
pip install python-dotenv llama-index-vector-stores-faiss
```

Copy `.env.sample` to `.env` and add your [OpenAI API token](https://platform.openai.com/api-keys). 
(The `.env` file will not be checked into source control.)


# License
MIT