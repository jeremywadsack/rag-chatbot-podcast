from dotenv import load_dotenv
import faiss
import feedparser
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore
import nltk
# import tiktoken
from unstructured.partition.html import partition_html


class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
    # cancel SGR codes if we don't write to a terminal
    if not __import__("sys").stdout.isatty():
        for _ in dir():
            if isinstance(_, str) and _[0] != "_":
                locals()[_] = ""
    else:
        # set Windows console in VT mode
        if __import__("platform").system() == "Windows":
            kernel32 = __import__("ctypes").windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            del kernel32


# Required by unstructured
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

load_dotenv()

podcast_atom_link = "https://api.substack.com/feed/podcast/1084089.rss"
parsed = feedparser.parse(podcast_atom_link)

episode = [ep for ep in parsed.entries if ep['title'] == "RAG Is A Hack - with Jerry Liu from LlamaIndex"][0]
episode_summary = episode['summary']
print(episode_summary[:100])


parsed_summary = partition_html(text=''.join(episode_summary)) 
start_of_transcript = [x.text for x in parsed_summary].index("Transcript") + 1
print(f"First line of the transcript: {start_of_transcript}")

documents = [Document(text=t.text) for t in parsed_summary[start_of_transcript:]]

d = 1536 # dimensions of text-ada-embeddings-002, the embedding model that we're going to use
faiss_index = faiss.IndexFlatL2(d)

Settings.llm = OpenAI(temperature=0.2, model="gpt-4")


# Cost estimation — 
#   0.65886 to put into a prompt of GPT
#   0.0021962 to embed
# e = tiktoken.encoding_for_model("gpt-4")
# total_tokens=sum([len(e.encode(x.text)) for x in parsed_summary[start_of_transcript:]])
# print(f"{total_tokens  * 0.03 / 1000} to put into a prompt of GPT")
# print(f"{total_tokens * 0.0001 / 1000} to embed")

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
chat_eng = index.as_chat_engine(similarity_top_k=10, chat_mode="context")

query = "What does Jerry think about RAG?"
print(f"{Colors.BOLD}{query}{Colors.END}")

response = chat_eng.chat(query)
print(f"{Colors.CYAN}{response.response}{Colors.END}")

for node in response.source_nodes:
    print(f"{Colors.GREEN}{node.get_score()} 👉 {node.text}{Colors.END}")

query_2 = "How does he think that it will evolve over time?"
print(f"{Colors.BOLD}{query_2}{Colors.END}")

response_2 = chat_eng.chat(query_2)
print(f"{Colors.CYAN}{response_2.response}{Colors.END}")

for node in response_2.source_nodes:
    print(f"{Colors.GREEN}{node.get_score()} 👉 {node.text}{Colors.END}")


