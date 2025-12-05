
import subprocess
import sys, platform
from importlib import metadata as md
from rag_db import * 
# Download your GGUF from HF Hub


db = init_vectorstore()
retriever = db.as_retriever(search_kwargs={"k": 1}) # how much to retrive
