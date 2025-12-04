import gradio as gr
from huggingface_hub import hf_hub_download
import subprocess
import sys, platform
from importlib import metadata as md
from rag_db import * 

subprocess.run("pip install -V llama_cpp_python==0.3.15", shell=True)
from llama_cpp import Llama
# Download your GGUF from HF Hub
model_path = hf_hub_download(
    repo_id="StefanCoder1/Qwen-tunded-Q4_K_M-GGUF",
    filename="qwen-tunded-q4_k_m.gguf",
    # token=True,  # uncomment + set HF_TOKEN in Space secrets if repo is private
)

db = init_vectorstore()
retriever = db.as_retriever(search_kwargs={"k": 1}) # how much to retrive




# Create llama.cpp LLM instance
llm = Llama(
    model_path=model_path,
    n_ctx=2048, #org 4096
    n_threads=2,   # org 4
    n_batch=96,        # ny
    use_mmap=True,     # ny
    use_mlock=False,    #ny
)

def respond(message, history):
    # 1. Retrieve Context
    context = ask(message, retriver_moedel=retriever)
    print("----------------------    CONTEXT FOUND   --------------------",context)
    
    # 2. Define System/Contextual Prompt
    system_instruction = (
        "You are an expert on mythology and fantasy creatures. "
        "Use the provided CONTEXT to answer the USER's question accurately. "
        "If the CONTEXT is irrelvant to the question, do not mention it."
        "If the CONTEXT does not contain the answer, state that you don't know "
        "based on the available information."
        "Please use **MARKDOWN** when formatting the answer"
        "Do not respond longer than 3-4 sentances unless specifically asked."
        
    )

    messages = []

    # System message including the RAG context
    system_content = (
        f"{system_instruction}\n\n"
        f"CONTEXT:\n---\n{context}\n---"
    )
    messages.append(("system", system_content))

    # Conversation history
    for user_msg, assistant_msg in (history or []):
        messages.append(("user", user_msg))
        messages.append(("assistant", assistant_msg))

    # Final user turn
    messages.append(("user", message))

    # Convert messages -> Qwen-2.5 chat template
    prompt_parts = []
    for role, content in messages:
        prompt_parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
    # Last assistant tag is where the model should start generating
    prompt_parts.append("<|im_start|>assistant\n")
    prompt = "\n".join(prompt_parts)

    output = llm(
        prompt,
        max_tokens=200,   # kortare = snabbare
        temperature=0.5,
        stop=[
            "<|im_end|>",         # slut på assistant-svar
            "<|im_start|>user",   # nästa användarturn
            "<|im_start|>system",
        ],
    )

    reply = output["choices"][0]["text"].strip()
    return reply
## if we want LLama model instead: 
    """
    
    # 3. Start building the prompt with the system instruction and RAG context
    prompt = f"System Instruction: {system_instruction}\n\n"
    prompt += f"CONTEXT:\n---\n{context}\n---\n\n"
    
    # 4. Add Conversation History
    prompt += "CONVERSATION HISTORY:\n"
    for user_msg, assistant_msg in (history or []):
        # Use clear labels for history
        prompt += f"User: {user_msg} \n Assistant: {assistant_msg}\n"
    
    # 5. Add the final turn
    prompt += f"User: {message} \nAssistant:"

    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        stop=["User:", "Assistant:", "CONVERSATION HISTORY:", "CONTEXT:"],
    )

    reply = output["choices"][0]["text"].strip()
    return reply"""

dark_academia_css = """
body {
    background: radial-gradient(circle at top, #151521 0, #050509 55%, #000000 100%);
    color: #e0d9c6;
    font-family: "Georgia", "Times New Roman", serif;
}

/* Optional: classical-looking font for headings */
@import url('https://fonts.googleapis.com/css2?family=Cardo:wght@400;700&display=swap');

h1, h2, h3, .prose h1, .prose h2, .prose h3 {
    font-family: "Cardo", "Georgia", serif;
    color: #f5f1e6 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Main container */
.gradio-container {
    background: transparent !important;
}

/* Hero section with pillars / image */
.hero {
    position: relative;
    margin: 0 auto 1.5rem auto;
    max-width: 900px;
    padding: 1.8rem 1.6rem 1.6rem 1.6rem;
    border-radius: 22px;
    border: 1px solid rgba(196, 164, 110, 0.5);
    background:
        radial-gradient(circle at top, rgba(196,164,110,0.12), rgba(5,5,10,0.96)),
        linear-gradient(135deg, rgba(10,10,18,0.9), rgba(3,3,7,0.98));
    box-shadow:
        0 0 30px rgba(0, 0, 0, 0.9),
        0 0 80px rgba(84, 63, 140, 0.55);
    overflow: hidden;
}



/* Inner hero content so text doesn’t sit under pillars */
.hero-inner {
    position: relative;
    z-index: 1;
    display: flex;
    gap: 1.2rem;
    align-items: center;
}

/* Circle “medallion” with a statue/Greek image */
.hero-image {
    flex-shrink: 0;
    width: 96px;
    height: 96px;
    border-radius: 999px;
    overflow: hidden;
    border: 2px solid rgba(196, 164, 110, 0.7);
    box-shadow: 0 0 24px rgba(0, 0, 0, 0.9);
    background: radial-gradient(circle at top, rgba(255,255,255,0.1), rgba(5,5,10,1));
}

.hero-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* Hero title + subtitle */
.hero-text h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.3rem;
    letter-spacing: 0.12em;
}

.hero-text p {
    margin: 0;
    font-size: 0.92rem;
    color: #e0d9c6;
    line-height: 1.5;
}

/* Subtle Greek key (meander) line under hero */
.hero-keyline {
    margin-top: 0.9rem;
    height: 1px;
    background-image: linear-gradient(
        90deg,
        rgba(196,164,110,0),
        rgba(196,164,110,0.8),
        rgba(196,164,110,0)
    );
    opacity: 0.7;
}

/* Chat box card */
.gr-chat-interface {
    position: relative !important;
    background:
        linear-gradient(rgba(5, 5, 10, 0.90), rgba(5, 5, 12, 0.95)), /* dark overlay */
        url("https://i0.wp.com/www.bookofthrees.com/wp-content/uploads/2005/02/parthenon.jpg?w=1600&ssl=1"); /* your Parthenon image */
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    border-radius: 18px !important;
    border: 1px solid rgba(196, 164, 110, 0.2);
    box-shadow:
        0 0 25px rgba(0, 0, 0, 0.9),
        0 0 60px rgba(84, 63, 140, 0.4);
    overflow: hidden;
}

.gr-chat-interface::before {
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.45); /* additional dark veil */
    backdrop-filter: blur(2px);      /* slight blur makes it elegant */
    pointer-events: none;
    z-index: 0;
}


/* Chat messages */
.gr-chat-message {
    position: relative;
    z-index: 1; 
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(6px);
}

.gr-chat-message.user {
    background: radial-gradient(circle at top left,
        rgba(196, 164, 110, 0.16),
        rgba(15, 15, 25, 0.95)
    ) !important;
    border-left: 3px solid #c4a46e !important;
}

.gr-chat-message.bot {
    background: radial-gradient(circle at top left,
        rgba(127, 90, 240, 0.18),
        rgba(8, 8, 18, 0.96)
    ) !important;
    border-left: 3px solid #7f5af0 !important;
}

/* Input area */
textarea, .gr-text-input, .gr-textbox {
    background: rgba(10, 10, 18, 0.95) !important;
    border-radius: 999px !important;
    border: 1px solid rgba(196, 164, 110, 0.5) !important;
    color: #f5f1e6 !important;
}

/* Buttons */
button, .gr-button {
    background: linear-gradient(135deg, #7f5af0, #c4a46e) !important;
    border-radius: 999px !important;
    border: none !important;
    color: #fdfaf0 !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    box-shadow: 0 0 18px rgba(0, 0, 0, 0.8);
}

button:hover, .gr-button:hover {
    filter: brightness(1.06);
    box-shadow:
        0 0 18px rgba(127, 90, 240, 0.7),
        0 0 30px rgba(196, 164, 110, 0.6);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(196, 164, 110, 0.6);
    border-radius: 999px;
}
"""


with gr.Blocks(css=dark_academia_css) as demo:
    gr.HTML(
        """
        <div class="hero">
          <div class="hero-inner">
            <div class="hero-image">
              <img src="https://preview.redd.it/ancient-greek-marble-statue-of-in-a-toga-v0-ajlvj2xddwec1.jpg?width=640&crop=smart&auto=webp&s=9bf7b1439ea460e824e188c63633fe4bda87056f" alt="Homer">
            </div>
            <div class="hero-text">
              <h1>Homer LLM</h1>
              <p>
                I am well read on Homer's most famous works – The Iliad and The Odyssey.
                Ask me of heroes, gods, and monsters.
              </p>
            </div>
          </div>
          <div class="hero-keyline"></div>
        </div>
        """
    )
    chat = gr.ChatInterface(
        fn=respond,
        title="Expert of the Iliad and the Odyssey",
        description=(
            "I have direct access to The Odyssey and The Illiad, so detailed questions are encouraged."
            "Ask me about these, or other mythological questions you may have. "
            "I am older than the trees, and the goblins that live inside them, "
            "and even the ghouls that haunt them. Tread lightly."
        ),
        css=dark_academia_css,
    )

if __name__ == "__main__":
    demo.launch()
