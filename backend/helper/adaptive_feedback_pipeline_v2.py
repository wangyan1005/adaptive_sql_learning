import pickle
from openai import OpenAI
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import gc
from threading import Lock

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

DEVICE = "cpu"

# Globals 
_tokenizer = None
_model = None
_example_meta = None
_example_emb = None
_example_emb_norm = None
_loaded = False
_model_lock = Lock()

client = OpenAI()

# Path
BASE_DIR = os.path.dirname(__file__)
DB_DIR = os.path.join(BASE_DIR, "db")
META_PATH = os.path.join(DB_DIR, "example_meta_codebert.pkl")
EMB_PATH = os.path.join(DB_DIR, "example_embeddings_codebert.pkl")

# # initialize CodeBERT model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
# model.eval()

# with open("helper/db/example_meta_codebert.pkl", "rb") as f:
#     meta = pickle.load(f)

# with open("helper/db/example_embeddings_codebert.pkl", "rb") as f:
#     example_embeddings = pickle.load(f)    # numpy array [N, 768]

# # convert to torch tensor and normalize for cosine similarity
# example_embeddings = torch.tensor(example_embeddings, dtype=torch.float32)

client = OpenAI()

def _load_once():
    global _loaded, _tokenizer, _model, _example_meta, _example_emb, _example_emb_norm
    if _loaded:
        return

    with _model_lock:
        if _loaded:
            return

    _tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    _model = AutoModel.from_pretrained("microsoft/codebert-base", low_cpu_mem_usage=True).to(DEVICE)
    _model.eval()

    with open(META_PATH, "rb") as f:
        _example_meta = pickle.load(f)

    with open(EMB_PATH, "rb") as f:
        emb_np = pickle.load(f)

    _example_emb = torch.tensor(emb_np, dtype=torch.float32)  # [N, 768]
    _example_emb_norm = F.normalize(_example_emb, dim=1).to(DEVICE) 

    del emb_np
    gc.collect()

    _loaded = True
    print(f"[adaptive] Loaded {len(_example_meta)} examples | shape={_example_emb.shape}")

@torch.no_grad()
def _embed(text: str) -> torch.Tensor:
    """Return normalized CodeBERT CLS embedding (1 x 768)."""
    if not _loaded or _tokenizer is None:
        return torch.empty((1, 768), dtype=torch.float32).to(DEVICE)
    
    tokens = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(DEVICE)

    outputs = _model(**tokens)
    cls = outputs.last_hidden_state[:, 0, :]  # [1, 768]
    cls = F.normalize(cls, dim=1)

    del outputs, tokens
    gc.collect()
    return cls   

# # generate embedding function
# def get_embedding(text: str):
#     """CodeBERT embedding"""
#     tokens = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=256
#     )
    
#     with torch.no_grad():
#         outputs = model(**tokens)
#         cls_embedding = outputs.last_hidden_state[0, 0]
#         cls_embedding = torch.nn.functional.normalize(cls_embedding, dim=0)

#     emb = cls_embedding.numpy().astype("float32")

#     del outputs, cls_embedding
#     gc.collect()
#     return emb

def retrieve_similar_examples(query: str, k=2):
    _load_once()

    emb = _embed(query).squeeze() # [768]
    
    scores = torch.matmul(_example_emb_norm, emb) 

    top_idx = torch.topk(scores, k=min(k, scores.size(0))).indices.tolist()
    
    results = [_example_meta[i] for i in top_idx]

    del emb, scores
    gc.collect()
    return results


def build_prompt(query, examples, user_profile):
    SCHEMA = """
    ### Database Schema
        Employees(
            Employee_ID INT PK, 
            Name TEXT, 
            Job_Role TEXT, 
            Division TEXT, 
            Last_Login_Time DATETIME
        )

        Robots(
            Robot_ID INT PK, 
            Model TEXT, 
            Manufacturing_Date DATETIME, 
            Status TEXT, 
            Last_Software_Update DATETIME, 
            Employee_ID INT FK
        )

        Logs(
            Log_ID INT PK, 
            Employee_ID INT FK, 
            Action_Description TEXT, 
            Timestamp DATETIME, 
            Robot_ID INT FK
        )

        Incidents(
            Incident_ID INT PK, 
            Description TEXT, 
            Timestamp DATETIME, 
            Robot_ID INT FK, 
            Employee_ID INT FK
        )

        Access_Codes(
            Access_Code_ID INT PK, 
            Employee_ID INT FK, 
            Level_of_Access TEXT, 
            Timestamp_of_Last_Use DATETIME  
        )
    """
    
    TAXONOMY = """
    Error Types:
    1. Syntax Error
        - misspelling
        - missing quotes
        - missing commas
        - missing semicolons
        - non-standard operators
        - unmatched brackets
        - data type mismatch
        - incorrect wildcard usage
        - incomplete query
        - incorrect SELECT usage
        - incorrect DISTINCT usage
        - wrong positioning
        - aggregation misuse
    
    2. Schema Error
        - undefined table
        - undefined column
        - undefined function

    3. Logic Error
        - ambiguous reference
        - incorrect GROUP BY usage
        - incorrect HAVING clause
        - incorrect JOIN usage
        - incorrect ORDER BY usage
        - operator misuse
    
    4. Construction Error
        - inefficient query
    """

    USER_BEHAVIOR_RULES = """
    ### User Behavior Interpretation Rules 

    Typing Speed:
    - <= 2.48 keys/s → slow and careful 
    - 2.48–2.93 keys/s → normal pace 
    - > 2.93 keys/s → fast and energetic 

    Dwell Time:
    - <= 99.30 ms → quick decisive keypresses
    - 90.30–110.35 ms → normal dwell time 
    - > 110.35 ms → thoughtful, cautious pressing 

    Flight Time:
    - <= 237.08 ms → very fast transitions 
    - 237.08–296.72 ms → normal transitions 
    - > 296.72 ms → longer pauses, possible uncertainty

    Correction Rate (backspace/delete combined):
    - < 0.015 → very low correction behavior
    - 0.015–0.04 → normal corrections
    - > 0.04 → high correction behavior

    Retry Count
    - <= 1 → low retry 
    - 2 → moderate retry 
    - >= 3 → high retry 

    Use these thresholds to classify the user's behavior and reflect it accurately in the feedback.
    """

    SQL_SEMICOLON_RULE = """
    ### Semicolon Rules
    - Single SQL statements typically don't require semicolons in most environments
    - Mark Error type as "Syntax" and Error subtype as "missing semicolons" if it ends without a semicolon
    """

    shots = "\n\n".join([
        f"### Example {i+1}\n"
        f"SQL Query: {ex['query']}\n"
        f"Error Type: {ex['error_type']}\n"
        f"Error Subtype: {ex['error_subtype']}\n"
        f"Feedback: {ex['feedback']}\n"
        for i, ex in enumerate(examples)
    ])

    user_text = f"""
### User Profile
Typing Speed: {user_profile['typing_speed']} keys/sec
Flight Time: {user_profile['avg_flight_time']} ms
Dwell Time: {user_profile['avg_dwell_time']} ms
Backspace Rate: {user_profile['backspace_rate']}
Delete Rate: {user_profile['delete_rate']}
Retry Count: {user_profile['retry_count']}
"""

    return f"""
You are an intelligent SQL debugging tutor.

{SCHEMA}
{TAXONOMY}
{USER_BEHAVIOR_RULES}
{SQL_SEMICOLON_RULE}
{user_text}

### Task
Given the new SQL query, analyze it using the style shown in the examples.
Return **strict JSON** in the following format:

{{
  "error_type": "",
  "error_subtype": "",
  "personalized_feedback": ""
}}

### Feedback Guidelines
Your feedback **must** follow this 3-part structure:
1. **Behavior Observation**  
   - Based ONLY on user metrics (typing speed, dwell time, etc.)  
   - MUST explicitly reflect LOW/MODERATE/HIGH behavior categories  

2. **Technical Hint**  
   - Relate directly to the identified SQL error  
   - DO NOT reveal the full answer  

3. **Encouragement**  
   - Tone adapts to behavior  
   - Must end on a supportive, reassuring note  

### Examples of good feedback:
- Fast typer with syntax error: "Working quickly! The issue is in your WHERE clause - check the column name. Your speed is great, just verify details."
- Slow typer with logic error: "Taking your time to think through this. The GROUP BY needs all non-aggregated columns. Your careful approach will master this!"
- High retry with schema error: "I see you're experimenting with different approaches. The table name needs to match exactly. Persistence like yours leads to expertise!"

### Technical hint: Must be specific to the error but not reveal the answer

### Length: 2-3 sentences total

### Few-Shot Examples
{shots}

### New SQL Query
{query}

### JSON Response:
"""

def generate_sql_feedback(query, user_profile):
    _load_once() 
    examples = retrieve_similar_examples(query, k=2)
    prompt = build_prompt(query, examples, user_profile)

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return only valid JSON"},
                {"role": "user", "content": prompt}
            ],
        temperature=0.5
    )

    content = response.choices[0].message.content
    return content

    