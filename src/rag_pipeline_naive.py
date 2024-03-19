# Reference : Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks by Patrick Lewis, Ethan Perez, Aleksandara Piktus et al.
import torch
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration, AdamW
from tqdm import tqdm 

# Define a simple synthetic dataset
dataset = [
    ("What is the capital of France?", "The capital of France is Paris.", ["Paris is the capital and most populous city of France.", "France is a country in Europe."]),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet.", ["Romeo and Juliet is a tragedy written by William Shakespeare early in his career.", "Shakespeare was an English playwright and poet."]),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system.", ["Jupiter is the fifth planet from the Sun and the largest in the Solar System.", "The solar system consists of the Sun and its planetary system."]),
]

# Initialize the BERT tokenizer and model for the retriever
retriever_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
retriever_model = BertModel.from_pretrained('bert-base-uncased')

# Initialize the BART tokenizer and model for the generator
generator_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
generator_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')



# Define the retriever function
def retrieve(query, knowledge_base, top_k=1):
    query_embedding = retriever_model(**retriever_tokenizer(query, return_tensors='pt')).pooler_output
    doc_embeddings = [retriever_model(**retriever_tokenizer(doc, return_tensors='pt')).pooler_output for doc in knowledge_base]
    
    scores = [torch.dot(query_embedding.squeeze(), doc_embedding.squeeze()) for doc_embedding in doc_embeddings]
    top_docs = [knowledge_base[i] for i in torch.topk(torch.tensor(scores), top_k).indices]
    return ' '.join(top_docs)

# Fine-tune the RAG model
retriever_model.train()
generator_model.train()
optimizer = AdamW(list(retriever_model.parameters()) + list(generator_model.parameters()), lr=1e-5)
for epoch in tqdm(range(3)):
    for query, answer, knowledge_base in dataset:
        retrieved_docs = retrieve(query, knowledge_base)
        input_ids = generator_tokenizer.encode(query + " " + retrieved_docs, return_tensors='pt')
        target_ids = generator_tokenizer.encode(answer, return_tensors='pt')
        
        outputs = generator_model(input_ids, labels=target_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Evaluate the RAG model
retriever_model.eval()
generator_model.eval()
for query, answer, knowledge_base in dataset:
    retrieved_docs = retrieve(query, knowledge_base)
    input_ids = generator_tokenizer.encode(query + " " + retrieved_docs, return_tensors='pt')
    output_ids = generator_model.generate(input_ids)
    response = generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("Query:", query)
    print("Retrieved documents:", retrieved_docs)
    print("Generated response:", response)
    print()

