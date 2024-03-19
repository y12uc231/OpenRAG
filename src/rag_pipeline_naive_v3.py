import torch
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration, AdamW
from tqdm import tqdm

dataset = [
    ("What is the capital of France?", "The capital of France is Paris.", ["Paris is the capital and most populous city of France.", "France is a country in Europe."]),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet.", ["Romeo and Juliet is a tragedy written by William Shakespeare early in his career.", "Shakespeare was an English playwright and poet."]),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system.", ["Jupiter is the fifth planet from the Sun and the largest in the Solar System.", "The solar system consists of the Sun and its planetary system."]),
]

retriever_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
retriever_model = BertModel.from_pretrained('bert-base-uncased')

generator_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
generator_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

optimizer = AdamW(list(retriever_model.parameters()) + list(generator_model.parameters()), lr=1e-5)

def retrieve(query, knowledge_base, top_k=1):
    query_embedding = retriever_model(**retriever_tokenizer(query, return_tensors='pt')).pooler_output
    doc_embeddings = [retriever_model(**retriever_tokenizer(doc, return_tensors='pt')).pooler_output for doc in knowledge_base]
    
    scores = [torch.dot(query_embedding.squeeze(), doc_embedding.squeeze()) for doc_embedding in doc_embeddings]
    top_indices = torch.topk(torch.tensor(scores), top_k).indices
    top_docs = [knowledge_base[i] for i in top_indices]
    
    return ' '.join(top_docs)

# Training Code. 
retriever_model.train()
generator_model.train()
num_epochs = 15
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    progress_bar = tqdm(dataset, desc="Training", unit="example")
    for query, answer, knowledge_base in progress_bar:
        retrieved_docs = retrieve(query, knowledge_base)
        
        input_text = query + ' ' + retrieved_docs
        input_ids = generator_tokenizer.encode(input_text, return_tensors='pt')
        target_ids = generator_tokenizer.encode(answer, return_tensors='pt')
        
        outputs = generator_model(input_ids, labels=target_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({"Loss": loss.item()})

# Inference code. 
retriever_model.eval()
generator_model.eval()
for query, answer, knowledge_base in dataset:
    retrieved_docs = retrieve(query, knowledge_base)
    input_text = query + ' ' + retrieved_docs
    input_ids = generator_tokenizer.encode(input_text, return_tensors='pt')
    
    output_ids = generator_model.generate(input_ids)
    response = generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("Query:", query)
    print("Retrieved documents:", retrieved_docs)
    print("Generated response:", response)
    print()