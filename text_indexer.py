from llama_index.core.schema import TextNode
from typing import List
from PIL import Image
import tiktoken

def get_text_nodes(text_content: str, images: List[Image.Image], max_tokens: int = 2000, max_nodes: int = 10) -> List[TextNode]:
    print("Creating text nodes")
    nodes = []
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Split text into sentences
    sentences = text_content.split(". ")
    
    current_chunk = ""
    current_tokens = 0
    
    for idx, sentence in enumerate(sentences):
        sentence_tokens = len(enc.encode(sentence))
        
        if current_tokens + sentence_tokens > max_tokens or len(nodes) >= max_nodes:
            # Create a node with the current accumulated text
            metadata = {"chunk_id": len(nodes)}
            if len(nodes) < len(images):
                image_path = f"data_images/image_{len(nodes)}.jpg"
                images[len(nodes)].save(image_path)
                metadata["image_path"] = image_path
            
            node = TextNode(text=current_chunk.strip(), metadata=metadata)
            nodes.append(node)
            
            if len(nodes) >= max_nodes:
                break
            
            # Reset for the next node
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += ". " + sentence if current_chunk else sentence
            current_tokens += sentence_tokens
    
    # Add the last chunk if there's any left and we haven't reached max_nodes
    if current_chunk and len(nodes) < max_nodes:
        metadata = {"chunk_id": len(nodes)}
        if len(nodes) < len(images):
            image_path = f"data_images/image_{len(nodes)}.jpg"
            images[len(nodes)].save(image_path)
            metadata["image_path"] = image_path
        
        node = TextNode(text=current_chunk.strip(), metadata=metadata)
        nodes.append(node)
    
    print(f"Created {len(nodes)} text nodes")
    return nodes