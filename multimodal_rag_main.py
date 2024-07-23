import os
from dotenv import load_dotenv
from pdf_processor import process_pdf
from text_indexer import get_text_nodes
from image_indexer import index_images
from query_engine import MultimodalQueryEngine
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
import logging

# Load environment variables
load_dotenv()

# Setup models
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-4o-mini")
gpt_4v = OpenAIMultiModal(model="gpt-4o-mini", max_new_tokens=4096)

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Multimodal RAG System")

    # Step 1: Process PDF
    pdf_path = "PDFs/User_manual_Robin_600_LEDWash.pdf"
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return

    text_content, images = process_pdf(pdf_path)
    logger.info(f"Processed PDF: {len(text_content)} characters of text, {len(images)} images")

    # Step 2: Index text
    storage_dir = "./storage_nodes"
    try:
        if not os.path.exists(storage_dir):
            logger.info(f"Creating directory: {storage_dir}")
            os.makedirs(storage_dir)

        if not os.path.exists(os.path.join(storage_dir, "docstore.json")):
            logger.info("Creating new index...")
            text_nodes = get_text_nodes(text_content, images, max_tokens=4000, max_nodes=10)
            index = VectorStoreIndex(text_nodes, embed_model=embed_model)
            index.set_index_id("vector_index")
            logger.info(f"Saving index to {storage_dir}")
            index.storage_context.persist(storage_dir)
            logger.info("Index created and saved.")
        else:
            logger.info("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context, index_id="vector_index")
            logger.info("Index loaded.")
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        return

    # Step 3: Create query engine
    try:
        query_engine = MultimodalQueryEngine(
            retriever=index.as_retriever(similarity_top_k=9),
            multi_modal_llm=gpt_4v
        )
    except Exception as e:
        logger.error(f"Error creating query engine: {str(e)}")
        return

    # Step 4: Create agent
    try:
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="vector_tool",
            description="Useful for retrieving specific context from the data."
        )
        agent = FunctionCallingAgentWorker.from_tools(
            [vector_tool], llm=llm, verbose=True
        ).as_agent()
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        return

    # Step 5: Query processing
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        try:
            response = agent.query(query)
            #print(f"Agent response: {response}")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")

    logger.info("Exiting Multimodal RAG System")

if __name__ == "__main__":
    main()
