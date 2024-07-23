from typing import Any, List, Optional
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode, QueryBundle
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response

QA_PROMPT_TMPL = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: """

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

class MultimodalQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        retriever: BaseRetriever,
        multi_modal_llm: OpenAIMultiModal,
        qa_prompt: PromptTemplate = QA_PROMPT
    ):
        self.retriever = retriever
        self.multi_modal_llm = multi_modal_llm
        self.qa_prompt = qa_prompt

    def _query(self, query_bundle: QueryBundle) -> Response:
        query_str = query_bundle.query_str
        nodes = self.retriever.retrieve(query_str)
        image_nodes = [
            NodeWithScore(node=ImageNode(image_path=n.metadata["image_path"]))
            for n in nodes
            if "image_path" in n.metadata
        ]

        context_str = "\n\n".join([n.get_content(metadata_mode=MetadataMode.LLM) for n in nodes])
        fmt_prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the query."},
            {"role": "user", "content": fmt_prompt}
        ]

        llm_response = self.multi_modal_llm.chat(
            messages=messages,
            image_documents=[image_node.node for image_node in image_nodes],
        )
        return Response(response=str(llm_response.content), source_nodes=nodes)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._query(query_bundle)

    def _get_prompt_modules(self) -> List[Any]:
        return [self.qa_prompt]