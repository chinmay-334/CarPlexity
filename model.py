import os
import re
import joblib
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any, List

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureChatOpenAI
from langchain.memory.buffer import ConversationBufferMemory
from langgraph.graph import StateGraph


# ---------------------------
# State Definition
# ---------------------------
class GraphRAGState(Dict[str, Any]):
    question: str
    decision: str
    cypher_query: str
    results: List[Dict[str, Any]]
    final_answer: str
    memory_context: str
    previous_answer: str


# ---------------------------
# Main RAG Pipeline
# ---------------------------
class GraphRAGPipeline:
    def __init__(self, **kwargs):
        load_dotenv("content.env")
        load_dotenv("neo4j.env")

        self.openai_api_key = kwargs.get("openai_api_key") or os.getenv("openai_api_key")
        self.openai_api_endpoint = kwargs.get("openai_api_endpoint") or os.getenv("openai_api_base")
        self.openai_api_version = kwargs.get("openai_api_version") or os.getenv("openai_api_version")
        self.deployment_name = kwargs.get("deployment_name") or os.getenv("deployment_name")

        self.neo4j_uri = kwargs.get("neo4j_uri") or os.getenv("uri")
        self.neo4j_username = kwargs.get("neo4j_username") or os.getenv("username")
        self.neo4j_password = kwargs.get("neo4j_password") or os.getenv("password")

        self.top_k = kwargs.get("top_k", 10)

        # Validate config
        if not all([self.openai_api_key, self.openai_api_endpoint, self.deployment_name]):
            raise ValueError("Missing OpenAI/Azure configuration.")
        if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password]):
            raise ValueError("Missing Neo4j configuration.")

        # Memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize components
        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
        )

        self.llm_client = AzureChatOpenAI(
            api_key=self.openai_api_key,
            azure_endpoint=self.openai_api_endpoint,
            azure_deployment=self.deployment_name,
            api_version=self.openai_api_version,
            temperature=0.0,
        )

        # Build workflow
        self.workflow = self._build_workflow()


    # ---------------------------
    # Build Workflow
    # ---------------------------
    def _build_workflow(self):
        workflow = StateGraph(GraphRAGState)

        workflow.add_node("decide", self._decide)
        workflow.add_node("memory", self._memory)
        workflow.add_node("generate_cypher", self._generate_cypher)
        workflow.add_node("query_graph", self._query_graph)
        workflow.add_node("reasoning", self._reasoning)

        workflow.set_entry_point("decide")
        workflow.add_edge("decide", "memory")
        workflow.add_edge("memory", "generate_cypher")
        workflow.add_edge("generate_cypher", "query_graph")
        workflow.add_edge("query_graph", "reasoning")

        return workflow.compile()


    # ---------------------------
    # Node: Decide Relevance
    # ---------------------------
    async def _decide(self, state: GraphRAGState):
        q = state["question"]
        mem = state.get("memory_context", "")

        prompt = f"Is this question about cars? Reply one word: relevant/irrelevant/greeting.\nQ: {q}\nMemory: {mem}"

        try:
            ans = (await self.llm_client.ainvoke(prompt)).content.lower().strip()
            if "irrelevant" in ans:
                state["decision"] = "irrelevant"
            elif "greeting" in ans:
                state["decision"] = "greeting"
            else:
                state["decision"] = "cypher"
        except:
            state["decision"] = "reasoning"

        return state


    # ---------------------------
    # Node: Load Memory + previous answer
    # ---------------------------
    async def _memory(self, state: GraphRAGState):

        mem_vars = self.memory.load_memory_variables({})
        history = mem_vars.get("chat_history", [])

        # Compress last ~6 messages
        if isinstance(history, list):
            state["memory_context"] = "\n".join(f"{m.type}: {m.content}" for m in history[-6:])
        else:
            state["memory_context"] = history

        # Extract last AI answer
        prev = ""
        if isinstance(history, list):
            for m in reversed(history):
                if m.type == "ai":
                    prev = m.content
                    break
        state["previous_answer"] = prev

        return state


    # ---------------------------
    # Node: Generate Cypher Query       
    # ---------------------------
    async def _generate_cypher(self, state: GraphRAGState):

        if state["decision"] != "cypher":
            return state

        q = state["question"]
        mem = state["memory_context"]

        try:
            with open("rulebook.txt", "r", encoding="utf-8") as f:
                template = f.read()
        except:
            template = (
                "Generate a Cypher query for the following question.\n"
                "Schema: {schema}\nMemory: {memory}\nQuestion: {question}"
            )

        prompt = PromptTemplate(
            input_variables=["schema", "question", "memory"],
            template=template,
        )

        schema = self.graph.get_schema
        formatted = prompt.format(schema=schema, question=q, memory=mem)

        try:
            raw = (await self.llm_client.ainvoke(formatted)).content
        except:
            state["decision"] = "reasoning"
            return state

        # Cleanup query
        match = re.search(r"```(?:cypher)?([\s\S]*?)```", raw, re.IGNORECASE)
        if match:
            state["cypher_query"] = match.group(1).strip()
        else:
            match = re.search(r"(MATCH[\s\S]*)", raw, re.IGNORECASE)
            state["cypher_query"] = match.group(1).strip() if match else raw.strip()

        return state


    # ---------------------------
    # Node: Query Neo4j
    # ---------------------------
    async def _query_graph(self, state: GraphRAGState):
        if state["decision"] != "cypher":
            state["results"] = []
            return state

        cy = state.get("cypher_query")
        if not cy:
            state["results"] = []
            return state

        try:
            data = self.graph.query(cy)
            state["results"] = data[: self.top_k] if isinstance(data, list) else []
        except:
            state["results"] = []

        return state


    # ---------------------------
    # Node: Final Reasoning
    # ---------------------------
    async def _reasoning(self, state: GraphRAGState):

        q = state["question"]
        results = state["results"]
        mem = state["memory_context"]
        prev = state["previous_answer"]
        decision = state["decision"]

        # Explanation included for non-cypher paths
        explanation = f"\nBrief explanation of previous answer: {prev}\n" if prev and decision != "cypher" else ""

        if decision == "greeting":
            prompt = f"Greet user briefly with car context. {explanation} Q: {q}"

        elif decision == "irrelevant":
            prompt = f"Not car-related. Politely redirect to car topics. {explanation} Q: {q}"

        else:  # relevant or fallback
            if results:
                prompt = f"""
Use ONLY retrieved data. Don't add any other data from outside show only what you have honestly. 

{explanation}
Question: {q}
Retrieved: {results}
Memory: {mem}
"""
            else:
                prompt = f"""
No matching data. {explanation}
Say you lack sufficient data without naming cars.
Question: {q}
Memory: {mem}
"""

        try:
            out = await self.llm_client.ainvoke(prompt)
            state["final_answer"] = out.content.strip()
        except:
            state["final_answer"] = "Could not generate a proper response."

        # update memory
        self.memory.chat_memory.add_user_message(q)
        self.memory.chat_memory.add_ai_message(state["final_answer"])

        return state


    # ---------------------------
    # Public API
    # ---------------------------
    async def run(self, question: str):
        initial = {
            "question": question,
            "decision": "",
            "cypher_query": "",
            "results": [],
            "memory_context": "",
            "previous_answer": "",
        }
        return await self.workflow.ainvoke(initial)

    def answer(self, question: str) -> Dict[str, Any]:
        final = asyncio.run(self.run(question))
        return {
            "answer": final.get("final_answer", ""),
            "intent": final.get("decision", ""),
            "used_context": bool(final.get("results")),
            "cypher_query": final.get("cypher_query", ""),
            "results_count": len(final.get("results", [])),
        }


    # ---------------------------
    # Save/Load config
    # ---------------------------
    def save(self, path="GraphRagConfig.joblib"):
        joblib.dump(
            {
                "openai_api_key": self.openai_api_key,
                "openai_api_endpoint": self.openai_api_endpoint,
                "openai_api_version": self.openai_api_version,
                "deployment_name": self.deployment_name,
                "neo4j_uri": self.neo4j_uri,
                "neo4j_username": self.neo4j_username,
                "neo4j_password": self.neo4j_password,
                "top_k": self.top_k,
            },
            path,
        )

    @classmethod
    def load(cls, path="GraphRagConfig.joblib"):
        return cls(**joblib.load(path))


# ---------------------------
# Example
# ---------------------------
if __name__ == "__main__":
    rag = GraphRAGPipeline.load()
    print(rag.answer("Which car has the best engine performance?"))
    print(rag.answer("And what about its warranty?"))
    print(rag.answer("Tell me about yourself?"))
    print(rag.answer("Tell me a joke"))
