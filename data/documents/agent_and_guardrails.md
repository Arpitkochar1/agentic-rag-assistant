# Agents, Tool Routing and Guardrails

## Agentic routing
An agentic RAG system decides how to answer each question instead of always searching the same index. A router classifies the query into one of several routes: document retrieval for questions about the private knowledge base, web search for current events or facts the documents cannot know, a calculator tool for exact arithmetic, or a direct answer for small talk. LangGraph models this as a state graph, where nodes are steps such as guard, route, retrieve and generate, and conditional edges choose the next step based on the state.

If document retrieval returns nothing, a fallback edge can send the question to web search instead. The calculator tool should never use Python eval; the language model only converts the question into an expression and a whitelist based abstract syntax tree evaluator computes the result safely.

## Guardrails
Guardrails are checks that run before and after the language model. Input guardrails include length limits, prompt injection detection, PII redaction and content moderation. Output guardrails include removing hallucinated citation numbers, redacting PII, and a grounding check that verifies the answer is supported by the retrieved context.

Indirect prompt injection is a special risk in RAG: a retrieved document or web page may contain hidden instructions such as "ignore previous instructions". Retrieved text must therefore be treated as untrusted data, sanitised, and clearly delimited in the prompt. An LLM-as-judge guardrail adds extra latency and cost, so it is usually optional.

## SOLID principles in practice
The Single Responsibility Principle keeps loaders, chunkers, retrievers and guardrails in separate classes. The Open/Closed Principle lets a new retrieval strategy or LLM provider be added by registering a new implementation instead of editing existing code. The Liskov Substitution Principle means any retriever can replace another. Interface Segregation favours small protocols such as Embedder and Reranker. Dependency Inversion means the agent depends on abstractions, and a single composition root wires up the concrete classes.

## Streaming and deployment
A FastAPI backend can stream tokens to the client using Server-Sent Events. The service is packaged with Docker and can be deployed to platforms such as Render, Railway or Hugging Face Spaces. A Streamlit front end can run the agent in-process and be deployed on Streamlit Community Cloud by providing the API key through the app's secrets.
