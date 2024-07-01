English | [简体中文](README_zh-CN.md)


# LLMFlow



## What's LLMFlow?

**LLMFlow** is a lightweight framework for building LLM applications. It offers an API interface similar to LangChain's
LCEL, but with a more straightforward and simple internal implementation. Users familiar with LangChain can quickly get
started with LLMFlow and extend or modify it as needed.

For example, LLMFlow implements the core functionality of LangChain's AgentExecutor
in less than 300 lines of [code](https://github.com/zhiguoxu/llmflow/blob/main/core/agents/agent.py). This is
particularly beneficial for users who want to understand the underlying implementation of agents and make further
optimizations.

LLMFlow is also very easy to use, as demonstrated by the following example of an Agent:

```python
from core.tool import tool
from core.llm.openai.openai_llm import OpenAILLM
from core.agents.agent import Agent

llm = OpenAILLM(model="gpt-4o")


@tool
def multiply(left: int, right: int) -> int:
    """multiply"""
    return left * right


agent = Agent(llm=llm, tools=[multiply])
agent.invoke("what's 1024*2024?")
```

## Why do we need LLMFlow?


Although LangChain provides a powerful framework for building LLM applications, its complex design also brings several
issues with deeper use.

LangChain's extensive use of abstraction results in highly complex code, making it difficult to understand and maintain.
While it initially simplifies the development process, as the complexity of requirements increases, these abstractions
gradually become obstacles to productivity. Additionally, these abstractions increase the learning and debugging burden,
forcing development teams to spend significant time dealing with internal framework code rather than focusing on
application functionality.

Especially in today's rapidly evolving AI and LLM landscape, where new concepts and methods emerge weekly, a framework
that is easy to understand and iterate on quickly is crucial!

To address these issues, LLMFlow adopts a more concise design with more refined code implementation, balancing
abstraction and simplicity by the use of necessary abstractions without being overly complex. Ultimately, LLMFlow
provides the same functionality and similar interfaces as LangChain but with significantly reduced code and fewer
abstract concepts. This makes it easy to understand and modify LLMFlow, allowing for quick experimentation or customized
development.

The project is still under development, with more features to be added. Your valuable feedback is welcome!
