[English](README.md) | 简体中文 | [日本語](README_ja-JP.md)


# FlowX


## FlowX 是什么?


**FlowX** 是一个用于构建大语言模型（LLM）应用的轻量级框架。它提供了与 LangChain 类似的 API 接口（LCEL），但内部实现更加简洁直观。熟悉 LangChain 的用户可以轻松上手，并根据需求进行快速扩展和改造。

例如，FlowX 用不到4四百行的[代码](https://github.com/zhiguoxu/FlowX/blob/main/core/agents/agent.py)实现了 LangChain 的
AgentExecutor 的核心功能，这对于一个想了解 Agent 底层实现，并要做进一步优化的用户来说，是一个极大的福利。

FlowX 的使用也非常简单，以下是一个 Agent 的示例：

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

## 为什么需要 FlowX?


尽管 LangChain 提供了一个非常强大的 LLM 应用构建框架，但随着用户的深入使用，其复杂的设计也带来了许多问题。

LangChain 过多的抽象层次导致代码复杂且难以理解和维护。尽管这些抽象在初期简化了开发过程，但随着需求复杂性的提升，他们逐渐成为生产力的障碍。此外，这种复杂的抽象增加了学习和调试的负担，使开发团队不得不花费大量时间处理框架内部问题，而非专注于应用功能的开发。

在当下 AI 和 LLM 领域，技术变化迅速，每周都有新的概念和方法涌现。因此，一个易于理解、便于快速迭代的 LLM 框架尤为重要。

针对这个问题，FlowX 采用更加简洁的设计方案，配合更加精炼的代码实现，在抽象和简单之间找到了平衡点，即：使用必要的抽象，但又不至于过分复杂。最终，FlowX 提供了和 LangChain 相同的功能和类似的接口，但大大减少了代码量和抽象概念的数量。因此你可以非常容易理解和修改 FlowX，并快速进行实验或定制化开发。

目前项目仍在开发中，还有更多功能等待添加，欢迎提出宝贵意见！
