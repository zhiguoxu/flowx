[English](README.md) | [简体中文](README_zh-CN.md) | 日本語

# FlowX

## FlowXとは？

**FlowX** は、LLMアプリケーションを構築するための軽量フレームワークです。LangChainのLCELに似たAPIインターフェースを提供しますが、内部実装はよりシンプルで直感的です。LangChainに慣れたユーザーは、FlowXをすぐに使い始め、必要に応じて拡張や修正を行うことができます。

例えば、FlowXはLangChainのAgentExecutorのコア機能を400行の[コード](https://github.com/zhiguoxu/FlowX/blob/main/core/agents/agent.py)で実装しています。これは、エージェントの基礎的な実装を理解し、さらなる最適化を行いたいユーザーにとって非常に有益です。

FlowXの使用も非常に簡単で、以下はエージェントの例です：

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
agent.invoke("1024*2024は何ですか？")
```

## なぜFlowXが必要なのか？

LangChainはLLMアプリケーションを構築するための強力なフレームワークを提供していますが、その複雑な設計は使用が進むにつれていくつかの問題を引き起こします。

LangChainの広範な抽象化は、非常に複雑なコードを生み出し、理解と保守が難しくなります。初期段階では開発プロセスを簡素化しますが、要件の複雑さが増すにつれて、これらの抽象化は生産性の障害となります。さらに、これらの抽象化は学習とデバッグの負担を増加させ、開発チームはアプリケーションの機能に集中するのではなく、フレームワークの内部コードに多くの時間を費やすことを余儀なくされます。

特に、毎週新しい概念や方法が出現する今日の急速に進化するAIおよびLLMの分野では、理解しやすく、迅速に反復できるフレームワークが重要です。

これらの問題に対処するために、FlowXはより簡潔な設計とより洗練されたコード実装を採用し、必要な抽象化を使用しつつも過度に複雑にならないようにバランスを取っています。最終的に、FlowXはLangChainと同じ機能と類似のインターフェースを提供しながら、コード量と抽象概念の数を大幅に削減しています。これにより、FlowXを非常に理解しやすく、修正しやすくし、迅速な実験やカスタマイズ開発が可能になります。

プロジェクトはまだ開発中であり、追加される機能がたくさんあります。貴重なフィードバックをお待ちしています！
