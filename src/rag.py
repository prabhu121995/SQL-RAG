import logging
import re

from langchain_classic.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a TelecomCo support analytics assistant.
Given a user question and the SQL query result from our tickets database,
provide a clear, concise natural language answer.
Be specific with numbers and facts from the data."""


def clean_sql(raw: str) -> str:
    """Strip markdown fences and any preamble, leaving only the SQL statement."""
    if not raw:
        return ""
    raw = re.sub(r"```(?:sql)?", "", raw).strip("`").strip()
    if "SQLQuery:" in raw:
        raw = raw.split("SQLQuery:")[-1].strip()
    # Drop trailing semicolons stripped later by run() is fine; keep them.
    return raw


class SQLRAGService:
    def __init__(self, db: SQLDatabase):
        self.db = db
        self.llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
            api_key=settings.groq_api_key,
        )
        self.sql_query_chain = create_sql_query_chain(self.llm, self.db)
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Question: {question}\nSQL Result: {result}\n\nAnswer:",
                ),
            ]
        )
        self.answer_chain = self.answer_prompt | self.llm

    def _generate_sql(self, question: str) -> str:
        raw = self.sql_query_chain.invoke({"question": question})
        sql = clean_sql(raw)
        if not sql:
            # Retry once with an explicit instruction
            raw = self.sql_query_chain.invoke({"question": question})
            sql = clean_sql(raw)
        if not sql:
            raise ValueError("Model returned empty SQL for the question")
        return sql

    def ask(self, question: str) -> dict:
        sql = self._generate_sql(question)
        logger.info("Generated SQL: %s", sql)

        try:
            result = self.db.run(sql)
        except Exception as e:
            logger.exception("SQL execution failed")
            raise ValueError(f"SQL execution failed: {e}") from e

        answer = self.answer_chain.invoke(
            {"question": question, "result": result}
        ).content

        return {"sql": sql, "result": str(result), "answer": answer}