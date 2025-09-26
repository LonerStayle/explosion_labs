from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate

schemas = [
    ResponseSchema(name="formula", description="질문 반응식"),
    ResponseSchema(name="result", description="생성된 물질 (생성되지 않을 수 있음)"),
    ResponseSchema(name="status", description="반응 상태(Good, Bad, Nothing)")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()
