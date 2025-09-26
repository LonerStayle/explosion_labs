from langchain.output_parsers import PydanticOutputParser
from app.dto.CombinationRes import CombinationRes

combination_parser = PydanticOutputParser(pydantic_object=CombinationRes)