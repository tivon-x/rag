from langchain_core.output_parsers import BaseOutputParser

class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # 去除空行
    
line_list_output_parser = LineListOutputParser()