# Indexer Mapper ======================================
from indexing.chunker import RecursiveChunker, SemanticNLTKChunker, SemanticSpacyChunker, TokenChunker
from indexing.data_processor import PdfProcessor


LOADER_MAPPING = {
    ".pdf": (PdfProcessor, {}),
}

CHUNER_MAPPING = {
    "recursive": (RecursiveChunker, {}),
    "token": (TokenChunker, {}),
    "SemanticSpacyChunker" : (SemanticSpacyChunker, {}),
    "SemanticNLTKChunker" : (SemanticNLTKChunker, {}),
}