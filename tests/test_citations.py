import pytest
from backend.services.rag_pipeline import convert_citations_to_superscript

class MockDocument:
    def __init__(self, page_content: str, source: str, page: int, relevance_score: float = 0.9):
        self.page_content = page_content
        self.metadata = {
            "source": source,
            "page": page,
            "relevance_score": relevance_score
        }

def test_convert_citations_to_superscript():
    retrieved_docs = [
        MockDocument("This is doc 1 content", "c:/docs/iso27001.pdf", 10),
        MockDocument("This is doc 2 content", "c:/docs/tesla_revenue.pdf", 5),
        MockDocument("This is doc 3 content", "c:/docs/nist_framework.pdf", 1),
    ]
    
    # 1. Standard single citation
    text1 = "An ISMS is defined as a framework [Doc 1, Page 10]."
    clean1, sources1, count1 = convert_citations_to_superscript(text1, retrieved_docs)
    assert clean1.startswith("An ISMS is defined as a framework¹.")
    assert count1 == 1
    assert sources1[0]["source"] == "iso27001.pdf"
    assert sources1[0]["page"] == 10
    
    # 2. Compound semicolon citation
    text2 = "These objectives must be measurable [Doc 1, Page 10; Doc 2, Page 5]."
    clean2, sources2, count2 = convert_citations_to_superscript(text2, retrieved_docs)
    assert clean2.startswith("These objectives must be measurable¹².")
    assert count2 == 2
    assert sources2[0]["source"] == "iso27001.pdf"
    assert sources2[1]["source"] == "tesla_revenue.pdf"
    
    # 3. Partially invalid compound citation (some out of bounds)
    text3 = "Planning is key [Doc 1, Page 10; Doc 10, Page 5]."
    clean3, sources3, count3 = convert_citations_to_superscript(text3, retrieved_docs)
    assert clean3.startswith("Planning is key¹.")
    assert count3 == 1
    
    # 4. Completely invalid out of bounds citation (fully stripped)
    text4 = "Strategic direction [Doc 10, Page 5]."
    clean4, sources4, count4 = convert_citations_to_superscript(text4, retrieved_docs)
    assert clean4.startswith("Strategic direction.")
    
    # 5. PDF matching
    text5 = "Tesla revenue increased [tesla_revenue.pdf, Page 5]."
    clean5, sources5, count5 = convert_citations_to_superscript(text5, retrieved_docs)
    assert clean5.startswith("Tesla revenue increased¹.")
    assert count5 == 1
    
    # 6. Compound PDF and Doc
    text6 = "Comparison of standards [iso27001.pdf, Page 10; Doc 3, Page 1]."
    clean6, sources6, count6 = convert_citations_to_superscript(text6, retrieved_docs)
    assert clean6.startswith("Comparison of standards¹².")
    assert count6 == 2
    
    # 7. Normal bracket text containing no citations (must not be touched)
    text7 = "This is a normal bracket [and it should remain intact]."
    clean7, sources7, count7 = convert_citations_to_superscript(text7, retrieved_docs)
    assert clean7.startswith("This is a normal bracket [and it should remain intact].")
    
    # 8. Normal text containing "Doc" but not matching format (must not be touched)
    text8 = "Refer to instructions [see Doc 1 for details]."
    clean8, sources8, count8 = convert_citations_to_superscript(text8, retrieved_docs)
    assert clean8.startswith("Refer to instructions [see Doc 1 for details].")
