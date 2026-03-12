from crewai import Crew, Process
from src.agents import create_document_analyzer, create_insights_generator
from src.tasks import create_analyze_documents_task, create_generate_insights_task


def create_document_insights_crew(documents_content):
    """Create and configure the document insights crew."""

    # Create agents
    document_analyzer = create_document_analyzer()
    insights_generator = create_insights_generator()

    # Create tasks
    analyze_task = create_analyze_documents_task(document_analyzer, documents_content)
    insights_task = create_generate_insights_task(insights_generator, [analyze_task])

    # Create crew
    crew = Crew(
        agents=[document_analyzer, insights_generator],
        tasks=[analyze_task, insights_task],
        process=Process.sequential,
        verbose=True
    )

    return crew
