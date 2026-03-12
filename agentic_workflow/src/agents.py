from crewai import Agent


def create_document_analyzer():
    """Create the Document Analysis Specialist agent."""
    return Agent(
        role="Document Analysis Specialist",
        goal="Extract key information, themes, and data points from PDF documents including charts and graphs",
        backstory=(
            "You are an expert at reading and understanding various document types, "
            "with a special talent for interpreting visual data like charts, graphs, and tables. "
            "You can identify patterns, extract numerical data, and summarize complex information clearly."
        ),
        verbose=True,
        allow_delegation=False
    )


def create_insights_generator():
    """Create the Strategic Insights Analyst agent."""
    return Agent(
        role="Strategic Insights Analyst",
        goal="Synthesize information from multiple sources into actionable insights and recommendations",
        backstory=(
            "You are skilled at connecting dots between different pieces of information "
            "to generate valuable business insights. You excel at identifying trends, "
            "spotting opportunities, and providing strategic recommendations based on data analysis."
        ),
        verbose=True,
        allow_delegation=False
    )
