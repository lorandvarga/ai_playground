from crewai import Task


def create_analyze_documents_task(agent, documents_content):
    """Create the document analysis task."""
    return Task(
        description=(
            f"Analyze the following PDF documents and extract key information:\n\n"
            f"{documents_content}\n\n"
            "For each document:\n"
            "1. Summarize the main themes and key points\n"
            "2. Extract important data from any charts, graphs, or tables\n"
            "3. Identify trends, patterns, or notable statistics\n"
            "4. Highlight any significant findings or insights\n\n"
            "Provide a structured summary of each document with clear sections."
        ),
        expected_output=(
            "A comprehensive structured summary for each document containing:\n"
            "- Document name and overview\n"
            "- Main themes and key points\n"
            "- Data extracted from charts and visualizations\n"
            "- Notable trends or patterns identified\n"
            "- Key statistics and metrics"
        ),
        agent=agent
    )


def create_generate_insights_task(agent, context_tasks):
    """Create the insights generation task."""
    return Task(
        description=(
            "Based on the document analysis provided, generate strategic insights and recommendations.\n\n"
            "Your analysis should:\n"
            "1. Identify common themes across all documents\n"
            "2. Connect related findings from different documents\n"
            "3. Highlight the most important insights\n"
            "4. Identify potential opportunities or concerns\n"
            "5. Provide actionable recommendations based on the data\n\n"
            "Focus on delivering practical, data-driven insights that would be valuable for decision-making."
        ),
        expected_output=(
            "A comprehensive insights report containing:\n"
            "- Executive Summary (2-3 key takeaways)\n"
            "- Cross-document Analysis (themes and connections)\n"
            "- Key Insights (prioritized list with supporting data)\n"
            "- Strategic Recommendations (actionable next steps)\n"
            "- Supporting Evidence (references to specific data points from documents)"
        ),
        agent=agent,
        context=context_tasks
    )
