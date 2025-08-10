import os
import warnings
import logging
from dotenv import load_dotenv
from pprint import pprint
from config.settings import Settings
from src.workflow.workflow_builder import RAGWorkflowBuilder
from src.utils.logging_config import setup_logging

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    """Main application entry point"""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting RAG System...")
    
    try:
        # Load settings
        settings = Settings()
        
        # Build workflow
        workflow_builder = RAGWorkflowBuilder(settings)
        app = workflow_builder.build_workflow()
        
        # Test the system
        test_questions = [
            "What is prompt engineering?",
            "How to save LLM cost?",
            "What are the types of agent memory?",
            "When will the Euro of Football take place?"  # This should trigger web search
        ]
        
        for question in test_questions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing question: {question}")
            logger.info(f"{'='*60}")
            
            inputs = {"question": question}
            
            try:
                # Stream the workflow execution
                final_output = None
                for output in app.stream(inputs):
                    for key, value in output.items():
                        logger.info(f"Finished running: {key}")
                        final_output = value
                
                if final_output and "generation" in final_output:
                    logger.info(f"\nFinal Answer: {final_output['generation']}")
                    print(f"\nQuestion: {question}")
                    print(f"Answer: {final_output['generation']}\n")
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
    
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise

if __name__ == "__main__":
    main()